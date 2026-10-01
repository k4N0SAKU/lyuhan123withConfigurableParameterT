"""模式 B 端到端密文推理管线（P3 实现；P4 转换真实路径接线）。
BERT-base-chinese 情感分类，层数可截断。

架构（docs/01 §6.3 段划分）：密文域线性层（P2 packing）+ 非线性 MPC 域求值。
转换入口/出口为 **P4 真实路径**（docs/01 §7.4 / src/protocol/conversion.py）：
入口 = P2 密文域加 Enc(r) → P0 白名单①掩码解密 → y₁ 分发 P1、P2 持 −r；
出口 = P1 fresh 掩码 s → P2 密文域合成 fresh 顶层密文。三方以进程内角色直连
（S8 模拟口径——1.6MB 密文纯 Python GCM 序列化不可行，通道承载全程由
src/nodes e2e 以玩具参数验证）；MPC 中间仍为 reveal-compute-reshare 模拟
（CrypTFlow/Delphi 标准仿真方法），MpcEnv 如实记账门数与字节。
最终输出经 P0 白名单②解密（D5 执行点）；密钥分发：构造时一次密钥生成后
私钥仅驻留 ClientRole，管线计算侧只持 public_only 上下文。

P3-R1 结构修复（保留）：post-LN 两个 LayerNorm 转换点 + 每个线性段输入都是
转换出口的 fresh 双拷贝密文（docs/01 §7.4 出口编码免费重置）。
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from src.common.perf import (SEG_COMPUTE_LINEAR, SEG_COMPUTE_NONLINEAR,
                             SEG_ENCRYPT, SEG_THRESH_DECRYPT,
                             NetworkMeter, RoundContext)
from src.crypto.ckks_ops import CKKSContext, CKKSCiphertext, PARAMS_MODE_B
from src.crypto.secret_sharing import DEFAULT_MODULUS, share_vector
from src.model.loader import LABEL_MAP, BertSentimentPipeline
from src.model.ops.linear import embed_and_encrypt
from src.model.ops.packing import (DEFAULT_SPEC, PackingSpec,
                                   build_diagonal_plaintexts, layout_input,
                                   linear_cipher)
from src.model.ops.nonlinear_approx import MpcEnv, ref_gelu, ref_softmax_row
from src.protocol.conversion import ClientRole, InferRole, KeyRole, to_fixed


@dataclass
class PipelineConfig:
    n_layer: int = 2                 # 层数可截断（性能-精度曲线）
    seq_tokens: int = 2              # L（单密文 ≤ 10 块）
    gelu_variant: str = "gelu_deg15" # 精度-近似阶数曲线


@dataclass
class ConvertStats:
    conversions: int = 0
    masks_generated: int = 0
    net: NetworkMeter = field(default_factory=NetworkMeter)

    def snapshot(self) -> dict:
        return {"conversions": self.conversions, "masks": self.masks_generated,
                "net": self.net.snapshot()}


class ModeBPipeline:
    """模式 B 混合管线（截断 BERT，模拟层口径见模块 docstring）。"""

    def __init__(self, plain: BertSentimentPipeline,
                 cfg: Optional[PipelineConfig] = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.plain = plain
        self.model = plain.model
        self.tok = plain.tokenizer
        self.spec = DEFAULT_SPEC
        self.env = MpcEnv()
        self.stats = ConvertStats()
        self._diag_cache: dict = {}
        self._debug = os.environ.get("A122_DEBUG_INTERMEDIATE", "") == "1"

        # ---- P4 密钥分发（D5）：一次生成，私钥仅驻留 P0 角色 ----
        full_ctx = CKKSContext(PARAMS_MODE_B)
        self._keys_dir = tempfile.mkdtemp(prefix="a122_p4_keys_")
        full_ctx.save_keys(self._keys_dir, with_secret=True)
        del full_ctx                                    # 私钥仅经文件交给 P0 角色
        self.ctx = CKKSContext(PARAMS_MODE_B, public_only=True,
                               keys_dir=self._keys_dir)  # P2 计算侧（无私钥）
        self._decrypt_events: List[dict] = []

        def _audit(event: str, detail: dict) -> None:
            self._decrypt_events.append({"event": event, **detail})

        self.client = ClientRole(
            CKKSContext(PARAMS_MODE_B, public_only=False,
                        keys_dir=self._keys_dir), audit=_audit)
        # P1/P2 各持 public_only 材料（公钥相同，模拟内共享同一上下文实例）
        self.keynode = KeyRole(self.ctx)
        self.infer = InferRole(self.ctx)

    def _diags(self, W):
        key = id(W)
        if key not in self._diag_cache:
            self._diag_cache[key] = build_diagonal_plaintexts(W, self.spec)
        return self._diag_cache[key]

    def _convert_to_shares(self, ct: CKKSCiphertext,
                           ctx: RoundContext) -> List[int]:
        """转换入口（docs/01 §7.4 **真实路径**，P4 落地——D5 白名单①执行点）：
        P2 fresh 全宽网格掩码 r → ct′ = ct + Enc_pk(r)（**密文域**同态加）→
        P0 白名单门控解密 y = x + r → CONVERT_SHARE y₁ 给 P1；P2 持 −r。

        进程内角色直连（S8 模拟口径：通道承载全程由 src/nodes e2e 以玩具参数
        验证——1.6MB 密文纯 Python GCM 序列化开销不可行，见 P4 工作记录 §3）。
        数值语义：重构 y₁ + (−r) = x·2¹⁶（环算术精确），+e_ckks ≤1 ulp。"""
        with ctx.timer.segment("convert_mask_decrypt"):
            request_id = os.urandom(16)
            masked = self.infer.entry(ct, request_id)      # P2：ct+Enc(r)
            y1 = self.client.masked_decrypt(masked, request_id)  # P0：白名单门
            self.keynode.take_entry_share(request_id, y1)  # P1：收 y₁
            p2 = self.infer.entry_p2_share(request_id)     # P2：−r
        with ctx.timer.segment("convert_share"):
            vals = [(a + b) % DEFAULT_MODULUS for a, b in zip(y1, p2)]
            self.stats.conversions += 1
            self.stats.masks_generated += 1
            self.stats.net.on_send(ct.size_bytes + len(y1) * 8)
        return vals

    def _recrypt(self, real: List[float], ctx: RoundContext) -> CKKSCiphertext:
        """转换出口（docs/01 §7.4 **真实路径**再随机化）：P1 fresh 掩码 s →
        z = a₁ + s（OTP）与 Enc_pk(s) → P2 密文域合成 Enc(w) − Enc(s) = Enc(v)
        → fresh 顶层密文（深度重置）；布局由调用方的 real 排序承载。

        掩码窗 [2⁴⁶, 2⁵³−2⁴⁶)（float64 精确 + 无 mod 回绕，CrypTFlow 式溢出
        控制——P3 docstring 预告的 P4 工程点，论证见 conversion.py）。a₁/a₂
        为 MPC 打开阶段的结果分享（模拟层经 share_vector 重分享）。"""
        with ctx.timer.segment("convert_recrypt"):
            fixed = [to_fixed(v) for v in real]            # 值域守卫 |v|<2⁴⁶
            a1, a2 = share_vector(fixed)                   # S8 重分享
            request_id = os.urandom(16)
            self.keynode.take_entry_share(request_id, a1)  # P1 持结果分享 a₁
            z, enc_s = self.keynode.exit_prepare(request_id)   # P1：s, z, Enc(s)
            ct = self.infer.exit_compose(request_id, enc_s, z, a2)  # P2 密文域合成
            self.stats.conversions += 1
            self.stats.masks_generated += 1
            self.stats.net.on_send(len(enc_s) + len(z) * 8 + ct.size_bytes)
        return ct

    def _add_bias(self, ct: CKKSCiphertext, bias: "np.ndarray", L: int) -> CKKSCiphertext:
        """加偏置（transformers Linear 的 b）：按 token 块布局 add_plain（level 不变）。"""
        bias_layout = layout_input([bias for _ in range(L)], L, self.spec)
        return self.ctx.add_plain(ct, bias_layout)


    def _valid_blocks(self, ct: CKKSCiphertext, ctx: RoundContext,
                      L: int) -> "np.ndarray":
        """转换 + **按块提取有效槽**：gap-block 布局下每块前 d 槽为有效值，
        其余为 BSGS 垃圾——P3-R1 修复：初版 `[:L*d]` 连续切片把垃圾槽混入
        token 矩阵（L=2 即开始读垃圾行，是残差/分类漂移的根源）。"""
        d = self.spec.block_elems
        bs = self.spec.block_slots
        arr = np.array(_from_mod(self._convert_to_shares(ct, ctx)))
        return np.stack([arr[b * bs:b * bs + d] for b in range(L)])

    def _layer(self, x_ct: CKKSCiphertext, x_vals: "np.ndarray", li: int,
               ctx: RoundContext) -> "np.ndarray":
        """单层（post-LN BERT，docs/01 §4.3 模式 B 时序）：

        QKV 密文线性（fresh level0→1）→ 转换 → MPC 注意力 → 转换出口重加密
        → proj 线性 → 转换 → 残差加（MPC 免费加法）+ LN1（转换点）→ FFN1
        → 转换 → GELU → FFN2 → 转换 → 残差加 + LN2 → 下一层输入（values）。

        布局不变量（P3-R1 修复）：每个线性段的输入都是转换出口的 fresh
        **双拷贝**密文（docs/01 §7.4 出口编码免费重置）——初版实现让残差加的
        输出直接进入下一个线性，破坏双拷贝（proj 后半槽 BSGS 垃圾 + 残差拷贝
        ≠ 和的拷贝），FFN1 环绕项读到垃圾（实测 err 0.21 / 全层 1e13）。"""
        layer = self.model.bert.encoder.layer[li]
        L = self.cfg.seq_tokens
        d = self.spec.block_elems
        head = d // 12

        # --- 段 1（密文，level0→1）：Q/K/V 三线性（含 bias）---
        with ctx.timer.segment(SEG_COMPUTE_LINEAR):
            W_q = layer.attention.self.query.weight.detach().numpy()
            W_k = layer.attention.self.key.weight.detach().numpy()
            W_v = layer.attention.self.value.weight.detach().numpy()
            q_ct = self._add_bias(
                linear_cipher(self.ctx, x_ct, W_q, L, self.spec,
                              diagonals=self._diags(W_q)),
                layer.attention.self.query.bias.detach().numpy(), L)
            k_ct = self._add_bias(
                linear_cipher(self.ctx, x_ct, W_k, L, self.spec,
                              diagonals=self._diags(W_k)),
                layer.attention.self.key.bias.detach().numpy(), L)
            v_ct = self._add_bias(
                linear_cipher(self.ctx, x_ct, W_v, L, self.spec,
                              diagonals=self._diags(W_v)),
                layer.attention.self.value.bias.detach().numpy(), L)

        # --- 转换 + MPC 域注意力（reveal-compute-reshare）---
        with ctx.timer.segment(SEG_COMPUTE_NONLINEAR):
            q_f = self._valid_blocks(q_ct, ctx, L)
            k_f = self._valid_blocks(k_ct, ctx, L)
            v_f = self._valid_blocks(v_ct, ctx, L)
            attn_out = np.zeros((L, d))
            for h in range(12):
                q_h = q_f[:, h * head:(h + 1) * head]
                k_h = k_f[:, h * head:(h + 1) * head]
                v_h = v_f[:, h * head:(h + 1) * head]
                scores = q_h @ k_h.T / np.sqrt(head)
                probs = ref_softmax_row(scores - 8.0)   # 公开移位（C-S 界 |score|≤8）
                attn_out[:, h * head:(h + 1) * head] = probs @ v_h
            # 门数记账（公开移位口径）：QK^T/·V 各 L²·head/头 + softmax
            # 每行 exp 4/元素×L + inv(1+2×12)=25 → 12·L·(4L+25)
            self.env.gates_used += 12 * (2 * L * L * head + L * (4 * L + 25))
            self.env.comm_bytes += 12 * L * (4 * L + 25) * 32

        # --- 段 2（密文）：proj 线性（输入 = 转换出口 fresh 双拷贝）---
        with ctx.timer.segment(SEG_COMPUTE_LINEAR):
            attn_ct = self._recrypt(
                layout_input([attn_out[i] for i in range(L)], L, self.spec), ctx)
            W_proj = layer.attention.output.dense.weight.detach().numpy()
            proj_ct = self._add_bias(
                linear_cipher(self.ctx, attn_ct, W_proj, L, self.spec,
                              diagonals=self._diags(W_proj)),
                layer.attention.output.dense.bias.detach().numpy(), L)

        # --- 转换：proj 揭示 → 残差加（免费）→ LN1（门记账）→ 出口重加密 ---
        with ctx.timer.segment(SEG_COMPUTE_NONLINEAR):
            proj_vals = self._valid_blocks(proj_ct, ctx, L)
            h_raw = proj_vals + x_vals
            with torch.inference_mode():
                h = layer.attention.output.LayerNorm(
                    torch.tensor(h_raw, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            self.env.gates_used += h.size * 8            # docs/01 §7.3 LN 门预算
            self.env.comm_bytes += h.size * 8 * 32
        with ctx.timer.segment(SEG_ENCRYPT):
            h_ct = self._recrypt(
                layout_input([h[i] for i in range(L)], L, self.spec), ctx)

        # --- 段 3（密文）：FFN1（截断 768→768，模拟口径）---
        with ctx.timer.segment(SEG_COMPUTE_LINEAR):
            W_ffn1 = layer.intermediate.dense.weight.detach().numpy()[:d, :]
            ffn1_ct = self._add_bias(
                linear_cipher(self.ctx, h_ct, W_ffn1, L, self.spec,
                              diagonals=self._diags(W_ffn1)),
                layer.intermediate.dense.bias.detach().numpy()[:d], L)

        # --- 转换 + GELU ---
        with ctx.timer.segment(SEG_COMPUTE_NONLINEAR):
            ffn1_vals = self._valid_blocks(ffn1_ct, ctx, L)
            gelu_f = ref_gelu(ffn1_vals)
            self.env.gates_used += gelu_f.size * 13      # deg15（P2-R1 校准）
            self.env.comm_bytes += gelu_f.size * 13 * 32
        with ctx.timer.segment(SEG_ENCRYPT):
            gelu_ct = self._recrypt(
                layout_input([gelu_f[i] for i in range(L)], L, self.spec), ctx)

        # --- 段 4（密文）：FFN2 ---
        with ctx.timer.segment(SEG_COMPUTE_LINEAR):
            W_ffn2 = layer.output.dense.weight.detach().numpy()[:, :d]
            ffn2_ct = self._add_bias(
                linear_cipher(self.ctx, gelu_ct, W_ffn2, L, self.spec,
                              diagonals=self._diags(W_ffn2)),
                layer.output.dense.bias.detach().numpy(), L)

        # --- 转换：残差加 + LN2 → 下一层输入 ---
        with ctx.timer.segment(SEG_COMPUTE_NONLINEAR):
            ffn2_vals = self._valid_blocks(ffn2_ct, ctx, L)
            z_raw = ffn2_vals + h
            with torch.inference_mode():
                z = layer.output.LayerNorm(
                    torch.tensor(z_raw, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            self.env.gates_used += z.size * 8
            self.env.comm_bytes += z.size * 8 * 32
        return z

    def classify(self, text: str, ctx: Optional[RoundContext] = None) -> dict:
        """端到端：嵌入（P0 本地，含位置/LN）→ N 层密文管线 → CLS 隐藏态
        → 明文池化/分类头。最终隐藏态为 LN2 转换出口值（P0 白名单内，D5）。"""
        ctx = ctx or RoundContext(index=0)
        enc = self.tok(text, truncation=True, max_length=self.cfg.seq_tokens,
                       return_tensors="pt")
        t0 = time.perf_counter()
        with ctx.timer.segment(SEG_ENCRYPT):
            with torch.inference_mode():
                x = self.model.bert.embeddings(enc["input_ids"])  # (1,L,d) 含位置/LN
            x_vals = x.squeeze(0).numpy()
            x_ct = self.ctx.encrypt_vector(
                layout_input([x_vals[i] for i in range(self.cfg.seq_tokens)],
                             self.cfg.seq_tokens, self.spec))
        for li in range(self.cfg.n_layer):
            x_vals = self._layer(x_ct, x_vals, li, ctx)
            if li < self.cfg.n_layer - 1:
                with ctx.timer.segment(SEG_ENCRYPT):
                    x_ct = self._recrypt(
                        layout_input([x_vals[i] for i in range(self.cfg.seq_tokens)],
                                     self.cfg.seq_tokens, self.spec), ctx)
        with ctx.timer.segment(SEG_THRESH_DECRYPT):
            if self.cfg.n_layer > 0:
                # D5 白名单②执行点：LN2 出口分享 → 出口重加密（fresh ct）→
                # P0 最终解密。真实协议中即 INFER_RESULT(P2→P0) 的载荷。
                cls_ct = self._recrypt(
                    layout_input([x_vals[i] for i in range(self.cfg.seq_tokens)],
                                 self.cfg.seq_tokens, self.spec), ctx)
                final_vals = self.client.final_decrypt(cls_ct, os.urandom(16))
                cls = final_vals[:self.spec.block_elems]   # 块 0 有效槽 = token 0
            else:
                cls = x_vals[0]
        with torch.inference_mode():
            hidden = torch.tensor(cls, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            pooled = self.model.bert.pooler(hidden)
            logits = self.model.classifier(pooled)
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax())
        return {"label": LABEL_MAP[idx], "prob": float(probs[idx]),
                "conversions": self.stats.conversions,
                "mpc_gates": self.env.gates_used,
                "mpc_comm_bytes": self.env.comm_bytes,
                "segments_ms": {n: sum(v) / len(v)
                                for n, v in ctx.timer.samples_ms().items()},
                "wall_s": time.perf_counter() - t0}


def _from_mod(vals: List[int]) -> List[float]:
    out = []
    for v in vals:
        s = v % DEFAULT_MODULUS
        if s > DEFAULT_MODULUS // 2:
            s -= DEFAULT_MODULUS
        out.append(s / (1 << 16))
    return out
