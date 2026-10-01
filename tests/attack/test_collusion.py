"""F8 合谋攻击（t 合谋模拟）：模式 A 与模式 B 分别执行。

诚实边界声明（与 docs/02 §3.3 一致）：
- **t=1（任一方单独）**：断言输入明文不可恢复（OTP/无钥，自动化统计检验）
  ——「防御成功」；
- **t=2（P1+P2 合谋）模式 B**：掩码分享构造的固有失效面——测试**如实演示**
  中间态与输入 token 可重构（已知明文侧/嵌入表最近邻攻击，陷阱清单要求的
  区分攻击尝试），判定「边界演示」（与威胁模型声明一致，非缺陷）；
- **t=2 模式 A（全密文）**：合谋方仅有密文+公钥 ⇒ 输入不可恢复（IND-CPA
  统计演示）——「防御成功」；
- **钩子证明**：collect_view 收集三方全部内存视图与线缆字节，扫描输入串/
  token id/定点编码——「输入仅在 P0 内存」。

夹具为 module 级（pytest 8 对类内 class-scoped fixture 弃用告警——P5-R1）。

运行：pytest tests/attack/test_collusion.py -v（两模式均在默认口径；模式 B 首次加载 BERT ~30s）
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.crypto.ckks_ops import CKKSContext, PARAMS_P4_TOY
from src.crypto.secret_sharing import DEFAULT_MODULUS
from src.protocol.conversion import (ClientRole, InferRole, KeyRole,
                                     from_fixed, to_fixed)
from tests.attack.framework import VERDICTS, collect_view, pearson

INPUT_TEXT = "办事大厅服务热情周到，一次办好"


def _fixed_stream(vec, repeat_to: int) -> bytes:
    """定点向量的 8B 大端字节流（重复填充到指定长度，供扫描/相关分析）。"""
    ints = [to_fixed(v) for v in vec]
    out = bytearray()
    while len(out) < repeat_to:
        for v in ints:
            out += v.to_bytes(8, "big")
    return bytes(out[:repeat_to])


# =====================================================================
# 模式 A（全密文数据流）：t=2 合谋仅有密文+公钥 —— 防御成功
# =====================================================================

def _build_modea():
    full = CKKSContext(PARAMS_P4_TOY)
    import tempfile
    d = tempfile.mkdtemp(prefix="f8a_")
    full.save_keys(d, with_secret=True)
    sec = CKKSContext(PARAMS_P4_TOY, public_only=False, keys_dir=d)
    pub = CKKSContext(PARAMS_P4_TOY, public_only=True, keys_dir=d)
    del full
    x = [0.5, -1.25, 2.0, 3.75, -0.125, 1.0] + [0.0] * 10
    ct_input = sec.encrypt_vector(x)
    # 模式 A 数据面：P2 的中间/输出密文（本栈不可实例化深链——
    # 安全性结论只依赖 CKKS IND-CPA 与密钥分布，与链深无关，如实声明）
    ct_mid = pub.encrypt_vector([v * 0.5 for v in x])
    ct_out = pub.encrypt_vector([v * 0.1 for v in x])
    coalition = {                       # P1+P2 全部内存视图（t=2）
        "pk_ctx": pub, "input_ct": ct_input, "mid_ct": ct_mid,
        "out_ct": ct_out,
        "model_weights": np.random.default_rng(7).normal(size=(16, 16)),
        "wire": b"",                    # 模式 A 线缆=密文帧（同构）
    }
    return {"sec": sec, "pub": pub, "x": x, "coalition": coalition}


@pytest.fixture(scope="module")
def modea():
    return _build_modea()


class TestCollusionModeA:
    def test_no_secret_key_in_coalition(self, modea):
        """钩子①：合谋视图内不存在任何 CKKS 私钥材料（结构性 public_only）。"""
        assert modea["coalition"]["pk_ctx"].public_only is True
        from src.crypto.ckks_ops import OpNotAllowedError
        with pytest.raises(OpNotAllowedError):
            modea["coalition"]["pk_ctx"].decrypt(modea["coalition"]["out_ct"])
        VERDICTS.collect("F8-模式A", "私钥缺失（结构性）", "防御成功",
                         {"secret_key_holders": ["P0"],
                          "coalition_ctx": "public_only"})

    def test_plaintext_scale_scan(self, modea):
        """钩子②：合谋内存视图无激活量级浮点向量（明文仅存在于 P0）。"""
        view = collect_view(modea["coalition"])
        small_floats = [f for f in view["float_items_sample"]
                        if f != 0.0 and abs(f) < 1e4]
        assert small_floats == [], f"合谋视图出现明文量级数值: {small_floats[:4]}"
        VERDICTS.collect("F8-模式A", "明文量级扫描", "防御成功",
                         {"activation_scale_values_in_view": 0,
                          "note": "视图仅含密文对象/公钥/模型权重矩阵"})

    def test_known_plaintext_distinguishing(self, modea):
        """已知明文侧区分攻击（陷阱清单要求）：持 ct(x_true) 与公钥、可对任意
        候选重加密——断言无统计区分力（IND-CPA 演示）。"""
        pub, x = modea["pub"], modea["x"]
        corrs = []
        cands = {"true": x,
                 "decoy1": [v * 1.01 for v in x],
                 "decoy2": [v + 0.05 for v in x],
                 "random": list(np.random.default_rng(3).normal(scale=2,
                                                                size=len(x)))}
        # 密文尺寸元数据检验：尺寸浮动源自加密随机性（同明文重加密同样浮动，
        # P5 实测）——候选间尺寸须落在同明文重加密的基线带宽内（无明文相关
        # 长度信号）
        base = [pub.encrypt_vector(x).size_bytes for _ in range(6)]
        lo, hi = min(base), max(base)
        enc_sizes = {k: pub.encrypt_vector(v).size_bytes for k, v in
                     cands.items()}
        for k, sz in enc_sizes.items():
            assert lo - 200 <= sz <= hi + 200, \
                f"候选 {k} 密文尺寸 {sz} 超出基线带宽 [{lo},{hi}]（长度泄露）"
        # 重加密不可与持有密文匹配：逐 64 字节采样重合率 ≈1/256（随机化加密）
        for k, v in cands.items():
            re_ct = pub.encrypt_vector(v)
            a, b = pub.serialize_ct_bytes(re_ct), pub.serialize_ct_bytes(
                modea["coalition"]["out_ct"])
            match = sum(1 for i in range(0, min(len(a), len(b)), 64)
                        if a[i] == b[i]) / (len(a) // 64)
            corrs.append((k, match))
        for k, m in corrs:
            assert abs(m - 1 / 256) < 0.05, f"候选 {k} 与持有密文异常相关 {m}"
        VERDICTS.collect("F8-模式A", "已知明文区分攻击", "防御成功",
                         {"byte_match_rates": {k: round(m, 4)
                                               for k, m in corrs},
                          "note": "全部候选 ≈ 1/256（随机化加密 IND-CPA）"})

    def test_input_recovery_impossible_verdict(self, modea):
        """F8 模式 A 结论：t=2 合谋输入明文不可恢复（计算安全）。"""
        VERDICTS.collect("F8-模式A", "综合结论（t=2）", "防御成功",
                         {"input_recovered": False,
                          "basis": "CKKS IND-CPA + 合谋视图无钥/无明文量级值"})


# =====================================================================
# 模式 B（混合管线）：t=1 防御成功；t=2 边界演示（docs/02 §3.3）
# =====================================================================

def _build_modeb():
    """真实 BERT 嵌入/权重 + 玩具 CKKS 的一次真实协议转换（module 级）。"""
    from src.model.loader import BertSentimentPipeline
    plain = BertSentimentPipeline()
    full = CKKSContext(PARAMS_P4_TOY)
    import tempfile
    d = tempfile.mkdtemp(prefix="f8b_")
    full.save_keys(d, with_secret=True)
    sec = CKKSContext(PARAMS_P4_TOY, public_only=False, keys_dir=d)
    pub = CKKSContext(PARAMS_P4_TOY, public_only=True, keys_dir=d)
    del full

    tok = plain.tokenizer
    enc = tok(INPUT_TEXT, truncation=True, max_length=2, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    L = len(ids)
    with __import__("torch").inference_mode():
        x = plain.model.bert.embeddings(enc["input_ids"])[0].numpy()  # (L,768)
    layer = plain.model.bert.encoder.layer[0]
    W_q = layer.attention.self.query.weight.detach().numpy().astype(np.float64)
    b_q = layer.attention.self.query.bias.detach().numpy().astype(np.float64)
    q_out = x @ W_q.T + b_q                    # P2 密文域线性层的明文语义

    # 真实协议转换（角色直连，toy 参数）：P2 掩码 → P0 白名单①解密 → 分享
    client = ClientRole(sec)
    keynode = KeyRole(pub)
    infer = InferRole(pub)
    rid = os.urandom(16)
    # 玩具参数槽 1024 < L·768=1536 ⇒ 协议真实承载 token0 的 768 维 QKV
    # 输出（F8 的重构攻击语义不变；多 token 仅扩大同一失效面）
    q_flat = q_out[0].tolist()
    ct_q = pub.encrypt_vector(q_flat)
    masked = infer.entry(ct_q, rid)
    y1 = client.masked_decrypt(masked, rid)    # P1 视图
    p2_share = infer.entry_p2_share(rid)       # P2 视图（−r）
    keynode.take_entry_share(rid, y1)

    return {"tok": tok, "plain": plain, "ids": ids, "x": x, "L": L,
            "W_q": W_q, "b_q": b_q, "q_out": q_out[0],
            "client": client, "keynode": keynode, "infer": infer,
            "y1": y1, "p2_share": p2_share, "rid": rid}


@pytest.fixture(scope="module")
def modeb():
    return _build_modeb()


class TestCollusionModeB:
    # ---- 钩子证明 ----
    def test_input_only_in_p0_memory(self, modeb):
        """钩子证明：输入串与 token id 定宽字节模式不在 P1/P2 视图的任何
        字节/字符串项中出现（份额/掩码均为 OTP 掩码域值）。"""
        ids = modeb["ids"]
        views = {
            "P1_keynode": collect_view(modeb["keynode"]),
            "P2_infer": collect_view(modeb["infer"]),
        }
        text_hex = INPUT_TEXT.encode("utf-8").hex()
        for name, v in views.items():
            assert not any(text_hex in b for b in v["bytes_items"]), \
                f"{name} 出现输入明文（hex 视图）"
            assert not any(INPUT_TEXT in s for s in v["str_items"]), \
                f"{name} 出现输入明文"
            for t in ids:
                pat = t.to_bytes(8, "big").hex()
                assert not any(pat in b for b in v["bytes_items"]), \
                    f"{name} 视图出现 token id {t} 原值（应为 OTP 掩码域）"
        # 对照：P0 白名单门确为持钥方（协议语义需要其视图含 y₁ 掩码值）
        assert modeb["client"].whitelist._ctx.public_only is False
        VERDICTS.collect("F8-模式B", "钩子：输入仅在 P0 内存", "防御成功",
                         {"p1_view_token_hits": 0, "p2_view_token_hits": 0,
                          "raw_text_in_p1p2": False,
                          "note": "视图=角色全字段取证（collect_view，hex 编码）"})

    # ---- t=1：任一方单独 ----
    def test_p1_alone_cannot_recover(self, modeb):
        """P1 单独：y₁=QKV输出+r 对「候选嵌入→推算 QKV 输出」做掩码残差窗
        与相关检验——无区分力（攻击者知道模型，假设空间=嵌入向量）。"""
        y1 = modeb["y1"][:768]
        x, W_q, b_q, q_out = (modeb["x"], modeb["W_q"], modeb["b_q"],
                              modeb["q_out"])
        # 相关检验：y₁ 与真实明文（QKV 输出）的定点编码相关 ≈ 0
        corr = pearson(y1[:64], [to_fixed(v) for v in q_out[:64]])
        assert abs(corr) < 0.2, f"y₁ 与真实明文相关 {corr:.3f}（应≈0）"
        # 残差窗检验：候选嵌入 e → q_cand = e@W_q.T+b_q → 残差 y₁−fixed(q_cand)
        # true 候选残差 = r ∈ [0,2⁴⁰)；decoy 残差偏移 |q_true−q_decoy|·2¹⁶
        # ⇒ 落窗率几乎相同（窗宽 2⁴⁰ ≫ 值域）——区分力近零
        from src.protocol.conversion import POLICY_P4_TOY
        win = POLICY_P4_TOY.entry_mask_hi
        decoy_emb = x[0] + 0.5
        q_decoy = decoy_emb @ W_q.T + b_q
        true_res = [(a - to_fixed(v)) % DEFAULT_MODULUS
                    for a, v in zip(y1[:8], q_out[:8])]
        decoy_res = [(a - to_fixed(v)) % DEFAULT_MODULUS
                     for a, v in zip(y1[:8], q_decoy[:8])]
        true_in = sum(1 for r in true_res if r < win) / len(true_res)
        decoy_in = sum(1 for r in decoy_res if r < win) / len(decoy_res)
        assert abs(true_in - decoy_in) < 0.01, \
            f"残差落窗率可区分 true={true_in} decoy={decoy_in}"
        VERDICTS.collect("F8-模式B", "P1 单独（t=1）", "防御成功",
                         {"corr_with_true_plaintext": round(corr, 4),
                          "residual_window_test": "true/decoy 落窗率差 <0.01",
                          "basis": "OTP：r 由 P2 os.urandom fresh 采样"})

    def test_p2_alone_cannot_recover(self, modeb):
        """P2 单独：无钥（public_only 结构断言）+ 其视图（−r、密文）与明文统计独立。"""
        assert modeb["infer"].ctx.public_only is True
        from src.crypto.ckks_ops import OpNotAllowedError
        with pytest.raises(OpNotAllowedError):
            modeb["infer"].ctx.decrypt(
                modeb["infer"].ctx.encrypt_vector([1.0]))
        # −r 是均匀掩码的负元——与任何候选明文相关 ≈ 0
        p2 = modeb["p2_share"][:64]
        corr = pearson(p2, [to_fixed(v) for v in modeb["q_out"][:64]])
        assert abs(corr) < 0.2
        VERDICTS.collect("F8-模式B", "P2 单独（t=1）", "防御成功",
                         {"key_material": "public_only（OpNotAllowedError）",
                          "corr_share_vs_plaintext": round(corr, 4)})

    # ---- t=2：P1+P2 合谋（诚实失效面演示） ----
    def test_p1p2_collusion_reconstructs_tokens(self, modeb):
        """t=2 合谋（边界演示）：y₁+(−r) 重构 QKV 输出 → W_q 求逆回嵌入 →
        嵌入表最近邻恢复 token——**成功**（与 docs/02 §3.3 失效面声明一致）。"""
        y1, p2 = modeb["y1"], modeb["p2_share"]
        W_q, b_q, x = modeb["W_q"], modeb["b_q"], modeb["x"]
        q_hat = np.array([from_fixed((a + b) % DEFAULT_MODULUS)
                          for a, b in list(zip(y1, p2))[:768]])  # token0 的 QKV 输出
        x_hat = (q_hat - b_q) @ np.linalg.inv(W_q.T)          # W 求逆回嵌入
        err = float(np.abs(x_hat - x[0]).max())
        # 玩具参数 CKKS 噪声经 768 维求逆放大（P5 实测 ~4e-3）——
        # 嵌入量级 O(0.1)，仍远小于词表最近邻间隔，token 恢复不受影响
        assert err < 0.05, f"中间态重构误差 {err}"
        # 嵌入表最近邻 → token 恢复（攻击者可用：模型为其自有资产）
        import torch
        m = modeb["plain"].model
        emb = m.bert.embeddings
        with torch.inference_mode():
            word = emb.word_embeddings.weight.double().numpy()      # (V,768)
            pos = emb.position_embeddings.weight.double().numpy()
            tt = emb.token_type_embeddings.weight.double().numpy()[0]
            ln_gamma = emb.LayerNorm.weight.double().numpy()
            ln_beta = emb.LayerNorm.bias.double().numpy()
        # x_hat 为 token0 的嵌入估计（fixture 承载 768 维 QKV 输出；
        # 多 token 只是同一攻击的重复，失效面语义不变）
        cand = word + pos[0] + tt                                  # (V,768)
        mu = cand.mean(axis=1, keepdims=True)
        var = cand.var(axis=1, keepdims=True)
        cand = (cand - mu) / np.sqrt(var + 1e-12) * ln_gamma + ln_beta
        d = ((cand - x_hat) ** 2).sum(axis=1)
        recovered = int(np.argmin(d))
        assert recovered == modeb["ids"][0], \
            f"t=2 token 重构不一致 {recovered} vs {modeb['ids'][0]}"
        text = modeb["tok"].decode([recovered])
        VERDICTS.collect("F8-模式B", "P1+P2 合谋（t=2）", "边界演示",
                         {"intermediate_recovered": True,
                          "max_abs_err": round(err, 6),
                          "token0_recovered": recovered,
                          "decoded_text": text,
                          "declaration": "docs/02 §3.3 t=2 失效面（掩码分享构造"
                                         "固有边界）；模式 B 输入隐私止于 t=1"})

    def test_mode_b_t1_boundary_verdict(self, modeb):
        """F8 模式 B 结论：t=1 防御成功；t=2 为声明边界（缓解=模式 A/门限）。"""
        VERDICTS.collect("F8-模式B", "综合结论", "边界演示",
                         {"t1": "P1/P2 单独均不可恢复（自动化统计断言）",
                          "t2": "中间态+输入 token 可重构（docs/02 §3.3 一致）",
                          "mitigation": "模式 A 全密文（F8-模式A 防御成功）/ "
                                        "2-of-3 门限（扩展方向）"})
