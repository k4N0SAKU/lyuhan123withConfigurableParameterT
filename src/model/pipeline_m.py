"""模式 B m 方参数化管线（P8；ModeBPipeline 的门限族子类，零改动原文件）。

与 ModeBPipeline 的差异仅在两个转换点（_convert_to_shares / _recrypt）：
- 入口：掩码链 P2→P3→…→Pm→P1 各加自己的 Enc(rᵢ) → P0 白名单①解密 y →
  锚点 P1 份额 y−r₁、他方 −rᵢ（conversion_m.ComputePartyRole）；
- 出口：m-of-m 结果分享，各方 (zᵢ=aᵢ+sᵢ, Enc(sᵢ))，合成方密文域求和合成
  （exit_compose_m）。
MPC 中间仍为 reveal-compute-reshare 模拟（S8 口径不变）；安全语义：
任意 t ≤ m−1 个计算方合谋缺至少一片均匀掩码 ⇒ 信息论 OTP（P0 信任根）。

m=2 时与 ModeBPipeline 数值等价（重构精确一致；等价性测试见
tests/e2e/test_pipeline_m_equivalence.py）。密钥仪式与父类同构（参数可注入
以便测试贯通；上游 ModeBPipeline.__init__ 改动需同步本文件——结构重复已
在两侧 docstring 声明）。n_compute 上限由 ConversionPolicy 的 float64 2⁵³
墙强制（mode-b ⇒ 15，conversion_m.check_m）。
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Optional

from src.common.perf import RoundContext
from src.crypto.ckks_ops import CKKSContext, CKKSCiphertext, PARAMS_MODE_B
from src.crypto.secret_sharing_m import share_vector_n
from src.model.ops.nonlinear_approx import MpcEnv
from src.model.ops.packing import DEFAULT_SPEC
from src.model.pipeline import (ConvertStats, ModeBPipeline, PipelineConfig,
                                _from_mod)
from src.protocol.conversion import (ClientRole, policy_for, to_fixed)
from src.protocol.conversion_m import ComputePartyRole, check_m, exit_compose_m


class PipelineConfigM(PipelineConfig):
    """m 方管线配置：n_compute = 计算方数 m（P1..Pm），t = m−1。"""

    n_compute: int = 2


class ModeBPipelineM(ModeBPipeline):
    """m 方模式 B 管线：转换点走 D5′ 协议族，其余段继承父类。"""

    def __init__(self, plain, cfg: Optional[PipelineConfigM] = None,
                 params=None) -> None:
        self.cfg = cfg or PipelineConfigM()
        self.plain = plain
        self.model = plain.model
        self.tok = plain.tokenizer
        self.spec = DEFAULT_SPEC
        self.env = MpcEnv()
        self.stats = ConvertStats()
        self._diag_cache: dict = {}
        self._debug = os.environ.get("A122_DEBUG_INTERMEDIATE", "") == "1"
        self.m = self.cfg.n_compute

        # ---- 密钥分发（D5′ 延续）：一次生成，私钥仅驻留 P0 角色 ----
        params = params or PARAMS_MODE_B
        full_ctx = CKKSContext(params)
        self._keys_dir = tempfile.mkdtemp(prefix="a122_p8_keys_")
        full_ctx.save_keys(self._keys_dir, with_secret=True)
        del full_ctx                                    # 私钥仅经文件交给 P0
        self.ctx = CKKSContext(params, public_only=True,
                               keys_dir=self._keys_dir)  # 计算侧（无私钥）
        self._decrypt_events: List[dict] = []

        def _audit(event: str, detail: dict) -> None:
            self._decrypt_events.append({"event": event, **detail})

        self.client = ClientRole(
            CKKSContext(params, public_only=False,
                        keys_dir=self._keys_dir), audit=_audit)
        check_m(policy_for(self.ctx), self.m)           # float64 2⁵³ 墙强制
        # 计算方 P1..Pm：P1=锚点（收 y），P2=合成方（出口汇总），其余=掩码方
        self.parties = [ComputePartyRole(self.ctx, index=i + 1, m=self.m,
                                         is_anchor=(i == 0), audit=_audit)
                        for i in range(self.m)]

    def _entry_order(self) -> List[ComputePartyRole]:
        """掩码链调用序：P2→P3→…→Pm→P1（锚点最后加片并转投 P0）。"""
        return self.parties[1:] + self.parties[:1]

    def _convert_to_shares(self, ct: CKKSCiphertext,
                           ctx: RoundContext) -> List[int]:
        """D5′ 入口（白名单①执行点）：链式 m 片掩码 → P0 解密 y → m 方份额。

        数值语义：Σ份额 = y − Σrᵢ = x（环算术精确）；掩码链每跳转发
        masked ct（net 记 m 跳密文字节 + y 分发一次）。"""
        with ctx.timer.segment("convert_mask_decrypt"):
            request_id = os.urandom(16)
            masked = ct
            for p in self._entry_order():
                masked = p.entry_apply_piece(masked, request_id)
            y = self.client.masked_decrypt(masked, request_id)
        with ctx.timer.segment("convert_share"):
            shares = [p.entry_own_share(request_id,
                                        y=y if p.is_anchor else None)
                      for p in self.parties]
            vals = [sum(col) % (1 << 64) for col in zip(*shares)]
            self.stats.conversions += 1
            self.stats.masks_generated += self.m
            self.stats.net.on_send(ct.size_bytes * self.m + len(y) * 8)
        return vals

    def _recrypt(self, real: List[float], ctx: RoundContext) -> CKKSCiphertext:
        """D5′ 出口（再随机化）：m-of-m 结果分享 → 各片 (zᵢ, Enc(sᵢ)) →
        合成方 Enc(w) − ΣEnc(sᵢ) = Enc(v)，fresh 顶层密文（深度重置）。"""
        with ctx.timer.segment("convert_recrypt"):
            fixed = [to_fixed(v) for v in real]         # 值域守卫
            shares = share_vector_n(fixed, self.m)
            request_id = os.urandom(16)
            pieces = []
            for p, a in zip(self.parties, shares):
                p.set_result_share(request_id, a)
                pieces.append(p.exit_piece(request_id))
            ct = exit_compose_m(self.ctx, pieces)
            self.stats.conversions += 1
            self.stats.masks_generated += self.m
            self.stats.net.on_send(
                sum(len(z) for z, _ in pieces) * 8
                + sum(len(b) for _, b in pieces) + ct.size_bytes)
        return ct
