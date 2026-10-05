"""D5′ 参数化 m 方转换协议族（P8；t = m−1，计算方间门限，P0 信任根）。

与 conversion.py（2 方单片 OTP）的关系：把「整片掩码」泛化为「m 片掩码」——
m 个计算方各自独持 fresh 掩码片，入口链式同态叠加、出口各片公钥加密后同态
求和。任意 t ≤ m−1 个计算方合谋至少缺一片均匀掩码 ⇒ 信息论 OTP 不破
（docs/02 §3.3′ 修订论证）。P0（数据属主/钥持者）为信任根：{P0, ·} 组合
超出声明模型，留置已知限制如实申报（不隐藏）。

协议（m 个计算方 P1..Pm；P1=锚点（密钥节点语义），P2=合成方（推理节点语义））：
- 入口（ct → m 份额）：掩码链 P2→P3→…→Pm→P1（各自 ct += Enc(rᵢ)，
  rᵢ 仅自己知）→ P0 白名单①解密 y = x + Σrᵢ → P0 将 y 分发锚点 P1
  （ConvertSharePayload holder=1）；
  份额：x₁ = y − r₁（锚点），xᵢ = −rᵢ（i ≥ 2）⇒ Σxᵢ = x (mod 2⁶⁴)。
- 出口（m 份额 → ct）：各 Pᵢ 持结果分享 aᵢ（Σaᵢ = v），采样 sᵢ，发
  (zᵢ = aᵢ + sᵢ, Enc(sᵢ)) 给合成方 P2；P2：w = Σzᵢ = v + Σsᵢ（OTP 域，
  m 缩放界守卫）→ ct = Enc(w) − ΣEnc(sᵢ) = Enc(v)，fresh 顶层密文。

窗与量级（每片复用单片版 ConversionPolicy 同窗；总窗按 m 线性放大）：
- 入口 y = x + Σrᵢ ≤ m·entry_hi + value_bound ≤ 2⁵³（float64 精确整数域）
  ⇒ mode-b m ≤ 15、p4-toy m ≤ 8191（:func:`m_cap` 强制）；
- 出口 w = v + Σsᵢ ≤ m·exit_hi + value_bound，同一 2⁵³ 界；SEAL 编码幅值
  随 m 线性（mode-b m=15: coeff ≈ 2⁹³ ≪ 顶层 q/2 ≈ 2¹⁹⁹，测试实测兜底）；
- 统计泄漏：合谋缺片的条件分布 = 单片窗（缺哪片，剩余未知量就是那片的
  均匀窗）⇒ 泄漏界与单片版相同，不随 m 劣化（docs/02 §3.3′）。

角色边界（D5 延续）：P0 持 CKKS 私钥且仅经 DecryptionWhitelist 解密；
全部计算方只持 public_only 上下文（OpNotAllowedError 结构性防线）。
本模块零改动 conversion.py——旧 2 方单片 API 原样保留供存量路径与对照测试。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from src.crypto.ckks_ops import CKKSContext, CKKSCiphertext
from src.crypto.secret_sharing import DEFAULT_MODULUS
from src.protocol.conversion import (ConversionError, ConversionPolicy,
                                     FIXED_ONE, sample_entry_mask,
                                     sample_exit_mask, policy_for)

FLOAT64_EXACT_LIMIT = 1 << 53          # P0/合成侧 float64 精确整数域（P4 实测墙）


def m_cap(policy: ConversionPolicy) -> int:
    """该掩码策略允许的最大计算方数 m（float64 2⁵³ 精确域反推）。"""
    budget = FLOAT64_EXACT_LIMIT - policy.value_bound
    return max(1, min(budget // policy.entry_mask_hi,
                      budget // policy.exit_mask_hi))


def check_m(policy: ConversionPolicy, m: int) -> None:
    """m 方数合法性：2 ≤ m ≤ m_cap（m=1 无隐藏意义，显式拒绝）。"""
    if m < 2:
        raise ConversionError(10, f"m={m} 无门限意义（计算方数须 ≥ 2）")
    cap = m_cap(policy)
    if m > cap:
        raise ConversionError(
            10, f"m={m} 超出策略 {policy.name!r} 上限 {cap}"
                "（float64 2⁵³ 精确域约束——见模块 docstring）")


def entry_add_piece(ctx_pub: CKKSContext, ct: CKKSCiphertext,
                    r_piece: List[int]) -> CKKSCiphertext:
    """链式入口单步：ct′ = ct + Enc_pk(rᵢ)（按目标精确 scale 编码 +
    mod_switch 对齐——rescale 素数非精确 2^k 的 P4 教训，逐片执行）。"""
    enc = ctx_pub.encrypt_vector_at_scale(
        [ri / FIXED_ONE for ri in r_piece], ct.raw.scale)
    if ct.level > 0:
        enc = ctx_pub.mod_switch_to(enc, ct.level)
    return ctx_pub.add(ct, enc)


def exit_compose_m(ctx_pub: CKKSContext,
                   pieces: List[Tuple[List[int], bytes]]) -> CKKSCiphertext:
    """合成方 P2 视角（m 方出口）：w = Σzᵢ = v + Σsᵢ（OTP 域，m 缩放界守卫）
    → ct = Enc(w) − ΣEnc(sᵢ) = Enc(v)，fresh 顶层密文。

    pieces：各计算方 (z, enc_s_bytes)（与 :meth:`ComputePartyRole.exit_piece`
    返回序一致）；P2 自己的片也经同一入参进入（不特殊化）。任一 sᵢ 明文
    不出现在合成方——只有 m 片的密文与 OTP 值 w。"""
    if len(pieces) < 2:
        raise ConversionError(10, "出口合成至少需要 2 方片")
    policy = policy_for(ctx_pub)
    m = len(pieces)
    check_m(policy, m)
    n = len(pieces[0][0])
    w = [0] * n
    for z, enc_s_bytes in pieces:
        if len(z) != n:
            raise ConversionError(10, "出口片长度不一致")
        w = [(wi + zi) % DEFAULT_MODULUS for wi, zi in zip(w, z)]
    hi_bound = m * policy.exit_mask_hi + policy.value_bound
    lo_bound = m * policy.exit_mask_lo - policy.value_bound
    for wi in w:
        # w = v + Σsᵢ 无回绕（m·lo ≫ value_bound，且 m·hi ≪ 2⁶⁴）：
        # 残差即真值，落 [lo_bound, hi_bound) 之外即守卫被绕过/回绕
        if wi >= hi_bound or wi < lo_bound:
            raise ConversionError(10, f"出口掩码值越界 {wi}"
                                     f"（合法窗 [{lo_bound}, {hi_bound})）")
    ct = ctx_pub.encrypt_vector([wi / FIXED_ONE for wi in w])
    for _z, enc_s_bytes in pieces:
        ct = ctx_pub.sub(ct, ctx_pub.load_ct_bytes(enc_s_bytes))
    return ct


class ComputePartyRole:
    """计算方 Pᵢ（i = 1..m）：public_only；独持入口片 rᵢ 与出口片 sᵢ。

    锚点（P1，is_anchor=True）额外接收 P0 分发的掩码解密值 y——其入口
    份额为 y − r₁；非锚点份额为 −rᵢ。任何一方都无法从自己的视图恢复 x/v。"""

    def __init__(self, ctx_pub: CKKSContext, index: int, m: int,
                 is_anchor: bool = False,
                 audit: Optional[object] = None) -> None:
        if not ctx_pub.public_only:
            raise ValueError("计算方必须是 public_only 上下文（D5 延续）")
        if not 1 <= index <= m:
            raise ValueError(f"index {index} 超出 [1, {m}]")
        self.ctx = ctx_pub
        self.index = index
        self.m = m
        self.is_anchor = is_anchor
        self._policy = policy_for(ctx_pub)
        self._audit = audit
        self._entry_pieces: dict = {}         # request_id → rᵢ
        self._result_shares: dict = {}        # request_id → aᵢ

    def _emit(self, event: str, detail: dict) -> None:
        if self._audit is not None:
            self._audit(event, detail)

    # ---- 入口：链式加片（调用序 = 链序 P2→…→Pm→P1） ----
    def entry_apply_piece(self, ct: CKKSCiphertext,
                          request_id: bytes) -> CKKSCiphertext:
        r = sample_entry_mask(self.ctx.slot_count, self._policy)
        self._entry_pieces[request_id] = r
        masked = entry_add_piece(self.ctx, ct, r)
        self._emit("MPARTY_ENTRY_PIECE", {"request_id": request_id.hex(),
                                          "index": self.index,
                                          "level": masked.level})
        return masked

    def entry_own_share(self, request_id: bytes,
                        y: Optional[List[int]] = None) -> List[int]:
        """本方入口份额：锚点 y − r₁（y 为 P0 白名单①解密值）；他方 −rᵢ。"""
        r = self._entry_pieces.pop(request_id)
        if self.is_anchor:
            if y is None:
                raise ConversionError(10, "锚点入口份额需要 P0 分发的 y")
            share = [(yv - rv) % DEFAULT_MODULUS for yv, rv in zip(y, r)]
        else:
            share = [(DEFAULT_MODULUS - rv) % DEFAULT_MODULUS for rv in r]
        self._emit("MPARTY_ENTRY_SHARE", {"request_id": request_id.hex(),
                                          "index": self.index,
                                          "n": len(share)})
        return share

    # ---- 出口：片准备（结果分享 + 自采样掩码片） ----
    def set_result_share(self, request_id: bytes, a_i: List[int]) -> None:
        self._result_shares[request_id] = a_i

    def exit_piece(self, request_id: bytes) -> Tuple[List[int], bytes]:
        """(zᵢ = aᵢ + sᵢ mod 2⁶⁴, Enc_pk(sᵢ) 序列化)——发给合成方 P2。"""
        if request_id not in self._result_shares:
            raise ConversionError(10, f"P{self.index} 无结果分享"
                                     "（MPC 结果份额未设置）")
        a = self._result_shares.pop(request_id)
        s = sample_exit_mask(len(a), self._policy)
        z = [(ai + si) % DEFAULT_MODULUS for ai, si in zip(a, s)]
        enc_s = self.ctx.encrypt_vector([si / FIXED_ONE for si in s])
        self._emit("MPARTY_EXIT_PIECE", {"request_id": request_id.hex(),
                                         "index": self.index})
        return z, self.ctx.serialize_ct_bytes(enc_s)

    def exit_piece_stored(self, request_id: bytes, s_i: List[int],
                          enc_s_bytes: bytes) -> Tuple[List[int], bytes]:
        """库存版出口片（P9 优化 3a）：sᵢ 与 Enc(sᵢ) 来自离线预生成库存
        （offline_mask_gen，D7 延伸），在线不采样、不加密、**不传 Enc(sᵢ)**。

        安全语义与在线版等价：sᵢ 仍为一次性均匀掩码片（离线采样、本方独持），
        size-(m−1) 合谋仍恰缺一片；在线出口通信只剩 z 向量（8B/槽）。"""
        if request_id not in self._result_shares:
            raise ConversionError(10, f"P{self.index} 无结果分享"
                                     "（MPC 结果份额未设置）")
        a = self._result_shares.pop(request_id)
        if len(s_i) < len(a):
            raise ConversionError(10, "库存掩码片短于结果分享——库存与管线槽位不匹配")
        if len(s_i) > len(a):
            s_i = s_i[:len(a)]      # 均匀随机向量的前缀截取仍均匀（OTP 性质保持）
        z = [(ai + si) % DEFAULT_MODULUS for ai, si in zip(a, s_i)]
        self._emit("MPARTY_EXIT_PIECE_STORED", {"request_id": request_id.hex(),
                                                "index": self.index})
        return z, enc_s_bytes
