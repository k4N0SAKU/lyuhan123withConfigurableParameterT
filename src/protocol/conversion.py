"""模式 B 密文↔分享转换真实路径（docs/01 §7.4 的 P4 协议层实现；D5 白名单执行点）。

P3 模拟捷径（解密后本地加掩码）在本模块被替换为**真协议**：
- 入口：P2 fresh 掩码 r → ct′ = ct(x) + Enc_pk(r)（**密文域**同态加）→
  P0 单钥解密（DecryptionWhitelist 门控，D5 白名单①）得 y = x + r →
  CONVERT_SHARE 分发 y₁ 给 P1；P2 本地持 −r。分享 x = y₁ + (−r) (+e_ckks)。
- 出口：结果分享 (a₁[P1], a₂[P2])；P1 fresh 掩码 s → z = a₁ + s（OTP）
  与 Enc_pk(s)；P2 计算 w = (z + a₂) mod 2⁶⁴ = v + s 后在**密文域**合成
  ct = Enc(w) − Enc(s) = Enc(v)，fresh 顶层密文（深度重置）。

偏置 OTP 掩码（P3 docstring 预告的「mod 2⁶⁴ 溢出控制——CrypTFlow 式处理」，
两方向约束不同，按 :class:`ConversionPolicy` 分参数集配置）：

- **入口 r（P2 采样）**：[0, hi) 上 2¹¹ 网格均匀（r = k·2¹¹）。网格使 r/2¹⁶
  恰 ≤ 53 有效位 ⇒ float64 **精确**编码；OTP 一致集覆盖值域（窗宽 ≫ 值域，
  边界事件概率见 policy 注释）；回绕由 mod 2⁶⁴ 环算术自然吸收
  （重构 (y₁ − r) mod 2⁶⁴ = x·2¹⁶ 精确）。hi 的硬上限：P0 侧
  round((x + r/2¹⁶)·2¹⁶) 须落在 float64 精确整数域 ≤ 2⁵³（超出即引入
  ±2¹¹ 量级分享偏差——P4 mode-b 规模测试实测发现并修正）。
- **出口 s（P1 采样）**：[lo, hi) 均匀。lo ≥ value_bound 保证 w = v + s ≥ 0
  （无 mod 回绕）；hi 保证 w/2¹⁶ 在 SEAL 可编码域且 ≤ 53 有效位。
  统计泄漏仅出现在掩码窗端点邻域（宽 |v|）：窗宽 ≫ 值域 ⇒
  mode-b 实测激活域（|v| ≤ 2²⁴）≤ 2⁻³² bit/slot、理论最坏 2⁻⁶；
  详见 docs/02 与 test_conversion。

掩码窗与 SEAL 编码幅值（P4 探针实测，全随机向量口径）：明文系数 =
值×scale，SEAL 对系数幅值有硬上限——mode-b（链 200bit/scale 2⁴⁰）值域 2⁴⁸
可编码（系数 2⁸⁸ OK）；玩具参数（链 100bit/scale 2³⁰）值域上限 ~2²⁸ ⇒
掩码窗必须按参数集缩放（:func:`policy_for`，未登记参数集显式报错）。

角色边界（D5）：P0 持 CKKS 私钥且**仅**经 DecryptionWhitelist 解密
（masked_conversion / final_output 两类，前者为 OTP 域定点、后者过激活域
值域守卫）；P1/P2 只持 public_only 上下文（OpNotAllowedError 结构性防线）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.crypto.ckks_ops import CKKSContext, CKKSCiphertext
from src.crypto.secret_sharing import DEFAULT_MODULUS

FIXED_FRAC_BITS = 16
FIXED_ONE = 1 << FIXED_FRAC_BITS
ENTRY_MASK_GRID = 1 << 11               # 入口掩码网格（float64 精确性来源）
DEFAULT_VALUE_BOUND = 1 << 46           # mode-b 定点值域守卫
WHITELIST_KINDS = ("masked_conversion", "final_output")


class ConversionError(Exception):
    """转换协议失败（值域越界/白名单拒绝/策略缺失）。error_code 供审计引用。"""

    def __init__(self, error_code: int, detail: str) -> None:
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail


class WhitelistError(ConversionError):
    """D5 白名单拒绝（非掩码转换值/非最终输出的解密请求）。"""

    def __init__(self, detail: str) -> None:
        super().__init__(7, detail)     # ErrorCode.DECRYPT_FAILURE 语义域


@dataclass(frozen=True)
class ConversionPolicy:
    """一套 CKKS 参数集的转换掩码参数（P4 探针实测，见 P4 工作记录 §3）。"""

    name: str
    value_bound: int        # |定点值| 守卫（必须 ≪ 掩码窗宽）
    entry_mask_hi: int      # 入口掩码 ∈ [0, hi)，网格 ENTRY_MASK_GRID
    exit_mask_lo: int       # 出口掩码 ∈ [lo, hi)：lo ≥ value_bound 保无回绕，
    exit_mask_hi: int       # hi 使 w=v+s 的定点值仍在可编码域且 ≤53 位


# float64 硬约束（P4 实测发现，mode-b 规模测试暴露）：P0 侧 round(y·2¹⁶) 的
# 精确整数域上限 2⁵³ ⇒ 掩码值 r/2¹⁶ 与定点值之和 ×2¹⁶ 必须 ≤ 2⁵³ ⇒
# 全宽 2⁶⁴ 掩码不可用（y₁ 偏差 ±2¹²）——入口窗收缩为 [0, 2⁵³)，
# 值域守卫相应取 2³²（边界泄漏 P ≈ 2·2³²/2⁵³ = 2⁻²⁰/slot，仍可忽略）。
# 掩码量级标定（P4 实测两次迭代，见工作记录 §3）：
# - float64 ulp 抖动（入口解密重构）∝ 掩码值量级：2³⁷ ⇒ δ≤−5、2³³ ⇒ δ≤4；
# - CKKS 编码残差（出口合成）∝ 掩码值量级：2³³ 值域 ⇒ 2.4e-6、2²⁰ ⇒ ~1e-11；
# ⇒ 掩码值统一压到 ≤2²⁰：值域守卫 2²⁴（激活域 ±256，覆盖实测激活/分数），
#   入口窗 [0,2⁴⁹)、出口窗 [2²⁵,2³⁶)——边界泄漏 ≤2·2²⁴/2³⁶ = 2⁻¹¹/slot。
POLICY_MODE_B = ConversionPolicy(
    name="mode-b-segment", value_bound=1 << 24,
    entry_mask_hi=1 << 49, exit_mask_lo=1 << 25, exit_mask_hi=1 << 36)

POLICY_P4_TOY = ConversionPolicy(
    name="p4-protocol-toy", value_bound=1 << 24,
    entry_mask_hi=1 << 40, exit_mask_lo=1 << 25,
    exit_mask_hi=(1 << 40) - (1 << 24))


def policy_for(ctx: CKKSContext) -> ConversionPolicy:
    """按参数名取掩码策略；未登记的参数集显式报错（禁止静默猜测）。"""
    name = ctx.params.name
    if "mode-b" in name:
        return POLICY_MODE_B
    if "p4-toy" in name or "protocol-toy" in name:
        return POLICY_P4_TOY
    raise ConversionError(
        10, f"参数集 {name!r} 未登记 ConversionPolicy——"
            "请在 conversion.py 显式登记（掩码窗依赖 SEAL 编码幅值实测）")


def _rand_entry_mask(policy: ConversionPolicy) -> int:
    """入口掩码：[0, hi) 上 2¹¹ 网格均匀（拒绝采样，S3）。"""
    span = policy.entry_mask_hi // ENTRY_MASK_GRID
    while True:
        nbytes = (span.bit_length() + 7) // 8
        k = int.from_bytes(os.urandom(nbytes), "big") % (1 << (nbytes * 8))
        if k < span:
            return k * ENTRY_MASK_GRID


def _rand_exit_mask(policy: ConversionPolicy) -> int:
    """出口掩码：[lo, hi) 均匀整数（拒绝采样，S3）。"""
    span = policy.exit_mask_hi - policy.exit_mask_lo
    while True:
        nbytes = (span.bit_length() + 7) // 8
        v = int.from_bytes(os.urandom(nbytes), "big") % (1 << (nbytes * 8))
        if v < span:
            return policy.exit_mask_lo + v


def sample_entry_mask(n: int, policy: ConversionPolicy) -> List[int]:
    """入口 fresh 掩码向量（全宽 OTP + float64 精确，模块 docstring 论证）。"""
    return [_rand_entry_mask(policy) for _ in range(n)]


def sample_exit_mask(n: int, policy: ConversionPolicy) -> List[int]:
    """出口 fresh 掩码向量（偏置窗）。"""
    return [_rand_exit_mask(policy) for _ in range(n)]


def to_fixed(value: float, value_bound: int = DEFAULT_VALUE_BOUND) -> int:
    """激活域浮点 → Q16 定点（mod 2⁶⁴ 环；值域守卫显式报错）。"""
    f = int(round(value * FIXED_ONE))
    if not -value_bound < f < value_bound:
        raise ConversionError(10, f"定点值越界 {f}（|v| 必须 < {value_bound}——"
                                 "OTP 掩码/float64 精度论证前提）")
    return f % DEFAULT_MODULUS


def from_fixed(v: int) -> float:
    """Q16 定点（mod 2⁶⁴ 环）→ 居中浮点。"""
    s = v % DEFAULT_MODULUS
    if s > DEFAULT_MODULUS // 2:
        s -= DEFAULT_MODULUS
    return s / FIXED_ONE


def pack_ints(values: List[int]) -> bytes:
    """定点分享向量 → 字节（8B 大端/槽；通道载荷口径）。"""
    out = bytearray()
    for v in values:
        out += (v % DEFAULT_MODULUS).to_bytes(8, "big")
    return bytes(out)


def unpack_ints(data: bytes) -> List[int]:
    if len(data) % 8 != 0:
        raise ConversionError(10, "分享向量字节长必须为 8 的倍数")
    return [int.from_bytes(data[i:i + 8], "big")
            for i in range(0, len(data), 8)]


# ---- 入口（ct → shares）：D5 白名单① 执行点 ----

def entry_make_masked_ct(ctx_pub: CKKSContext, ct: CKKSCiphertext
                         ) -> Tuple[CKKSCiphertext, List[int]]:
    """P2 视角：fresh 掩码 r → Enc(r) 对齐 level → ct′ = ct + Enc(r)。

    返回 (ct_masked, r)。r 同时是 P2 的入口分享基数（−r mod 2⁶⁴）。"""
    policy = policy_for(ctx_pub)
    r = sample_entry_mask(ctx_pub.slot_count, policy)
    # 掩码密文必须按目标密文的**精确 scale** 编码（rescale 素数非精确 2^k，
    # 深层密文 scale 有相对漂移——relabeling 在 OTP 大值域误差数千，P4 实测）
    enc_r = ctx_pub.encrypt_vector_at_scale([ri / FIXED_ONE for ri in r],
                                            ct.raw.scale)
    if ct.level > 0:
        enc_r = ctx_pub.mod_switch_to(enc_r, ct.level)
    masked = ctx_pub.add(ct, enc_r)
    return masked, r


def entry_decrypt_masked(ctx_sec: CKKSContext, ct_masked: CKKSCiphertext
                         ) -> List[int]:
    """P0 视角（白名单①内）：解密掩码值 y = x + r/2¹⁶ → 定点 y₁ = round(y·2¹⁶)。

    注意：掩码值 y 在 OTP 域（~掩码量级），**不过**激活域值域守卫——
    环算术 (y₁ − r) mod 2⁶⁴ 自动吸收回绕，重构后才是激活域定点。"""
    y = ctx_sec.decrypt(ct_masked)
    return [int(round(v * FIXED_ONE)) % DEFAULT_MODULUS for v in y]


def entry_p2_share(r: List[int]) -> List[int]:
    """P2 的入口分享：−r mod 2⁶⁴。"""
    return [(DEFAULT_MODULUS - ri) % DEFAULT_MODULUS for ri in r]


# ---- 出口（shares → ct）：再随机化 ----

def exit_keynode_prepare(ctx_pub: CKKSContext, a1: List[int]
                         ) -> Tuple[List[int], bytes]:
    """P1 视角：fresh 偏置掩码 s → z = (a₁ + s) mod 2⁶⁴（OTP）与 Enc_pk(s)。

    返回 (z, enc_s_bytes)。P2 从不获得 s 明文；Enc(s) 由 P1 编码——目标布局
    由调用方的 a1 排序承载（布局免费重置，docs/01 §7.4）。"""
    policy = policy_for(ctx_pub)
    s = sample_exit_mask(len(a1), policy)
    z = [(ai + si) % DEFAULT_MODULUS for ai, si in zip(a1, s)]
    enc_s = ctx_pub.encrypt_vector([si / FIXED_ONE for si in s])
    return z, ctx_pub.serialize_ct_bytes(enc_s)


def exit_infernode_compose(ctx_pub: CKKSContext, enc_s_bytes: bytes,
                           z: List[int], a2: List[int]) -> CKKSCiphertext:
    """P2 视角：w = (z + a₂) mod 2⁶⁴（= v + s，无回绕守卫）→
    ct = Enc(w) − Enc(s) = Enc(v)，fresh 顶层密文。

    P2 只见 OTP 掩码值 w；P1 从不获得 a₂（docs/01 §7.4 t=1 论证）。"""
    policy = policy_for(ctx_pub)
    w = [(zi + ai) % DEFAULT_MODULUS for zi, ai in zip(z, a2)]
    for wi in w:
        if wi >= policy.exit_mask_hi + policy.value_bound:
            raise ConversionError(10, f"出口掩码值越界 {wi}"
                                     "（疑似回绕/值域守卫被绕过）")
    ct_w = ctx_pub.encrypt_vector([wi / FIXED_ONE for wi in w])
    ct_s = ctx_pub.load_ct_bytes(enc_s_bytes)
    return ctx_pub.sub(ct_w, ct_s)


# ---- D5 解密白名单（P0 侧结构性防线） ----

class DecryptionWhitelist:
    """P0 私钥的唯一访问门：只放行 masked_conversion / final_output 两类。

    结构性纪律：持钥上下文私有（`_ctx`），除本方法外无任何解密入口；
    每次放行/拒绝都产生审计事件。两类语义分域：
    - masked_conversion：解密值为 OTP 掩码域定点（round(v·2¹⁶) mod 2⁶⁴），
      **不**过激活域值域守卫（掩码主导，守卫无意义）；
    - final_output：激活域定点，过 :meth:`to_fixed` 值域守卫。"""

    def __init__(self, ctx_secret: CKKSContext, node_id: str = "P0",
                 audit: Optional[Callable[[str, dict], None]] = None) -> None:
        if ctx_secret.public_only:
            raise ValueError("白名单门必须持私钥上下文")
        self._ctx = ctx_secret
        self._policy = policy_for(ctx_secret)
        self.node_id = node_id
        self._audit = audit

    def decrypt(self, kind: str, ct: CKKSCiphertext,
                request_id: bytes = b"") -> List[int]:
        """白名单门控解密。kind ∉ 白名单 → WhitelistError + 审计拒绝事件。"""
        if kind not in WHITELIST_KINDS:
            self._emit("DECRYPT_WHITELIST_REJECT",
                       {"kind": kind, "request_id": request_id.hex()})
            raise WhitelistError(f"解密请求 kind={kind!r} 不在 D5 白名单"
                                 f"（{WHITELIST_KINDS}）")
        y = self._ctx.decrypt(ct)
        if kind == "masked_conversion":
            vals = [int(round(v * FIXED_ONE)) % DEFAULT_MODULUS for v in y]
        else:   # final_output：激活域
            vals = [to_fixed(v, self._policy.value_bound) for v in y]
        self._emit("DECRYPT_OK", {"kind": kind, "n": len(vals),
                                  "request_id": request_id.hex()})
        return vals

    def _emit(self, event: str, detail: dict) -> None:
        if self._audit is not None:
            self._audit(event, detail)


# ---- 进程内角色封装（S8 模拟口径：管线直连角色；通道版经 nodes/ 消息流） ----

class ClientRole:
    """P0：持钥 + 白名单门（不向 P1/P2 暴露任何密钥材料）。"""

    def __init__(self, ctx_secret: CKKSContext,
                 audit: Optional[Callable[[str, dict], None]] = None) -> None:
        self.whitelist = DecryptionWhitelist(ctx_secret, audit=audit)
        self.decrypt_events = 0

    def masked_decrypt(self, ct_masked: CKKSCiphertext,
                       request_id: bytes) -> List[int]:
        return self.whitelist.decrypt("masked_conversion", ct_masked, request_id)

    def final_decrypt(self, ct: CKKSCiphertext, request_id: bytes) -> List[float]:
        """最终输出解密（D5 白名单②）：返回激活域浮点。"""
        vals = self.whitelist.decrypt("final_output", ct, request_id)
        return [from_fixed(v) for v in vals]


class KeyRole:
    """P1：public_only 上下文；入口分享接收 + 出口掩码准备。"""

    def __init__(self, ctx_pub: CKKSContext) -> None:
        if not ctx_pub.public_only:
            raise ValueError("P1 必须是 public_only 上下文（D5）")
        self.ctx = ctx_pub
        self.entry_shares: dict = {}          # request_id → 定点分享

    def take_entry_share(self, request_id: bytes, share: List[int]) -> None:
        self.entry_shares[request_id] = share

    def exit_prepare(self, request_id: bytes) -> Tuple[List[int], bytes]:
        a1 = self.entry_shares.pop(request_id)
        return exit_keynode_prepare(self.ctx, a1)


class InferRole:
    """P2：public_only 上下文；入口掩码 + 出口合成（计算主体所在侧）。"""

    def __init__(self, ctx_pub: CKKSContext) -> None:
        if not ctx_pub.public_only:
            raise ValueError("P2 必须是 public_only 上下文（D5）")
        self.ctx = ctx_pub
        self.pending_masks: dict = {}         # request_id → r

    def entry(self, ct: CKKSCiphertext, request_id: bytes) -> CKKSCiphertext:
        masked, r = entry_make_masked_ct(self.ctx, ct)
        self.pending_masks[request_id] = r
        return masked

    def entry_p2_share(self, request_id: bytes) -> List[int]:
        return entry_p2_share(self.pending_masks.pop(request_id))

    def exit_compose(self, request_id: bytes, enc_s_bytes: bytes,
                     z: List[int], a2: List[int]) -> CKKSCiphertext:
        return exit_infernode_compose(self.ctx, enc_s_bytes, z, a2)
