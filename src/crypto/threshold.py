"""CKKS 门限解密——协议逻辑级模拟层（P2 实现；主线不用，见 docs/01 §8.4 D5 修订）。

诚实声明（S8，评审 A/B 项结论）：
- **主线是 P0 单钥掩码解密**（TenSEAL 无门限实现载体：不暴露私钥多项式/
  密文分量运算/密钥聚合；evk 份额聚合存在 s² 交叉项数学错误）；
- 本模块为**门限协议逻辑的可测试模拟**：以标量 LWE（同 RLWE 的代数结构，
  单整数环）承载 partial decrypt / combine / 承诺校验 / 缺份额拒绝的完整
  协议逻辑，供失败路径测试与协议验证；**它不是真 CKKS 门限**；
- 真 2-of-2 门限须按多参与方 HE 文献（Mouchet et al. 2020）在 OpenFHE 等
  后端实现（P2 可选 spike，2 天 timebox）。

模拟层代数（模 q 标量环）：
    keygen：s₀ ←$，s ←$，s₁ = s − s₀（模拟 DKG 输出：任何单方不知 s）
    encrypt(m, Δ)：c = (a, b)，a ←$，b = a·s + Δ·m + e（e 小噪声，LWE）
    partial_decrypt：party_i 输出 pᵢ 及 SM3 承诺
        p₀ = b − a·s₀（P0），p₁ = −a·s₁（P1）⇒ p₀+p₁ = Δ·m + e
    combine：(p₀+p₁ mod q) 经 Δ 舍入恢复 m；任一承诺不符/缺份额 → 拒绝
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from gmssl import sm3


class ThresholdError(Exception):
    """门限解密失败（份额缺失/承诺不符/密钥已销毁/密文损坏）。"""


SCALING = 1 << 16          # Δ：消息缩放因子（模拟 CKKS scale 的角色）
NOISE_BOUND = 1 << 8       # e 的界（LWE 噪声；真实 CKKS 中由 SEAL 管理）
_MODULUS = 1 << 127        # 标量环 q（大素数域近似，2^127 保证无溢出歧义）


def _rand(below: int) -> int:
    return int.from_bytes(os.urandom(16), "big") % below


def _noise() -> int:
    return _rand(2 * NOISE_BOUND + 1) - NOISE_BOUND


@dataclass
class SimulatedKeyShare:
    party_id: int = 0
    s_i: int = 0                             # 私钥份额（仅驻内存，S3）
    destroyed: bool = False


@dataclass
class SimulatedCiphertext:
    a: int = 0
    b: int = 0
    scaling: int = SCALING


def threshold_keygen() -> Tuple[SimulatedKeyShare, SimulatedKeyShare]:
    """模拟 2-of-2 DKG 输出：返回 (P0 份额, P1 份额)，s=s₀+s₁ 无单方可知。"""
    s = _rand(_MODULUS)
    s0 = _rand(_MODULUS)
    s1 = (s - s0) % _MODULUS
    return SimulatedKeyShare(0, s0), SimulatedKeyShare(1, s1)


def threshold_encrypt(key_shares: Tuple[SimulatedKeyShare, ...],
                      value: int) -> SimulatedCiphertext:
    """模拟加密（需聚合公钥 = s，模拟层由份额合成，真 DKG 不出现完整 s）。"""
    s = sum(k.s_i for k in key_shares) % _MODULUS
    a = _rand(_MODULUS)
    b = (a * s + SCALING * value + _noise()) % _MODULUS
    return SimulatedCiphertext(a, b)


def partial_decrypt(share: SimulatedKeyShare,
                    ct: SimulatedCiphertext) -> Tuple[bytes, bytes]:
    """部分解密份额 + SM3 承诺（份额替换可检测，F7）。

    非对称份额构造：p₀ = b − a·s₀（P0），p₁ = −a·s₁（P1）
    ⇒ p₀ + p₁ = b − a·(s₀+s₁) = Δ·m + e。"""
    if share.destroyed:
        raise ThresholdError(f"party {share.party_id} 密钥已销毁（F5 断言点）")
    if share.party_id == 0:
        p_i = (ct.b - ct.a * share.s_i) % _MODULUS
    elif share.party_id == 1:
        p_i = (-ct.a * share.s_i) % _MODULUS
    else:
        raise ThresholdError("非法 party_id（2-of-2）")
    pb = p_i.to_bytes(16, "big")
    commitment = bytes.fromhex(sm3.sm3_hash(list(pb)))
    return pb, commitment


def combine(partials: list, ct: SimulatedCiphertext) -> int:
    """合并份额恢复明文。

    参数为 [(party_id, partial_bytes, commitment), ...]；必须恰有两方、承诺
    全部匹配，否则 ThresholdError（缺份额不可解——任务书失败路径）。"""
    if len(partials) != 2 or {p[0] for p in partials} != {0, 1}:
        raise ThresholdError("份额缺失或不完整：2-of-2 门限要求双方份额（缺一不可解）")
    total = 0
    for party_id, pb, commitment in partials:
        if bytes.fromhex(sm3.sm3_hash(list(pb))) != commitment:
            raise ThresholdError(f"party {party_id} 份额承诺不匹配（份额被替换）")
        total += int.from_bytes(pb, "big")
    value = total % _MODULUS                  # = Δ·m + e (mod q)
    if value > _MODULUS // 2:                 # 负数环绕修正
        value -= _MODULUS
    m = round(value / ct.scaling)             # e ∈ [−B,B] ≪ Δ/2，舍入恢复
    return int(m)


def destroy_share(share: SimulatedKeyShare) -> None:
    """销毁：置位后该方 partial_decrypt 永久拒绝（S3/F5）。"""
    share.s_i = 0
    share.destroyed = True
