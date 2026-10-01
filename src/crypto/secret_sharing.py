"""模 2^k 环上加法秘密分享（P2 实现；docs/01 §7.4 模式 B 分享域）。

安全性质：
- 单份额均匀分布（sample_fresh_mask / additive_share 的份额来自 os.urandom），
  统计均匀性测试见 tests/unit/test_secret_sharing.py；
- 加法/公开常数乘法为**本地操作**（零交互、零通信）；
- ``reconstruct`` 仅为测试与客户端（P0 终点）提供——按陷阱要求，**禁止出现在
  节点常驻代码路径**（nodes/ 不得 import 本函数，P3 评审检查项）。
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

DEFAULT_MODULUS_BITS = 64
DEFAULT_MODULUS = 1 << DEFAULT_MODULUS_BITS


def _rand_below(modulus: int) -> int:
    """均匀采样 [0, modulus)（拒绝采样，S3：os.urandom）。"""
    if modulus <= 0:
        raise ValueError("modulus 必须为正")
    nbits = (modulus - 1).bit_length()
    nbytes = (nbits + 7) // 8
    while True:
        v = int.from_bytes(os.urandom(nbytes), "big") % (1 << (nbytes * 8))
        if v < modulus:
            return v


def sample_fresh_mask(num_elements: int,
                      modulus: int = DEFAULT_MODULUS) -> List[int]:
    """均匀随机掩码向量（转换协议入口/出口的 fresh 掩码来源）。"""
    if num_elements < 0:
        raise ValueError("num_elements 不能为负")
    return [_rand_below(modulus) for _ in range(num_elements)]


def additive_share(value: int, modulus: int = DEFAULT_MODULUS) -> Tuple[int, int]:
    """标量加法分享：(share0, share1)，share0+share1 ≡ value (mod m)。"""
    share0 = _rand_below(modulus)
    return share0, (value - share0) % modulus


def share_vector(values: Sequence[int],
                 modulus: int = DEFAULT_MODULUS) -> Tuple[List[int], List[int]]:
    """向量加法分享（逐元素独立随机——共享同一随机数会破坏安全性）。"""
    s0 = sample_fresh_mask(len(values), modulus)
    s1 = [(v - r) % modulus for v, r in zip(values, s0)]
    return s0, s1


def reconstruct(share0: int, share1: int, modulus: int = DEFAULT_MODULUS) -> int:
    """重构（仅测试/客户端终点使用；节点常驻路径禁止 import，见模块 docstring）。"""
    return (share0 + share1) % modulus


def share_add(a: Sequence[int], b: Sequence[int],
              modulus: int = DEFAULT_MODULUS) -> List[int]:
    """份额加法（本地，零交互）。"""
    if len(a) != len(b):
        raise ValueError("份额长度不一致")
    return [(x + y) % modulus for x, y in zip(a, b)]


def share_mul_const(share: Sequence[int], c: int,
                    modulus: int = DEFAULT_MODULUS) -> List[int]:
    """公开常数乘法（本地，零交互；双方用同一常数即可保持一致性）。"""
    return [(x * c) % modulus for x in share]


def share_dot_const(share: Sequence[int], c: Sequence[int],
                    modulus: int = DEFAULT_MODULUS) -> int:
    """与公开向量的内积（本地）：⟨share, c⟩，双方同 c 即可恢复 ⟨x, c⟩。"""
    if len(share) != len(c):
        raise ValueError("维度不一致")
    return sum(x * y for x, y in zip(share, c)) % modulus
