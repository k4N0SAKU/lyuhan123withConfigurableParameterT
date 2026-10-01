"""Beaver 三元组 m 方拆分与在线乘法（beaver.py 的 m 方泛化，零改动原文件）。

D7 离线假设不变：三元组由可信离线生成器预生成、生成器不合谋不在线；
半诚实安全口径不变。m 方语义：任意 t ≤ m−1 方合谋时，三元组份额同样
m-of-m 拆分，缺方份额均匀 ⇒ open 值 d/e 不泄露 x−a 之外的份额信息。

在线协议（m-party，定点域 mod 2^k）：
    open 阶段：Pᵢ 广播 dᵢ = xᵢ−aᵢ, eᵢ = yᵢ−bᵢ；
    组合 d = Σdᵢ, e = Σeᵢ（公开值 x−a, y−b）；
    本地：zᵢ = cᵢ + e·aᵢ + d·bᵢ（+ d·e 仅 P1 一方）
    ⇒ Σzᵢ = c + e·a + d·b + d·e = (a+e)(b+d) = x·y (mod m)。
"""
from __future__ import annotations

import os
from typing import List, Sequence

from src.crypto.beaver import BeaverTriple, BeaverTripleShare
from src.crypto.secret_sharing import DEFAULT_MODULUS


def split_triple_n(triple: BeaverTriple, n: int,
                   modulus: int = DEFAULT_MODULUS) -> List[BeaverTripleShare]:
    """完整三元组 → n 方份额（Σaᵢ ≡ a, Σbᵢ ≡ b, Σcᵢ ≡ c，逐元素独立随机）。

    模拟层口径（同 beaver.py split_triple）：真实部署由离线生成器分别为
    各方写独立库存文件，完整值不出现在任一参与方。"""
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    shares = []
    acc = [0, 0, 0]
    for _ in range(n - 1):
        parts = [int.from_bytes(os.urandom(8), "big") % modulus for _ in range(3)]
        shares.append(BeaverTripleShare(triple.gate_id, *parts))
        acc = [(a + p) % modulus for a, p in zip(acc, parts)]
    shares.append(BeaverTripleShare(
        triple.gate_id,
        (triple.a - acc[0]) % modulus,
        (triple.b - acc[1]) % modulus,
        (triple.c - acc[2]) % modulus))
    return shares


def beaver_multiply_n(xs: Sequence[int], ys: Sequence[int],
                      shares: Sequence[BeaverTripleShare],
                      modulus: int = DEFAULT_MODULUS) -> List[int]:
    """在线 Beaver 乘法（m 方份额视图入参）：返回 [zᵢ]，Σzᵢ ≡ x·y (mod m)。

    通信口径（F9 延续）：open 阶段每方广播 2 个域元素（dᵢ、eᵢ）
    = 2·log₂m 比特/门/方，m 方合计 2·m·log₂m 比特/门。"""
    n = len(shares)
    if len(xs) != n or len(ys) != n:
        raise ValueError("份额与输入方数不一致")
    ds = [(x - s.a) % modulus for x, s in zip(xs, shares)]
    es = [(y - s.b) % modulus for y, s in zip(ys, shares)]
    d = sum(ds) % modulus                    # 公开值 x − a
    e = sum(es) % modulus                    # 公开值 y − b
    zs = []
    for i, s in enumerate(shares):
        z = (s.c + e * s.a + d * s.b) % modulus
        if i == 0:
            z = (z + d * e) % modulus        # d·e 记 P1 一方（Σzᵢ = x·y）
        zs.append(z)
    return zs
