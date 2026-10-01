"""m 方加性秘密分享（P8 协议族；secret_sharing.py 的 m 方泛化，零改动原文件）。

m-of-m 加性分享：Σᵢ shareᵢ ≡ value (mod 2⁶⁴)。安全性质与 2 方版一致：
每份来自 os.urandom（拒绝采样，S3），任意 t ≤ m−1 方合谋至少缺一份
均匀随机份额 ⇒ 信息论隐藏（OTP）。加法/公开常数乘为本地操作。

``reconstruct_n`` 仅为测试与客户端（P0 终点）提供——节点常驻代码路径
禁止 import（同 secret_sharing.py 纪律，P3 评审检查项延续）。
"""
from __future__ import annotations

from typing import List, Sequence

from src.crypto.secret_sharing import DEFAULT_MODULUS, sample_fresh_mask


def share_vector_n(values: Sequence[int], n: int,
                   modulus: int = DEFAULT_MODULUS) -> List[List[int]]:
    """向量 m-of-m 加性分享：返回 n 份，Σᵢ shares[i] ≡ values (mod m)。

    n=1 退化为明文（无隐藏——调用方应保证 n ≥ 2）。逐元素独立随机
    （共享同一随机数会破坏安全性，与 2 方版同纪律）。"""
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    if n == 1:
        return [[v % modulus for v in values]]
    shares = [sample_fresh_mask(len(values), modulus) for _ in range(n - 1)]
    acc = [sum(col) % modulus for col in zip(*shares)]
    shares.append([(v - a) % modulus for v, a in zip(values, acc)])
    return shares


def reconstruct_n(shares: List[Sequence[int]],
                  modulus: int = DEFAULT_MODULUS) -> List[int]:
    """m 方重构（仅测试/客户端终点；节点常驻路径禁止 import）。"""
    if not shares:
        raise ValueError("份额列表为空")
    n = len(shares[0])
    if any(len(s) != n for s in shares):
        raise ValueError("份额长度不一致")
    return [sum(col) % modulus for col in zip(*shares)]
