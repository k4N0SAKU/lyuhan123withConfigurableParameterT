"""row-sum 线性层实现（C3 双实现 A/B 的对照实现；docs/01 §6 选型决策实验）。

与对角线+gap-block+BSGS（packing.linear_cipher，**W@x 列向量语义**）计算
同一函数 y = W·x，采用**循环旋转 row-sum 模式**：

    y = Σ_{i=0}^{d-1} rot(X_rep, i) ⊙ q_i，  q_i[s] = W[(s+i) mod d, s mod d]

其中 X_rep 为输入在槽环上的 d 周期复制（d | 槽数）。正确性：rot 沿全环循环
移位，X_rep[(s+i) mod N] = x[(s+i) mod d]；⊙q_i 后对 i 求和并代换
k=(s+i) mod d ⇒ y[s mod d] = Σ_k x[k]·W[k, s mod d] ✓（与对角线法等价）。

算子账（对比口径，C3 A/B）：
    row-sum：d 次旋转 + d 次明文乘 + (d-1) 次密文加（无 BSGS 结构）
    对角线+BSGS：d 次明文乘 + ~2√d 次旋转（baby/giant）
同 level 累加不 rescale（微基准口径；decode 按密文 scale 自动还原）。
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.crypto.ckks_ops import CKKSContext, CKKSCiphertext


def linear_rowsum(ctx: CKKSContext, ct: CKKSCiphertext, W: np.ndarray,
                  rotation_counter: List[int]) -> CKKSCiphertext:
    """row-sum matvec：y[s mod d] = (W·x)[s mod d]，W: (d, d) 方阵且 d | 槽数。

    输入 ct 的槽 s 应为 x[s mod d]（环上 d 周期复制）；输出同布局。
    rotation_counter[0] 累计旋转次数（A/B 算子计数口径）。"""
    d, d_out = W.shape
    if d != d_out:
        raise ValueError("row-sum 循环旋转公式要求方阵 W")
    n_slots = ctx.slot_count
    if d > n_slots or n_slots % d != 0:
        raise ValueError("d 必须整除槽数（环上周期复制）")
    s_idx = np.arange(n_slots)
    acc = None
    for i in range(d):
        q = W[(s_idx + i) % d, s_idx % d]              # q_i[s] = W[(s+i)%d, s%d]
        shifted = ctx.rotate(ct, i)
        rotation_counter[0] += 1
        term = ctx.multiply_plain(shifted, q.tolist(), rescale=False)
        acc = term if acc is None else ctx.add(acc, term)
    return acc
