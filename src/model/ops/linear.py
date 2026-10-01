"""密文线性层与嵌入（P3 实现；基于 P2 packing 选型：对角线+gap-block+BSGS）。

嵌入策略（docs/01 §2.2 信任边界）：模型对 P0 公开，P0 本地查表后加密——
P0 已知 token，查表不增加其信息量；密文域表查找（行选择门）列为本路线的
可选扩展（模型保密需求时启用）。

注意力 QK^T 与 ·V 在**模式 B 走 MPC 域**（docs/01 §6.3 段划分：QKV 后转换，
scores/softmax/·V 全在分享域，规避密文域 L² 复制开销）——见 pipeline.py。
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from src.crypto.ckks_ops import CKKSContext, CKKSCiphertext
from src.model.ops.packing import (DEFAULT_SPEC, PackingSpec,
                                   build_diagonal_plaintexts, layout_input,
                                   linear_cipher)


def embed_and_encrypt(ctx: CKKSContext, token_ids: Sequence[int],
                      embedding_matrix: "np.ndarray",
                      spec: PackingSpec = DEFAULT_SPEC) -> CKKSCiphertext:
    """P0 本地嵌入查表 + gap-block 布局 + CKKS 加密（深度 0）。

    token_ids 不足 blocks_per_ct 时零填充（注意力 mask 由 layout 后段 0 隐式
    承载——零向量的 attention 贡献经 softmax 前的 mask 项处理，见 pipeline）。"""
    d = spec.block_elems
    if embedding_matrix.shape[1] != d:
        raise ValueError(f"嵌入维度 {embedding_matrix.shape[1]} ≠ {d}")
    if max(token_ids) >= embedding_matrix.shape[0]:
        raise ValueError("token id 越界")
    tokens = [embedding_matrix[int(t)] for t in token_ids]
    layout = layout_input(tokens, len(token_ids), spec)
    return ctx.encrypt_vector(layout)


def chunk_weight(weight: "np.ndarray", out_dim: int,
                 spec: PackingSpec = DEFAULT_SPEC) -> List["np.ndarray"]:
    """把 (in, out) 权重按输出维切成 out_dim//d 个方阵块（对角线打包要求方阵）。"""
    d = spec.block_elems
    if out_dim % d != 0:
        raise ValueError(f"out_dim {out_dim} 不是 {d} 的整数倍")
    return [np.asarray(weight[:, i * d:(i + 1) * d], dtype=np.float64)
            for i in range(out_dim // d)]


def linear_multi_out(ctx: CKKSContext, ct: CKKSCiphertext,
                     weight: "np.ndarray", num_tokens: int,
                     spec: PackingSpec = DEFAULT_SPEC) -> List[CKKSCiphertext]:
    """y = x·W（W: (d_in, d_out)，d_out = k×d）：k 个方阵块分别线性层。

    输入布局为上层输出的 gap-block（半块有效）；k 个输出块共用同一输入密文
    （各块对角线不同）。返回 k 个密文（各含 d 输出槽/块）。"""
    chunks = chunk_weight(weight, weight.shape[1], spec)
    return [linear_cipher(ctx, ct, Wc, num_tokens, spec=spec) for Wc in chunks]


def concat_chunks(ctx: CKKSContext, chunks: List[CKKSCiphertext],
                  spec: PackingSpec = DEFAULT_SPEC) -> CKKSCiphertext:
    """把 k 个输出块拼为一个密文：块间相加需把第 j 块左移 j·d·gap 对齐到
    连续布局。用 rotate + 掩码复合（每块掩出自身 d 槽，移位后相加）。"""
    d = spec.block_elems
    gap = spec.gap
    slots = spec.slots
    out = None
    for j, cc in enumerate(chunks):
        if j == 0:
            shifted = cc
        else:
            shifted = ctx.rotate(cc, -j * d)   # 左移负步 = 右移？P2 实测：res[s]=v[s+steps]
        # 掩出该块目标位置：目标块 b 的槽 [b*bs + j*d, b*bs + (j+1)*d)
        mask = [0.0] * slots
        bs = spec.block_slots
        for b in range(spec.blocks_per_ct):
            base = b * bs + j * d
            for i in range(d):
                if base + i < slots:
                    mask[base + i] = 1.0
        m = ctx.multiply_plain(shifted, mask, rescale=False)
        out = m if out is None else ctx.add(out, m)
    return ctx.rescale_next(out)
