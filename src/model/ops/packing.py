"""SIMD 打包与密文线性层（P2 实现；P1 §6 选型：对角线 + gap-block + BSGS）。

P2 实现说明（与 P1 §6.2 的两处差异，见工作记录）：
- 旋转语义按 TenSEAL/SEAL 实测为**左移**（res[s]=v[s+steps]），推导据此校准；
- "输出布局与下层输入天然对齐"修正为：输出仅块前 d 槽有效（对角线明文后段
  置 0），层间需一次**块内复制变换** ``replicate_blocks``（1 旋转 + 2 掩码乘
  + 1 加；掩码以正常 scale 编码、同 level 累加统一 rescale——净耗 1 层；
  模式 B 主线不使用本原语：布局重置由 P1 在转换出口编码时免费完成）；
- BSGS 深度记账：全部 d 次明文乘在**同一 level** 上累加（不逐乘 rescale），
  求和后统一 rescale 一次 ⇒ 每线性层净耗深度 1（P1 表 2 口径成立）。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from src.crypto.ckks_ops import CKKSContext, CKKSCiphertext


class PackingScheme(enum.IntEnum):
    ROW_SUM = 0          # 方案一：行打包 + 旋转求和（C3 对照实验用，P3 实现）
    DIAGONAL_BSGS = 1    # 方案二：对角线 + gap-block + BSGS（默认，本文件实现）


@dataclass(frozen=True)
class PackingSpec:
    """gap-block 布局参数：每 token 占 2×block_elems 槽（双拷贝支撑块内环绕旋转）。"""

    block_elems: int = 768            # d_model=768（GPT-2/BERT 真实 shape）
    gap: int = 2
    slots: int = 1 << 14              # 本栈实测上限：2^15 poly → 16384 槽

    @property
    def block_slots(self) -> int:
        return self.block_elems * self.gap

    @property
    def blocks_per_ct(self) -> int:
        return self.slots // self.block_slots

    def validate(self) -> None:
        if self.gap != 2:
            raise ValueError("本实现固定 gap=2（双拷贝环绕）")
        if self.block_slots > self.slots:
            raise ValueError("block_slots 超出槽数")


DEFAULT_SPEC = PackingSpec()                      # GPT-2/BERT 真实 shape（10 块/密文）
SMALL_SPEC = PackingSpec(block_elems=16, slots=1 << 12)   # 单测（128 块/密文）


def layout_input(token_values: Sequence[Sequence[float]], num_tokens: int,
                 spec: PackingSpec = DEFAULT_SPEC) -> List[float]:
    """明文槽布局：块 b 的两份拷贝（前 d 槽 + 后 d 槽），不足的块补 0。"""
    spec.validate()
    d = spec.block_elems
    if len(token_values) != num_tokens:
        raise ValueError("token_values 与 num_tokens 不一致")
    out: List[float] = []
    for b in range(spec.blocks_per_ct):
        if b < num_tokens:
            x = list(token_values[b])
            if len(x) != d:
                raise ValueError(f"token {b} 长度 {len(x)} ≠ {d}")
        else:
            x = [0.0] * d
        out.extend(x + x)          # 双拷贝：前 d + 后 d
    return out


def _mask(spec: PackingSpec, first_half: bool) -> List[float]:
    d = spec.block_elems
    m: List[float] = []
    for _ in range(spec.blocks_per_ct):
        m.extend([1.0] * d if first_half else [0.0] * d)
        m.extend([0.0] * d if first_half else [1.0] * d)
    return m


def build_diagonal_plaintexts(weight: "np.ndarray",
                              spec: PackingSpec = DEFAULT_SPEC) -> List[List[float]]:
    """构造 d 条对角线明文：d_k[i]=W[i][(i+k)%d]（i<d），后 d 槽掩 0；逐块重复。

    numpy 向量化：d_k = diag(roll(W, -k))（P2 实测等价于逐元素定义）。"""
    spec.validate()
    d = spec.block_elems
    W = np.asarray(weight, dtype=np.float64)
    if W.shape != (d, d):
        raise ValueError(f"W 必须是 {d}×{d} 方阵，得到 {W.shape}")
    plains: List[List[float]] = []
    for k in range(d):
        diag = np.diag(np.roll(W, -k, axis=1))
        full = np.zeros((spec.blocks_per_ct, spec.block_slots))
        full[:, :d] = diag
        plains.append(full.reshape(-1).tolist())
    return plains


def replicate_blocks(ctx: CKKSContext, ct: CKKSCiphertext,
                     spec: PackingSpec = DEFAULT_SPEC) -> CKKSCiphertext:
    """块内复制变换：把每块前 d 槽的值复制到后 d 槽（密文域布局衔接原语）。

    = mask_f ⊙ ct + rot(ct, −d) ⊙ mask_s（掩码正常 scale、同 level 累加、
    统一 rescale ⇒ 净耗 1 层）。

    模式 B 主线**不使用本原语**：布局重置由 P1 在转换出口（RECRYPT）编码时
    免费完成（docs/01 §7.4 出口协议补充）——本原语供模式 A / 无转换场景。
    注意：scale=1.0 的掩码编码会静默产生全零明文（P2 实测），掩码必须用
    正常 scale 编码。"""
    d = spec.block_elems
    mask_f = _mask(spec, first_half=True)
    mask_s = _mask(spec, first_half=False)
    a = ctx.multiply_plain(ct, mask_f, rescale=False)
    b = ctx.multiply_plain(ctx.rotate(ct, -d), mask_s, rescale=False)
    return ctx.rescale_next(ctx.add(a, b))


def linear_cipher(ctx: CKKSContext, ct: CKKSCiphertext,
                  weight: Sequence[Sequence[float]], num_tokens: int,
                  spec: PackingSpec = DEFAULT_SPEC,
                  diagonals: Optional[List[List[float]]] = None
                  ) -> CKKSCiphertext:
    """密文线性层 y_t = W·x_t（全部 token SIMD 并行），深度 +1。

    BSGS：baby=⌈√d⌉ 组内旋转 + giant 组间旋转；d 次明文乘同 level 累加，
    末尾统一 rescale。旋转次数 = (baby-1)+(giant-1)，明文乘 = d。
    giant 组明文需**预旋转 +j·baby**（对角线明文以块周期周期化，flat-roll 后
    giant 旋转才与明文索引对齐——P2 实现修正，见工作记录）。"""
    spec.validate()
    d = spec.block_elems
    if num_tokens > spec.blocks_per_ct:
        raise ValueError(f"num_tokens={num_tokens} 超出单密文容量 {spec.blocks_per_ct}")
    diags = diagonals if diagonals is not None else build_diagonal_plaintexts(weight, spec)
    baby = max(1, int(d ** 0.5))
    while d % baby != 0:
        baby += 1
    giant = d // baby

    # baby 步：x_k = rot(x, k)，k∈[0,baby)
    baby_rot = [ct]
    for k in range(1, baby):
        baby_rot.append(ctx.rotate(ct, k))
    # giant 组累加：明文预旋转 +j·baby（不逐乘 rescale，同 level 累加）
    acc: List[CKKSCiphertext] = []
    for j in range(giant):
        rolled = [np.roll(np.asarray(diags[j * baby + k]), j * baby).tolist()
                  for k in range(baby)]
        total = ctx.multiply_plain(baby_rot[0], rolled[0], rescale=False)
        for k in range(1, baby):
            total = ctx.add(total, ctx.multiply_plain(baby_rot[k], rolled[k],
                                                      rescale=False))
        acc.append(total)
    # giant 步：rot(acc_j, j·baby) 求和
    result = acc[0]
    for j in range(1, giant):
        result = ctx.add(result, ctx.rotate(acc[j], j * baby))
    # 统一 rescale（深度 +1；P1 表 2"每线性层净耗 1"成立）
    return ctx.rescale_next(result)
