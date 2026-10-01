"""秘密分享单元测试（P2）：环上代数 + 统计均匀性（单份额无信息）。"""
from __future__ import annotations

import math

import pytest

from src.crypto.secret_sharing import (DEFAULT_MODULUS, additive_share,
                                       reconstruct, sample_fresh_mask,
                                       share_add, share_dot_const,
                                       share_mul_const, share_vector)


class TestAlgebra:
    def test_scalar_share_reconstruct(self):
        for v in (0, 1, 42, 2**63, DEFAULT_MODULUS - 1):
            s0, s1 = additive_share(v)
            assert reconstruct(s0, s1) == v % DEFAULT_MODULUS

    def test_vector_share(self):
        vals = [1, 2, 3, 2**63]
        s0, s1 = share_vector(vals)
        assert [reconstruct(a, b) for a, b in zip(s0, s1)] == vals

    def test_share_add_local(self):
        x0, x1 = additive_share(100)
        y0, y1 = additive_share(23)
        z0, z1 = share_add([x0], [y0])[0], share_add([x1], [y1])[0]
        assert reconstruct(z0, z1) == 123                # 加法零交互

    def test_share_mul_const_local(self):
        x0, x1 = additive_share(100)
        z0, z1 = share_mul_const([x0], 3)[0], share_mul_const([x1], 3)[0]
        assert reconstruct(z0, z1) == 300                # 公开常数乘零交互

    def test_share_dot_const(self):
        xs = [5, 7]
        x0, x1 = share_vector(xs)
        c = [2, 3]
        lhs = share_dot_const(x0, c) + share_dot_const(x1, c)
        assert lhs % DEFAULT_MODULUS == 31               # ⟨x, c⟩ = 10+21

    def test_reconstruct_is_modular(self):
        s0, s1 = additive_share(5)
        assert reconstruct(s0 + DEFAULT_MODULUS, s1) == 5


class TestUniformity:
    """统计均匀性：单份额分布无信息（P2 任务书要求）。

    - 比特频率检验：每比特 0/1 频率与 0.5 的偏差需在二项检验 4σ 内；
    - 高位字节分桶卡方：256 桶，样本 25600，期望 100/桶，χ² < 上限。
    """

    def test_bit_frequency(self):
        """4σ 二项检验界（64 比特同时检验，3σ 会有 ~17% 偶发率）。"""
        n = 12800
        shares0, _ = zip(*(additive_share(i * 7919) for i in range(n)))
        ones = [0] * 64
        for s in shares0:
            for b in range(64):
                ones[b] += (s >> b) & 1
        expected = n / 2
        sigma = math.sqrt(n * 0.25)
        for b, cnt in enumerate(ones):
            assert abs(cnt - expected) <= 4 * sigma, \
                f"bit {b} 频率 {cnt}/{n} 偏离均匀（4σ 二项检验）"

    def test_high_byte_chi_square(self):
        n = 25600
        masks = sample_fresh_mask(n)
        buckets = [0] * 256
        for m in masks:
            buckets[(m >> 56) & 0xFF] += 1
        expected = n / 256
        chi2 = sum((o - expected) ** 2 / expected for o in buckets)
        # 自由度 255，χ²(255) 的 0.1% 分位约 325；均匀性被破坏时会显著超界
        assert chi2 < 400, f"卡方 {chi2:.1f} 超界——单份额分布非均匀"

    def test_single_share_reveals_nothing_about_value(self):
        """固定明文的份额 0 分布应与任意明文的份额 0 分布不可区分（同检验）。"""
        b0 = [0] * 256
        b1 = [0] * 256
        for i in range(12800):
            s0, _ = additive_share(0)
            b0[(s0 >> 56) & 0xFF] += 1
            t0, _ = additive_share(DEFAULT_MODULUS - 1)
            b1[(t0 >> 56) & 0xFF] += 1
        e = 12800 / 256
        chi0 = sum((o - e) ** 2 / e for o in b0)
        chi1 = sum((o - e) ** 2 / e for o in b1)
        assert chi0 < 400 and chi1 < 400
