"""打包方案单元测试（P2）：gap-block 布局 / BSGS 线性层 / 层间复制变换。

小型 spec（d=16）验证正确性逻辑；真实 shape（d=768, 2^15）为 slow 用例。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.crypto.ckks_ops import CKKSContext, PARAMS_MODE_B
from src.model.ops.packing import (DEFAULT_SPEC, SMALL_SPEC, PackingScheme,
                                   PackingSpec, build_diagonal_plaintexts,
                                   layout_input, linear_cipher, replicate_blocks)


class TestSpec:
    def test_default_spec_real_shape(self):
        assert DEFAULT_SPEC.block_elems == 768
        assert DEFAULT_SPEC.block_slots == 1536
        assert DEFAULT_SPEC.blocks_per_ct == 10          # 16384/1536
        assert DEFAULT_SPEC.slots == 1 << 14             # 本栈实测上限

    def test_packing_scheme_default(self):
        assert int(PackingScheme.DIAGONAL_BSGS) == 1

    def test_invalid_spec(self):
        with pytest.raises(ValueError):
            PackingSpec(gap=1).validate()
        with pytest.raises(ValueError):
            PackingSpec(block_elems=2048, slots=2048).validate()  # 块槽超出槽数


class TestSmallShape:
    """d=16 小型 spec：快速验证对角线/布局/复制变换的代数正确性。"""

    def test_diagonal_definition(self):
        rng = np.random.default_rng(0)
        W = rng.normal(0, 0.1, (16, 16))
        diags = build_diagonal_plaintexts(W, SMALL_SPEC)
        assert len(diags) == 16
        # d_k[i] = W[i][(i+k)%d]（块前段），后段为 0
        for k in (0, 5, 15):
            dk = np.asarray(diags[k]).reshape(SMALL_SPEC.blocks_per_ct,
                                              SMALL_SPEC.block_slots)
            assert all(abs(dk[0, i] - W[i][(i + k) % 16]) < 1e-12 for i in range(16))
            assert np.all(dk[:, 16:] == 0.0)

    def test_linear_vs_numpy(self):
        ctx = CKKSContext(PARAMS_MODE_B)
        rng = np.random.default_rng(1)
        W = rng.normal(0, 0.05, (16, 16))
        tokens = [rng.normal(0, 0.5, 16) for _ in range(3)]
        layout = layout_input(tokens, 3, SMALL_SPEC)
        ct = ctx.encrypt_vector(layout)
        out = linear_cipher(ctx, ct, W, 3, spec=SMALL_SPEC)
        got = np.array(ctx.decrypt(out)[:SMALL_SPEC.slots])
        for b in range(3):
            ref = W @ np.asarray(tokens[b])
            assert np.allclose(got[b * SMALL_SPEC.block_slots:
                                   b * SMALL_SPEC.block_slots + 16],
                               ref, atol=1e-5), f"block {b} 不匹配"
        # 注：明文预旋转（BSGS giant 组）后，块后半槽不再恒为 0——
        # 上层只需读取块前 d 槽（replicate_blocks 语义），后段内容无影响

    def test_two_layer_composition_with_replicate(self):
        """密文域 replicate（净耗 1 层）+ 两层线性：模式 A 场景的布局衔接验证。

        使用 19 层深链上下文（replicate 需要额外深度；模式 B 主线在转换出口
        由 P1 编码完成布局重置，不耗链深——见 replicate_blocks docstring）。"""
        from src.crypto.ckks_ops import PARAMS_DEEP19
        ctx = CKKSContext(PARAMS_DEEP19)
        rng = np.random.default_rng(2)
        W1 = rng.normal(0, 0.05, (16, 16))
        W2 = rng.normal(0, 0.05, (16, 16))
        tokens = [rng.normal(0, 0.5, 16) for _ in range(2)]
        x1 = np.asarray(tokens[0])
        ref1 = W1 @ x1
        ref2 = W2 @ ref1
        ct = ctx.encrypt_vector(layout_input(tokens, 2, SMALL_SPEC))
        y1 = linear_cipher(ctx, ct, W1, 2, spec=SMALL_SPEC)          # level 1
        x2_ct = replicate_blocks(ctx, y1, SMALL_SPEC)                # level 2
        y2 = linear_cipher(ctx, x2_ct, W2, 2, spec=SMALL_SPEC)       # level 3
        got = np.array(ctx.decrypt(y2)[:SMALL_SPEC.slots])
        assert np.allclose(got[:16], ref2, atol=1e-4), "两层复合与 numpy 不一致"
        assert y2.level == 3                                          # 1+1+1


@pytest.mark.slow
class TestRealShape:
    """真实模型 shape（GPT-2 hidden=768）：单层密文矩阵乘 vs numpy。"""

    def test_linear_768(self):
        ctx = CKKSContext(PARAMS_MODE_B)
        rng = np.random.default_rng(3)
        W = rng.normal(0, 0.05, (768, 768))
        tokens = [rng.normal(0, 0.5, 768) for _ in range(2)]
        ct = ctx.encrypt_vector(layout_input(tokens, 2, DEFAULT_SPEC))
        out = linear_cipher(ctx, ct, W, 2, spec=DEFAULT_SPEC)
        got = np.array(ctx.decrypt(out)[:DEFAULT_SPEC.slots])
        for b in range(2):
            ref = W @ np.asarray(tokens[b])
            assert np.allclose(got[b * DEFAULT_SPEC.block_slots:
                                   b * DEFAULT_SPEC.block_slots + 768],
                               ref, atol=1e-4), f"block {b} 不匹配"
        assert out.level == 1
