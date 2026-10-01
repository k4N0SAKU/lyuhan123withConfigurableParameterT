"""定点化模块单元测试（含溢出断言——P0 陷阱清单要求）。"""
from __future__ import annotations

import pytest
import torch

from src.model.quantize import (FixedPointQuantizer, INT32_MAX,
                                _scheme_apply, fp16_qdq, per_channel_int8_qdq,
                                quantize_model_weights, register_activation_qdq)


class TestFixedPointQuantizer:
    def test_qdq_error_bound_half_step(self):
        torch.manual_seed(0)
        x = torch.empty(10000).uniform_(-100, 100)
        for frac_bits in (13, 16):
            q = FixedPointQuantizer(frac_bits=frac_bits)
            err = (q.qdq(x) - x).abs().max().item()
            assert err <= 2 ** -(frac_bits + 1) + 1e-12

    def test_to_int_on_grid(self):
        q = FixedPointQuantizer(frac_bits=13)
        ints = q.to_int(torch.tensor([0.5, -0.25, 3.0]))
        assert ints.dtype == torch.float64  # 整数域用 float64 承载防溢出
        assert torch.equal(ints, torch.tensor([4096.0, -2048.0, 24576.0]))

    def test_overflow_raises(self):
        q = FixedPointQuantizer(frac_bits=13, on_overflow="raise")
        # 3e5 · 2^13 ≈ 2.46e9 > int32 上限 2^31-1
        with pytest.raises(OverflowError):
            q.qdq(torch.tensor([3.0e5]))
        assert q.stats.overflow_raises == 1

    def test_overflow_clamp(self):
        q = FixedPointQuantizer(frac_bits=13, on_overflow="clamp")
        out = q.qdq(torch.tensor([3.0e5, 1.0]))
        assert out[0] == pytest.approx(INT32_MAX / q.scale)
        assert out[1] == pytest.approx(1.0)
        assert q.stats.overflow_clamps == 1

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            FixedPointQuantizer(frac_bits=0)
        with pytest.raises(ValueError):
            FixedPointQuantizer(frac_bits=31)
        with pytest.raises(ValueError):
            FixedPointQuantizer(on_overflow="drop")

    def test_max_representable(self):
        q = FixedPointQuantizer(frac_bits=13)
        assert q.max_representable == pytest.approx((2**31 - 1) / 2**13)


class TestPerChannelInt8:
    def test_linear_layout_per_output_channel(self):
        torch.manual_seed(1)
        w = torch.empty(64, 32).uniform_(-0.1, 0.1)
        wq = per_channel_int8_qdq(w, out_dim=0)
        row_amax = w.abs().amax(dim=1, keepdim=True)
        bound = (row_amax / 127.0 / 2 + 1e-12)  # 每行 scale/2
        assert ((wq - w).abs() <= bound).all()

    def test_conv1d_layout_per_output_channel(self):
        torch.manual_seed(2)
        w = torch.empty(32, 48).uniform_(-0.2, 0.2)  # transformers Conv1D: (in, out)
        wq = per_channel_int8_qdq(w, out_dim=1)
        col_amax = w.abs().amax(dim=0, keepdim=True)
        bound = col_amax / 127.0 / 2 + 1e-12
        assert ((wq - w).abs() <= bound).all()

    def test_values_on_int8_grid(self):
        torch.manual_seed(3)
        w = torch.empty(8, 8).uniform_(-0.5, 0.5)
        wq = per_channel_int8_qdq(w, out_dim=0)
        row_amax = w.abs().amax(dim=1, keepdim=True)
        codes = torch.round(w / (row_amax / 127.0))
        assert (codes.abs() <= 127).all()


class TestQuantizeModelWeights:
    def _small_model(self):
        return torch.nn.ModuleDict({
            "fc": torch.nn.Linear(8, 8, bias=False),
            "emb": torch.nn.Embedding(8, 8),
        })

    def test_tied_weights_quantized_once(self):
        model = self._small_model()
        model["emb"].weight = model["fc"].weight  # 模拟 GPT-2 绑定权重
        stats = quantize_model_weights(model, linear_scheme="int8",
                                       embedding_scheme="int8")
        assert stats["tied_skipped"] == 1
        assert model["fc"].weight.data_ptr() == model["emb"].weight.data_ptr()

    def test_none_scheme_identity(self):
        model = self._small_model()
        before = model["fc"].weight.data.clone()
        quantize_model_weights(model, linear_scheme="none", embedding_scheme="none")
        assert torch.equal(model["fc"].weight.data, before)

    def test_q22_scheme_fine_grid(self):
        model = self._small_model()
        w0 = model["fc"].weight.data.clone()
        quantize_model_weights(model, linear_scheme="q22", embedding_scheme="q22")
        err = (model["fc"].weight.data - w0).abs().max().item()
        assert err <= 2 ** -23 + 1e-15

    def test_unknown_scheme_rejected(self):
        model = self._small_model()
        with pytest.raises(ValueError):
            quantize_model_weights(model, linear_scheme="int4", embedding_scheme="int8")

    def test_scheme_apply_dispatch(self):
        w = torch.tensor([[0.02, -0.03]])
        assert torch.equal(_scheme_apply("none", w, out_dim=0), w)
        assert torch.equal(_scheme_apply("fp16", w, out_dim=0),
                           fp16_qdq(w))  # fp16 往返


class TestActivationQdqHook:
    def test_tensor_output(self):
        q = FixedPointQuantizer(frac_bits=13)
        lin = torch.nn.Linear(4, 4)
        handles = register_activation_qdq(lin, q, [lin])
        x = torch.randn(2, 4)
        out = lin(x)
        # hook 作用于模块输出：out == qdq(Linear(x))
        assert torch.allclose(out, q.qdq(x @ lin.weight.T + lin.bias), atol=1e-12)
        for h in handles:
            h.remove()
        assert torch.allclose(lin(x), x @ lin.weight.T + lin.bias, atol=1e-12)

    def test_tuple_output(self):
        q = FixedPointQuantizer(frac_bits=13)

        class TupleModule(torch.nn.Module):
            def forward(self, x):
                return x, x

        mod = TupleModule()
        handles = register_activation_qdq(mod, q, [mod])
        x = torch.randn(3)
        a, b = mod(x)
        assert torch.allclose(a, q.qdq(x), atol=1e-12)
        assert torch.equal(b, x)  # tuple 其余元素不动
        for h in handles:
            h.remove()
