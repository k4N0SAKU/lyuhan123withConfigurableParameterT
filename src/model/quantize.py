"""模拟定点化与权重量化（决策 D4；P0 明文域预研，P2-R1 校准）。

口径与边界（S8 诚实声明）：
- 本模块是 **模拟** 定点化：量化-反量化（Q/DQ）后仍以浮点执行，用于在明文域
  评估定点化带来的精度损失；真实整型计算内核与 CKKS 编码在 P2/P3 接入。
- 权重档位阶梯（eval_quantize 实测）：INT8 per-channel / FP16 / Q22 定点。
- 激活：固定小数位定点化，scale = 2^frac_bits（起步 13，默认 16），整数域
  以 int32 承载，可表示范围 ±2^(31-frac_bits)。越界按策略处理（默认抛
  OverflowError，即"溢出断言"），确保定点范围不足时显式失败而非静默失真。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch

INT32_MAX = 2**31 - 1


@dataclass
class QuantizeStats:
    """量化过程统计（供评估报告如实记录）。"""

    overflow_raises: int = 0
    overflow_clamps: int = 0
    clamp_events: List[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.clamp_events is None:
            self.clamp_events = []


class FixedPointQuantizer:
    """固定小数位定点化模拟器（权重与激活共用）。

    参数：
        frac_bits: 小数位数，scale = 2^frac_bits（起步 13，可配置）。
        on_overflow: "raise" 抛 OverflowError（默认，断言语义）；
                     "clamp" 饱和截断并计数（用于探测量级上界）。
    """

    def __init__(self, frac_bits: int = 13, on_overflow: str = "raise",
                 stats: QuantizeStats | None = None) -> None:
        if frac_bits < 1 or frac_bits > 30:
            raise ValueError("frac_bits 必须在 [1, 30]")
        if on_overflow not in ("raise", "clamp"):
            raise ValueError('on_overflow 只能是 "raise" 或 "clamp"')
        self.frac_bits = frac_bits
        self.scale = float(2 ** frac_bits)
        self.on_overflow = on_overflow
        self.stats = stats if stats is not None else QuantizeStats()

    @property
    def max_representable(self) -> float:
        """定点可表示的最大绝对值（int32 整数域换算回实数域）。"""
        return INT32_MAX / self.scale

    def to_int(self, x: torch.Tensor) -> torch.Tensor:
        """量化到 int32 整数域（round(scale·x)），按策略处理溢出。"""
        scaled = x.to(torch.float64) * self.scale
        if self.on_overflow == "raise":
            over = scaled.abs().max().item() if scaled.numel() else 0.0
            if over > INT32_MAX:
                self.stats.overflow_raises += 1
                raise OverflowError(
                    f"定点溢出：|x|·2^{self.frac_bits} 最大为 {over:.0f}，"
                    f"超出 int32 上限 {INT32_MAX}；应增大 frac_bits 或使用 clamp 探测")
        else:  # clamp
            over_mask = scaled.abs() > INT32_MAX
            n_over = int(over_mask.sum().item())
            if n_over:
                self.stats.overflow_clamps += n_over
                self.stats.clamp_events.append(n_over)
                scaled = torch.clamp(scaled, -INT32_MAX, INT32_MAX)
        return torch.round(scaled)

    def qdq(self, x: torch.Tensor) -> torch.Tensor:
        """量化-反量化（Q/DQ）：返回模拟定点化后的浮点张量。"""
        return self.to_int(x).to(torch.float32) / self.scale


def per_channel_int8_qdq(weight: torch.Tensor, out_dim: int) -> torch.Tensor:
    """对称 per-channel INT8 权重 Q/DQ。

    out_dim 指输出通道所在维度；scale 按输出通道计算（对除 out_dim 外的所有
    维度取 amax）。典型布局：nn.Linear/nn.Embedding 权重 (out, in) 取 out_dim=0；
    transformers Conv1D 权重 (in, out) 取 out_dim=1。scale = max|w|/127，
    量化到 [-127, 127]（对称域不用 -128）。"""
    if weight.dim() < 2 or out_dim >= weight.dim():
        raise ValueError(f"非法 out_dim={out_dim}（weight.dim={weight.dim()}）")
    reduce_dims = [d for d in range(weight.dim()) if d != out_dim]
    amax = weight.abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.clamp(amax / 127.0, min=1e-12)
    q = torch.round(weight / scale).clamp(-127.0, 127.0)
    return q * scale


def fp16_qdq(weight: torch.Tensor) -> torch.Tensor:
    """FP16 模拟定点：经半精度往返舍入（相对误差约 5e-4），用于嵌入/输出投影。"""
    return weight.to(torch.float16).to(torch.float32)


_WEIGHT_SCHEMES = ("int8", "fp16", "q22", "none")


def _scheme_apply(scheme: str, weight: torch.Tensor, out_dim: int) -> torch.Tensor:
    """按方案对权重做 Q/DQ；out_dim 仅对 int8 有意义。"""
    if scheme == "int8":
        return per_channel_int8_qdq(weight, out_dim=out_dim)
    if scheme == "fp16":
        return fp16_qdq(weight)
    if scheme == "q22":
        # Q22 定点：步长 2^-22≈2.4e-7，int32 域可表示 |w|≤512，覆盖权重幅值
        # （实测 GPT-2 各类权重 amax<0.7）；用于贪心解码敏感性场景，见 docs/01
        return FixedPointQuantizer(frac_bits=22).qdq(weight)
    if scheme == "none":
        return weight
    raise ValueError(f"未知权重方案: {scheme}（可选 {_WEIGHT_SCHEMES}）")


def quantize_model_weights(model: torch.nn.Module,
                           linear_scheme: str = "int8",
                           embedding_scheme: str = "int8") -> dict:
    """对模型权重做模拟定点量化（原地），返回统计。

    - nn.Linear / transformers Conv1D：linear_scheme（int8/fp16/q22/none）；
    - nn.Embedding：embedding_scheme（同上；GPT-2 的 wte 与 lm_head 为绑定
      权重，嵌入精度直接决定 logits 精度）；
    - 绑定权重（同一 Parameter 对象）只量化一次；
    - bias 不量化（与 INT8 推理实践中 bias 常驻 int32 高精度的口径一致）。"""
    for s in (linear_scheme, embedding_scheme):
        if s not in _WEIGHT_SCHEMES:
            raise ValueError(f"未知权重方案: {s}（可选 {_WEIGHT_SCHEMES}）")
    stats = {"linear": 0, "conv1d": 0, "embedding": 0, "tied_skipped": 0}
    seen_params: set = set()
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if id(weight) in seen_params:
            stats["tied_skipped"] += 1
            continue
        seen_params.add(id(weight))
        name = type(module).__name__
        if name == "Linear" and weight.dim() == 2:
            module.weight.data = _scheme_apply(linear_scheme, weight.data, out_dim=0)
            stats["linear"] += 1
        elif name == "Conv1D" and weight.dim() == 2:
            module.weight.data = _scheme_apply(linear_scheme, weight.data, out_dim=1)
            stats["conv1d"] += 1
        elif name == "Embedding" and weight.dim() == 2:
            module.weight.data = _scheme_apply(embedding_scheme, weight.data, out_dim=0)
            stats["embedding"] += 1
    return stats


def _qdq_forward_hook(quantizer: FixedPointQuantizer):
    """构造对模块输出做 Q/DQ 的 forward hook（自动适配 tuple/对象输出）。"""

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (quantizer.qdq(output[0]),) + tuple(output[1:])
        if hasattr(output, "last_hidden_state"):  # BaseModelOutput 等
            output.last_hidden_state = quantizer.qdq(output.last_hidden_state)
            return output
        return quantizer.qdq(output)

    return hook


def register_activation_qdq(model: torch.nn.Module,
                            quantizer: FixedPointQuantizer,
                            modules: Iterable[torch.nn.Module]) -> list:
    """在给定模块上注册激活 Q/DQ hook，返回可移除的 handle 列表。"""
    handle = _qdq_forward_hook(quantizer)
    return [m.register_forward_hook(handle) for m in modules]


def encoder_layer_outputs(model: torch.nn.Module) -> List[torch.nn.Module]:
    """按模型类型返回编码器层模块列表（GPT-2: transformer.h；BERT: bert.encoder.layer）。"""
    for attr in ("transformer", "bert", "roberta"):
        top = getattr(model, attr, None)
        if top is not None:
            layers = getattr(getattr(top, "encoder", None), "layer", None)
            if layers is not None:
                return list(layers)
            h = getattr(top, "h", None)
            if h is not None:
                return list(h)
    raise ValueError(f"未识别的模型结构: {type(model).__name__}")
