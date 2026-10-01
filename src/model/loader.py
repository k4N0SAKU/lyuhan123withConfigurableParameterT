"""模型加载与推理管线（决策 D4：GPT-2 124M 生成 + BERT-base-chinese 分类）。

本模块是模型加载与解码逻辑的唯一实现（benchmarks 的基线与评估脚本均调用这里）。
管线支持可选的模拟定点化（见 quantize.py）：
- quantizer=None：纯 FP32 基线口径；
- 传入 FixedPointQuantizer：权重档位量化 + 激活定点 Q/DQ。

所有推理使用 ``torch.inference_mode``，贪心解码保证确定性（同输入同输出）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer)

from src.common.perf import SEG_DETOKENIZE, SEG_TOKENIZE, RoundContext
from src.model.quantize import (FixedPointQuantizer, encoder_layer_outputs,
                                quantize_model_weights, register_activation_qdq)

REPO_ROOT = Path(__file__).resolve().parents[2]
# 模型目录可经 A122_MODELS_DIR 覆盖（P6 复现脚本：干净 venv 复用本地模型数据，
# 模型是数据不是代码——环境洁净性不受影响，docs/06 §2）
import os as _os
MODELS_DIR = Path(_os.environ.get("A122_MODELS_DIR",
                                  str(REPO_ROOT / "data" / "models")))
SENTIMENT_FT_DIR = MODELS_DIR / "bert-base-chinese-sentiment"

LABEL_MAP = {0: "负面", 1: "正面"}


def resolve_model_path(name: str) -> str:
    """优先使用仓库本地已下载模型（data/models/<name>），否则回退 HF hub id。"""
    local = MODELS_DIR / name
    return str(local) if (local / "config.json").exists() else name


def _noop_ctx() -> RoundContext:
    return RoundContext(index=0)


class GPT2GreedyPipeline:
    """GPT-2 贪心解码生成管线（FP32 或模拟定点化）。

    默认权重方案（eval_quantize 实测校准的档位阶梯，见 docs/01）：
    - 权重 q22：贪心解码对权重扰动敏感（INT8 一致率 0.89、FP16 0.944 均
      低于 0.95 阈值），Q22（步长 2^-22）实测恢复；CKKS 明文权重编码本就
      保留 ~2^-40 相对精度，权重无需压至 INT8/FP16；
    - 激活 Q16（quantizer 参数）：与 CKKS 输入编码 scale 规划一致。"""

    def __init__(self, model_name: str = "gpt2",
                 quantizer: Optional[FixedPointQuantizer] = None,
                 embedding_scheme: str = "q22",
                 linear_scheme: str = "q22") -> None:
        self.model_name = model_name
        self.quantizer = quantizer
        path = resolve_model_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.model.eval()
        self.eos_id = self.tokenizer.eos_token_id
        self.quant_stats: dict = {}
        if self.quantizer is not None:
            self.quant_stats = quantize_model_weights(
                self.model, embedding_scheme=embedding_scheme,
                linear_scheme=linear_scheme)
            # 激活 Q/DQ 边界：每个编码器层输出 + 最终 LayerNorm + lm_head（logits）
            targets = encoder_layer_outputs(self.model)
            targets.append(self.model.transformer.ln_f)
            targets.append(self.model.lm_head)
            self._handles = register_activation_qdq(self.model, self.quantizer, targets)

    def generate(self, prompt: str, max_new_tokens: int,
                 ctx: Optional[RoundContext] = None) -> dict:
        """一轮生成：tokenize → 逐 token 贪心解码 → 反解码，全部经 ctx 埋点。"""
        ctx = ctx or _noop_ctx()
        with ctx.timer.segment(SEG_TOKENIZE):
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        generated = input_ids
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                with ctx.timer.segment("decode_step"):
                    logits = self.model(generated).logits[:, -1, :]
                    next_id = int(torch.argmax(logits, dim=-1).item())
                generated = torch.cat(
                    [generated, torch.tensor([[next_id]], dtype=generated.dtype)], dim=1)
                if next_id == self.eos_id:
                    break

        with ctx.timer.segment(SEG_DETOKENIZE):
            text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return {"prompt": prompt, "output": text,
                "num_new_tokens": int(generated.shape[1] - input_ids.shape[1]),
                "token_ids": generated[0].tolist()}


class BertSentimentPipeline:
    """BERT-base-chinese 中文情感二分类管线（FP32 或模拟定点化）。

    分类头权重来自 finetune_bert.py 产出的检查点（data/models/bert-base-chinese-sentiment）。
    """

    def __init__(self, model_path: Optional[str] = None,
                 quantizer: Optional[FixedPointQuantizer] = None) -> None:
        self.quantizer = quantizer
        path = str(model_path) if model_path else str(SENTIMENT_FT_DIR)
        if not (Path(path) / "config.json").exists():
            raise FileNotFoundError(
                f"未找到微调检查点 {path}；请先运行 python -m benchmarks.finetune_bert")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()
        self.quant_stats: dict = {}
        if self.quantizer is not None:
            self.quant_stats = quantize_model_weights(
                self.model, linear_scheme="int8", embedding_scheme="int8")
            # 激活 Q/DQ 边界：embeddings 输出 + 每个编码器层输出
            targets = [self.model.bert.embeddings] + encoder_layer_outputs(self.model)
            self._handles = register_activation_qdq(self.model, self.quantizer, targets)

    def predict(self, texts: list, ctx: Optional[RoundContext] = None) -> list:
        """批量预测：返回 [{"label": "正面"/"负面", "prob": float}]，顺序与输入一致。"""
        if not texts:
            return []
        ctx = ctx or _noop_ctx()
        with ctx.timer.segment(SEG_TOKENIZE):
            enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=64)

        with ctx.timer.segment("model_forward"):
            with torch.inference_mode():
                enc_out = self.model.bert(**enc)
                pooled = enc_out.pooler_output
                if self.quantizer is not None:
                    pooled = self.quantizer.qdq(pooled)
                logits = self.model.classifier(pooled)
                if self.quantizer is not None:
                    logits = self.quantizer.qdq(logits)
                probs = torch.softmax(logits, dim=-1)

        with ctx.timer.segment(SEG_DETOKENIZE):
            results = []
            for row in probs:
                idx = int(torch.argmax(row).item())
                results.append({"label": LABEL_MAP[idx], "prob": float(row[idx])})
        return results
