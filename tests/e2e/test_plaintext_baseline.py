"""明文推理管线端到端测试（F10；依赖已下载模型与微调检查点，未就绪时跳过）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.common.perf import SEG_DETOKENIZE, SEG_TOKENIZE, RoundContext
from src.model.quantize import FixedPointQuantizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "data" / "models"

gpt2_available = (MODELS_DIR / "gpt2" / "config.json").exists()
ft_available = (MODELS_DIR / "bert-base-chinese-sentiment" / "config.json").exists()

pytestmark = pytest.mark.slow


def _ctx() -> RoundContext:
    return RoundContext(index=0)


@pytest.mark.skipif(not gpt2_available, reason="gpt2 未下载")
class TestGPT2Pipeline:
    def test_greedy_generation_deterministic(self):
        from src.model.loader import GPT2GreedyPipeline, resolve_model_path
        pipe = GPT2GreedyPipeline(resolve_model_path("gpt2"))
        r1 = pipe.generate("The capital of France is", 4)
        r2 = pipe.generate("The capital of France is", 4)
        assert r1["num_new_tokens"] == 4
        assert r1["token_ids"] == r2["token_ids"]  # 贪心解码确定性
        assert len(r1["output"]) > len("The capital of France is")

    def test_segments_recorded(self):
        from src.model.loader import GPT2GreedyPipeline, resolve_model_path
        pipe = GPT2GreedyPipeline(resolve_model_path("gpt2"))
        ctx = _ctx()
        pipe.generate("Hello world", 3, ctx=ctx)
        seg = ctx.timer.samples_ms()
        assert set(seg) >= {SEG_TOKENIZE, "decode_step", SEG_DETOKENIZE}
        assert len(seg["decode_step"]) == 3

    def test_quantized_pipeline_runs(self):
        from src.model.loader import GPT2GreedyPipeline, resolve_model_path
        pipe = GPT2GreedyPipeline(resolve_model_path("gpt2"),
                                  quantizer=FixedPointQuantizer(frac_bits=16))
        assert pipe.quant_stats["conv1d"] == 48 and pipe.quant_stats["tied_skipped"] == 1
        result = pipe.generate("Once upon a time", 8)
        assert result["num_new_tokens"] == 8


@pytest.mark.skipif(not ft_available, reason="BERT 情感检查点未微调")
class TestBertSentimentPipeline:
    def test_predict_labels(self):
        from src.model.loader import BertSentimentPipeline
        pipe = BertSentimentPipeline()
        preds = pipe.predict(["办事大厅服务热情周到，一次办好",
                              "排了三小时队还被要求补材料"])
        assert all(p["label"] in ("正面", "负面") for p in preds)
        assert all(0.0 <= p["prob"] <= 1.0 for p in preds)
        assert preds[0]["label"] == "正面" and preds[1]["label"] == "负面"

    def test_missing_checkpoint_message(self, tmp_path):
        from src.model.loader import BertSentimentPipeline
        with pytest.raises(FileNotFoundError):
            BertSentimentPipeline(model_path=str(tmp_path / "nope"))

    def test_quantized_accuracy_on_subset(self):
        from benchmarks.baselines.plaintext_baseline import load_eval_rows
        from src.model.loader import BertSentimentPipeline
        rows = load_eval_rows(limit=20)
        texts = [t for _, t in rows]
        pipe = BertSentimentPipeline(quantizer=FixedPointQuantizer(frac_bits=16))
        preds = pipe.predict(texts)
        correct = sum(1 for p, (y, _) in zip(preds, rows)
                      if p["label"] == ("正面" if y == 1 else "负面"))
        assert correct / len(rows) >= 0.9
