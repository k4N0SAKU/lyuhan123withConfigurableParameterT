"""明文推理基线工作负载装配（F10）。

模型加载与推理逻辑的唯一实现在 src/model/loader.py；本文件只负责把管线包装成
PerfSession 可驱动的工作负载函数（含埋点约定），并注册空转自检负载。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

from src.common.perf import (SEG_AUTH, SEG_COMPUTE_LINEAR, SEG_COMPUTE_NONLINEAR,
                             SEG_DETOKENIZE, SEG_ENCRYPT, SEG_KEYNEG,
                             SEG_NOOP, SEG_THRESH_DECRYPT, SEG_TOKENIZE,
                             SEG_TRANSPORT, RoundContext)
from src.model.loader import (BertSentimentPipeline, GPT2GreedyPipeline,
                              resolve_model_path)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "demo"
SENTIMENT_DIR = REPO_ROOT / "data" / "sentiment"

# 协议占位分段（P0：仅验证埋点通路；P3/P4 由真实协议阶段填充）
PROTOCOL_PLACEHOLDER_SEGMENTS = (
    SEG_AUTH, SEG_KEYNEG, SEG_ENCRYPT, SEG_TRANSPORT,
    SEG_COMPUTE_LINEAR, SEG_COMPUTE_NONLINEAR, SEG_THRESH_DECRYPT,
)


def load_lines(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_eval_rows(limit: int = 0) -> List[tuple]:
    """读取 eval.tsv 前 limit 条（limit=0 表示全部），返回 [(label, text)]。"""
    rows = []
    for ln in (SENTIMENT_DIR / "eval.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        if not ln.strip():
            continue
        _sid, label, text = ln.split("\t")
        rows.append((int(label), text))
        if limit and len(rows) >= limit:
            break
    return rows


def make_gpt2_workload(baseline: GPT2GreedyPipeline, prompts: List[str],
                       max_new_tokens: int):
    """每轮处理一条 prompt（轮转），逐 token 解码埋点。"""

    def run_once(ctx: RoundContext) -> None:
        prompt = prompts[ctx.index % len(prompts)]
        baseline.generate(prompt, max_new_tokens, ctx)

    return run_once


def make_bert_workload(baseline: BertSentimentPipeline, sentences: List[str]):
    """每轮预测一条句子（batch=1，与单条推理时延口径一致）。"""

    def run_once(ctx: RoundContext) -> None:
        sentence = sentences[ctx.index % len(sentences)]
        baseline.predict([sentence], ctx)

    return run_once


def make_dryrun_workload():
    """空转自检（P0 出口条件 3）：逐一验证全部标准分段（含协议占位段）与
    网络/内存埋点通路。不代表任何真实推理或协议口径。"""

    def run_once(ctx: RoundContext) -> None:
        for name in PROTOCOL_PLACEHOLDER_SEGMENTS + (SEG_NOOP,):
            with ctx.timer.segment(name):
                time.sleep(0.0005)
        ctx.net.on_send(0)
        ctx.net.on_recv(0)

    return run_once


__all__ = ["BertSentimentPipeline", "GPT2GreedyPipeline", "resolve_model_path",
           "load_lines", "load_eval_rows", "make_gpt2_workload",
           "make_bert_workload", "make_dryrun_workload",
           "PROTOCOL_PLACEHOLDER_SEGMENTS"]
