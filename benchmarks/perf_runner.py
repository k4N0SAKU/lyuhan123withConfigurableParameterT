"""性能采集运行器（F9 的唯一数据入口，产出 benchmarks/results/*.json）。

用法（仓库根目录）：
    python -m benchmarks.perf_runner --workload dryrun --rounds 2
    python -m benchmarks.perf_runner --workload gpt2 --rounds 20 --max-new-tokens 16
    python -m benchmarks.perf_runner --workload bert --rounds 20

报告模式标注 ``plaintext-baseline``（密文模式报告由后续阶段接入本框架产出）；
文件名与 payload 时间戳统一为 UTC。
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA_DIR = REPO_ROOT / "data" / "demo"


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _require_models(names: list) -> None:
    missing = [n for n in names
               if not (REPO_ROOT / "data" / "models" / n / "config.json").exists()]
    if missing:
        print("ERROR: models/checkpoints not found: %s" % ", ".join(missing))
        print("run first: python -m benchmarks.download_models  "
              "#（BERT 检查点还需 python -m benchmarks.finetune_bert）")
        sys.exit(1)


def main(argv: list | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True,
                        choices=["dryrun", "gpt2", "bert"])
    parser.add_argument("--rounds", type=int, default=20,
                        help="统计轮数（F9 要求 ≥20）")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="GPT-2 每轮贪心解码的新 token 数")
    parser.add_argument("--keep-rounds", action="store_true", default=True,
                        help="报告中保留每轮原始数据")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出 JSON 路径；默认 benchmarks/results/<工作负载>_<模式>_<UTC时间戳>.json")
    args = parser.parse_args(argv)

    # 延迟导入：dryrun 不应拖入 torch
    from src.common.envinfo import collect_environment
    from src.common.perf import PerfSession, write_report

    env = collect_environment(probe_gpu=False)
    config: dict = {"rounds": args.rounds}

    if args.workload == "dryrun":
        from benchmarks.baselines.plaintext_baseline import make_dryrun_workload
        workload_fn, mode = make_dryrun_workload(), "dryrun-selfcheck"
    elif args.workload == "gpt2":
        _require_models(["gpt2"])
        from benchmarks.baselines.plaintext_baseline import (GPT2GreedyPipeline,
                                                             load_lines,
                                                             make_gpt2_workload,
                                                             resolve_model_path)
        prompts = load_lines(DATA_DIR / "prompts.txt")
        baseline = GPT2GreedyPipeline(resolve_model_path("gpt2"))
        config.update({"model": "gpt2",   # 逻辑名；解析后的本地路径不入报告（G 项）
                       "max_new_tokens": args.max_new_tokens,
                       "decoding": "greedy", "prompts": prompts,
                       "precision": "fp32"})
        workload_fn = make_gpt2_workload(baseline, prompts, args.max_new_tokens)
        mode = "plaintext-baseline"
    else:  # bert
        _require_models(["bert-base-chinese-sentiment"])
        from benchmarks.baselines.plaintext_baseline import (BertSentimentPipeline,
                                                             load_eval_rows,
                                                             make_bert_workload)
        rows = load_eval_rows(limit=20)
        sentences = [t for _, t in rows]
        baseline = BertSentimentPipeline()
        config.update({"model": "bert-base-chinese-sentiment",
                       "task": "sentiment-classification",
                       "eval_source": "data/sentiment/eval.tsv (前20条)",
                       "precision": "fp32"})
        workload_fn = make_bert_workload(baseline, sentences)
        mode = "plaintext-baseline"

    session = PerfSession(workload=args.workload, mode=mode, config=config,
                          environment=env, track_memory=True,
                          keep_rounds=args.keep_rounds)
    t0 = time.perf_counter()
    report = session.run(workload_fn, args.rounds)
    wall_s = time.perf_counter() - t0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (RESULTS_DIR / f"{args.workload}_{mode.replace('-', '_')}_"
                               f"{_utc_slug()}.json")
    write_report(report, out_path)

    summary = report["summary"]
    print(f"workload={args.workload} mode={mode} rounds={args.rounds} wall={wall_s:.1f}s")
    print(f"round_total: p50={summary['round_total']['p50']:.2f}ms "
          f"p95={summary['round_total']['p95']:.2f}ms mean={summary['round_total']['mean']:.2f}ms")
    for name, block in summary["segments"].items():
        print(f"  segment {name}: n={block['n']} p50={block['p50']:.2f}ms p95={block['p95']:.2f}ms")
    if "peak_delta_max" in summary.get("memory", {}):
        print(f"memory peak delta (max): {summary['memory']['peak_delta_max'] / 2**20:.1f} MiB")
    try:
        print(f"report -> {out_path.relative_to(REPO_ROOT)}")
    except ValueError:
        print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
