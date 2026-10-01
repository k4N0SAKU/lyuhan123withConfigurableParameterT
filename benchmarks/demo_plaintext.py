"""一条命令明文推理 Demo（P0 出口条件 1）：GPT-2 贪心生成 + BERT 情感分类。

用法（仓库根目录）：
    python -m benchmarks.demo_plaintext
    python -m benchmarks.demo_plaintext --prompt "Once upon a time" --text "今天的政务服务很贴心"
"""
from __future__ import annotations

import argparse
import sys

from benchmarks.baselines.plaintext_baseline import (BertSentimentPipeline,
                                                     GPT2GreedyPipeline,
                                                     resolve_model_path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--text", default=None, help="单条中文情感分类输入")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    print("== A1-22 明文推理 Demo（P0 基线口径）==")
    gpt2 = GPT2GreedyPipeline(resolve_model_path("gpt2"))
    result = gpt2.generate(args.prompt, args.max_new_tokens)
    print(f"\n[GPT-2 贪心生成] prompt: {args.prompt!r}")
    print(f"  output ({result['num_new_tokens']} tokens): {result['output']!r}")

    bert = BertSentimentPipeline()
    texts = [args.text] if args.text else [
        "今天的政务服务很贴心，办事一次就办好了",
        "排了两个小时队，材料清单和网上写的完全对不上",
        "这家店的服务还行，说不上惊艳但也没啥毛病",
    ]
    print("\n[BERT-base-chinese 情感分类]（自建演示集微调检查点）")
    for text, pred in zip(texts, bert.predict(texts)):
        print(f"  {text!r} -> {pred['label']} (p={pred['prob']:.3f})")
    print("\nDEMO OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
