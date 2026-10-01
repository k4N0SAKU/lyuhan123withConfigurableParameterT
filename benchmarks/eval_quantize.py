"""定点化精度评估（P0 出口条件：分类下降 ≤1.5 个百分点、生成 token 一致率 ≥95%）。

对比口径：
- GPT-2：同一批 prompt（data/demo/prompts.txt）在 FP32 与
  "权重 Q22 定点 + 激活 Q16 定点"两种配置下分别贪心解码 16 token，
  统计 token 级一致率与整句一致率；
- BERT：eval.tsv 200 条上分别测 FP32 与定点化 accuracy，给出下降百分点。

结论写回 benchmarks/results/quantize_eval_<时间戳>.json，并对阈值做断言
（不达标退出码 1，run_ci 会拦截）。所有数字均来自本脚本实测（D8）。

用法：python -m benchmarks.eval_quantize [--frac-bits 16] [--max-new-tokens 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA_DIR = REPO_ROOT / "data" / "demo"

MAX_DROP_PP = 1.5      # 分类 accuracy 允许的最大下降（百分点）
MIN_AGREEMENT = 0.95   # 生成 token 一致率下限


def load_prompt_lines(path: Path) -> list:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def eval_gpt2_agreement(fp_model, q_model, prompts: list, max_new_tokens: int) -> dict:
    total = agree = seq_agree = 0
    examples = []
    for i, prompt in enumerate(prompts):
        r_fp = fp_model.generate(prompt, max_new_tokens)
        r_q = q_model.generate(prompt, max_new_tokens)
        fp_ids = r_fp["token_ids"]
        q_ids = r_q["token_ids"]
        # 以 FP32 生成的新增 token 序列对齐比较（EOS 提前结束时长度可能不同）
        new_fp = fp_ids[-r_fp["num_new_tokens"]:] if r_fp["num_new_tokens"] else []
        new_q = q_ids[-r_q["num_new_tokens"]:] if r_q["num_new_tokens"] else []
        n = max(len(new_fp), len(new_q))
        total += n
        agree += sum(1 for a, b in zip(new_fp, new_q) if a == b)
        seq_ok = new_fp == new_q
        seq_agree += int(seq_ok)
        if len(examples) < 3:
            examples.append({"prompt": prompt,
                             "fp32_output": r_fp["output"],
                             "quantized_output": r_q["output"],
                             "sequence_identical": seq_ok})
    return {
        "n_prompts": len(prompts),
        "total_new_tokens": total,
        "token_agreement": agree / total if total else 0.0,
        "sequence_agreement": seq_agree / len(prompts),
        "examples": examples,
    }


def eval_bert_accuracy(fp_model, q_model, rows: list, batch_size: int = 32) -> dict:
    texts = [t for _, t in rows]

    def run(model) -> list:
        preds = []
        for i in range(0, len(texts), batch_size):
            out = model.predict(texts[i:i + batch_size])
            preds.extend(o["label"] for o in out)
        return preds

    fp_preds = run(fp_model)
    q_preds = run(q_model)
    fp_correct = sum(1 for p, (y, _) in zip(fp_preds, rows) if p == ("正面" if y == 1 else "负面"))
    q_correct = sum(1 for p, (y, _) in zip(q_preds, rows) if p == ("正面" if y == 1 else "负面"))
    n = len(rows)
    fp_acc, q_acc = fp_correct / n, q_correct / n
    flipped = [(i, rows[i][1], fp_preds[i], q_preds[i])
               for i in range(n) if fp_preds[i] != q_preds[i]]
    return {
        "n_eval": n,
        "fp32_acc": fp_acc,
        "quantized_acc": q_acc,
        "drop_pp": (fp_acc - q_acc) * 100.0,
        "n_flipped": len(flipped),
        "flipped_examples": [
            {"id": rows[i][0] if len(rows[i]) == 3 else i, "label": y,
             "fp32_pred": f, "quant_pred": q}
            for i, y, f, q in flipped[:10]
        ],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frac-bits", type=int, default=16,
                        help="激活定点化小数位数（起步 13；实测 Q13 与权重扰动叠加时"
                             "在 ~1e-5 量级 logit 近并列处可能翻转，默认 16，范围 2^15"
                             "仍覆盖激活/logits 量级且实测一致率 1.0）")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--long", action="store_true",
                        help="使用 220 条扩测 prompt 集（P3-R1 [K] 项，data/demo/prompts_200.txt）")
    args = parser.parse_args()

    from src.common.envinfo import collect_environment
    from src.common.perf import write_report
    from src.model.loader import BertSentimentPipeline, GPT2GreedyPipeline
    from src.model.quantize import FixedPointQuantizer
    from benchmarks.finetune_bert import load_tsv

    quant_cfg = {"scheme": "GPT-2 weights Q22 fixed-point + BERT weights per-channel INT8; "
                            "activations fixed-point Q/DQ",
                 "activation_frac_bits": args.frac_bits,
                 "gpt2_weight_scheme": "q22 (linear & tied embedding/lm_head)",
                 "bert_weight_scheme": "int8 per-channel",
                 "activation_boundaries": "encoder layer outputs + final LN + lm_head/logits"}
    print(f"quantization: {quant_cfg['scheme']}, activation_frac_bits={args.frac_bits}")

    from src.model.loader import resolve_model_path
    fp_gpt2 = GPT2GreedyPipeline(resolve_model_path("gpt2"))
    q_gpt2 = GPT2GreedyPipeline(resolve_model_path("gpt2"),
                                quantizer=FixedPointQuantizer(frac_bits=args.frac_bits))
    print(f"gpt2 quantized weight tensors: {q_gpt2.quant_stats}")

    prompt_file = ("prompts_200.txt" if args.long else "prompts.txt")
    prompts = load_prompt_lines(REPO_ROOT / "data" / "demo" / prompt_file)
    t0 = time.perf_counter()
    gpt2_result = eval_gpt2_agreement(fp_gpt2, q_gpt2, prompts, args.max_new_tokens)
    print(f"gpt2: token_agreement={gpt2_result['token_agreement']:.4f} "
          f"sequence_agreement={gpt2_result['sequence_agreement']:.4f} "
          f"({gpt2_result['total_new_tokens']} tokens, {time.perf_counter() - t0:.1f}s)")

    fp_bert = BertSentimentPipeline()
    q_bert = BertSentimentPipeline(quantizer=FixedPointQuantizer(frac_bits=args.frac_bits))
    rows = load_tsv(REPO_ROOT / "data" / "sentiment" / "eval.tsv")
    t0 = time.perf_counter()
    bert_result = eval_bert_accuracy(fp_bert, q_bert, rows)
    print(f"bert: fp32_acc={bert_result['fp32_acc']:.4f} "
          f"quant_acc={bert_result['quantized_acc']:.4f} "
          f"drop={bert_result['drop_pp']:.2f}pp "
          f"flipped={bert_result['n_flipped']} ({time.perf_counter() - t0:.1f}s)")

    gpt2_pass = gpt2_result["token_agreement"] >= MIN_AGREEMENT
    bert_pass = bert_result["drop_pp"] <= MAX_DROP_PP
    report = {
        "schema": "a122-quantize-eval/1",
        "kind": "quantize_precision",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": collect_environment(probe_gpu=False),
        "config": {**quant_cfg, "max_new_tokens": args.max_new_tokens,
                   "eval_set": "data/sentiment/eval.tsv",
                   "prompt_file": prompt_file, "n_prompts": len(prompts)},
        "thresholds": {"min_token_agreement": MIN_AGREEMENT,
                       "max_drop_pp": MAX_DROP_PP},
        "gpt2": gpt2_result,
        "bert": bert_result,
        "verdict": {"gpt2_pass": gpt2_pass, "bert_pass": bert_pass,
                    "pass": gpt2_pass and bert_pass},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (RESULTS_DIR / "quantize_eval_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    write_report(report, out_path)
    try:
        print(f"report -> {out_path.relative_to(REPO_ROOT)}")
    except ValueError:
        print(f"report -> {out_path}")
    print("VERDICT:", "PASS" if report["verdict"]["pass"] else "FAIL")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    torch.manual_seed(0)
    raise SystemExit(main())
