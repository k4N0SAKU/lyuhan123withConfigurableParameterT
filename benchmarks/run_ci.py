"""本地全量跑脚本（CI 式质量门）：环境自检 → 全量测试 → 明文基线 → 定点化精度断言。

用法（仓库根目录）：
    python -m benchmarks.run_ci                # 全流程
    python -m benchmarks.run_ci --skip-tests   # 跳过 pytest（快速冒烟）

任一步骤失败即以非零码退出。定点化阈值（分类下降 ≤1.5pp、生成一致率 ≥95%）
由 eval_quantize 内部断言并落盘 JSON，本脚本复核 JSON 中的 verdict 字段。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
PY = sys.executable


def run_step(title: str, cmd: list, timeout_s: int = 1200) -> None:
    print(f"\n===== [{title}] {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, timeout=timeout_s)
    if proc.returncode != 0:
        print(f"STEP FAILED: {title} (exit={proc.returncode})")
        sys.exit(proc.returncode)


def latest_report(kind_prefix: str) -> Path:
    candidates = sorted(RESULTS_DIR.glob(f"{kind_prefix}*.json"))
    if not candidates:
        print(f"ERROR: no report matching {kind_prefix}* in {RESULTS_DIR}")
        sys.exit(1)
    return candidates[-1]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    print("== A1-22 本地全量跑（质量门）==")

    # 1) 环境自检（含 CKKS / SM3 冒烟）
    run_step("环境自检", [PY, "-m", "src.common.envinfo"], timeout_s=300)

    # 2) pytest 全量（覆盖 ini 中默认排除的 slow 用例）
    if not args.skip_tests:
        run_step("pytest 全量", [PY, "-m", "pytest", "-o", "addopts=",
                                 "-q", "tests"], timeout_s=1800)

    # 3) 明文基线（GPT-2 生成 + BERT 分类）
    run_step("明文基线 GPT-2", [PY, "-m", "benchmarks.perf_runner",
                                "--workload", "gpt2", "--rounds", str(args.rounds)])
    run_step("明文基线 BERT", [PY, "-m", "benchmarks.perf_runner",
                               "--workload", "bert", "--rounds", str(args.rounds)])

    # 4) 定点化精度（内部已含阈值断言）
    run_step("定点化精度评估", [PY, "-m", "benchmarks.eval_quantize"])

    # 5) 复核报告一致性与阈值 verdict
    checks = []
    for prefix, key_rounds in (("gpt2_plaintext", args.rounds),
                               ("bert_plaintext", args.rounds)):
        path = latest_report(prefix)
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["rounds_completed"] == key_rounds, (path, "rounds 不一致")
        p50 = report["summary"]["round_total"]["p50"]
        p95 = report["summary"]["round_total"]["p95"]
        checks.append((path.name, f"rounds={report['rounds_completed']} "
                                  f"p50={p50:.1f}ms p95={p95:.1f}ms"))
    q_path = latest_report("quantize_eval_")
    q_report = json.loads(q_path.read_text(encoding="utf-8"))
    verdict = q_report["verdict"]
    checks.append((q_path.name,
                   f"gpt2_token_agreement={q_report['gpt2']['token_agreement']:.4f} "
                   f"bert_drop_pp={q_report['bert']['drop_pp']:.2f} "
                   f"verdict={'PASS' if verdict['pass'] else 'FAIL'}"))
    if not verdict["pass"]:
        print("QUANTIZE VERDICT FAIL")
        return 1

    print("\n===== 汇总 =====")
    for name, line in checks:
        print(f"  {name}: {line}")
    print("\nRUN_CI OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
