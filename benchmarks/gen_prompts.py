"""评测 prompt 程序化生成器（P3-R1 [K] 项：量化评测扩至 ≥200 prompts）。

生成方式（透明可复现，D8）：主题名词短语 × 谓词模板 的笛卡尔组合并去重，
覆盖日常/科技/自然/叙事四类语域；输出一行一 prompt 到
data/demo/prompts_200.txt。生成器入库，任何人可重跑验证。

用法：python -m benchmarks.gen_prompts [--count 220] [--out data/demo/prompts_200.txt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from itertools import product

REPO_ROOT = Path(__file__).resolve().parents[1]

SUBJECTS = [
    "The old librarian", "A young engineer", "The morning train", "Our small team",
    "The village baker", "A curious child", "The night shift", "My neighbour's dog",
    "The city council", "A traveling musician", "The river ferry", "An elderly gardener",
    "The museum guide", "A local choir", "The mountain trail", "Our new database",
    "The winter market", "A retired teacher", "The harbor lights", "A student orchestra",
]
VERBS = [
    "carefully documented", "quietly celebrated", "promptly repaired",
    "patiently explained", "slowly transformed", "proudly presented",
    "gently restored", "reliably delivered", "honestly reviewed",
    "brightly decorated", "steadily improved", "warmly welcomed",
]
OBJECTS = [
    "every fragile manuscript", "the annual report", "a rusty bicycle",
    "three lost letters", "the community garden", "a weathered lighthouse",
    "the rehearsal schedule", "several cracked tiles", "the ferry timetable",
    "a broken clock tower", "the winter harvest", "an old harmonica",
    "the weekly newsletter", "a small blue kite", "the evening classes",
    "two wooden benches",
]
TAILS = [
    "before the rain arrived", "during the quiet afternoon",
    "after the long meeting", "near the old bridge",
    "at the edge of town", "without any complaint",
    "under the warm lamplight", "behind the closed doors",
    "along the coastal road", "in the early fog",
]


def generate(count: int) -> list:
    seen, out = set(), []
    for subj, verb, obj, tail in product(SUBJECTS, VERBS, OBJECTS, TAILS):
        s = f"{subj} {verb} {obj} {tail}."
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= count:
            return out
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=220)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "demo" / "prompts_200.txt"))
    args = parser.parse_args()
    prompts = generate(args.count)
    Path(args.out).write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"generated {len(prompts)} unique prompts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
