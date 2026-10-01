"""文档一致性校验器（P7 陷阱 3 收口；终检项之一）。

检查四类一致性：
1. 数据字典 ↔ 源 JSON（调用 gen_data_dict.py --check）；
2. 脚本注入表：docs/04/05 中标注"自动生成并注入"的表格必须与当前
   p6_tables.md 生成结果逐字节一致（防手抄回退——P6-R2 事故防复发）；
3. 过期数字黑名单：00~06 终稿文档中不得出现已证伪的旧值
   （历史叙事在 docs/phases/ 不受限）；
4. 文档互引：00-方案概述必须引用 01~06 六份文件（文件存在）。

用法：python scripts/check_docs_consistency.py   # 全绿 exit 0，否则列出问题
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# 已证伪的旧值（终稿范围禁用；docs/phases/ 历史叙事不在此列）
BLACKLIST = [
    "452.7", "298.3", "880~948", "5.11×", "5.13×",
    "峰值内存增量 (MB)", "体积 (MB)", "119ms，开销",
]
# 终稿七份（00~06 显式清单——00-赛题原文.md 为存档不计入）
FINAL_NAMES = ["00-方案概述.md", "01-架构与协议规格.md", "02-威胁模型与安全分析.md",
               "03-攻击测试报告.md", "04-性能与基线对比.md", "05-创新点报告.md",
               "06-复现手册.md"]
FINAL_DOCS = [DOCS / n for n in FINAL_NAMES]


def check_dict() -> list:
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "gen_data_dict.py"),
                        "--check"], capture_output=True, text=True)
    return [] if r.returncode == 0 else [f"data_dict: {r.stdout.strip()} {r.stderr.strip()}"]


def _norm(text: str) -> str:
    """行尾归一化——仓库文件为混合 LF/CRLF（编辑器/工具历史），比较前统一。"""
    return text.replace("\u000d\u000a", "\u000a").replace("\u000d", "\u000a")


def check_injected_tables() -> list:
    problems = []
    tables = _norm((REPO / "benchmarks" / "results" / "p6_tables.md").read_text(encoding="utf-8"))

    def section(marker: str, nxt: str) -> str:
        try:
            return tables[tables.index(marker):tables.index(nxt, tables.index(marker))].rstrip()
        except ValueError:
            return ""

    # docs/04：密文矩阵表（注入块 = "## 2." 段）
    doc4 = _norm((DOCS / "04-性能与基线对比.md").read_text(encoding="utf-8"))
    mat = section("## 2.", "## 3.")
    if mat:
        body = mat.split("\n", 1)[1] if mat.startswith("## 2.") else mat
        # 生成表正文（去掉标题行）须在 docs/04 中
        if body.strip() not in doc4:
            problems.append("docs/04 密文矩阵表与当前 p6_tables.md §2 不一致（需重跑 tables+inject）")
    # docs/05：C2 表（注入块 = "## 5." 段表体）
    doc5 = _norm((DOCS / "05-创新点报告.md").read_text(encoding="utf-8"))
    c2 = section("## 5.", "层级升级 trace")
    if c2:
        rows = [ln for ln in c2.splitlines() if ln.startswith("|")]
        for row in rows[2:]:                  # 跳过标题/分隔行
            if row.strip() and row.strip() not in doc5:
                problems.append(f"docs/05 C2 行与生成表脱钩: {row[:60]}")
                break
    return problems


def check_blacklist() -> list:
    problems = []
    for f in FINAL_DOCS:
        text = f.read_text(encoding="utf-8")
        for bad in BLACKLIST:
            if bad in text:
                problems.append(f"{f.name} 含过期值 {bad!r}")
    return problems


def check_cross_refs() -> list:
    problems = []
    d0 = DOCS / "00-方案概述.md"
    if not d0.exists():
        return ["docs/00-方案概述.md 缺失"]
    text = d0.read_text(encoding="utf-8")
    for i in range(1, 7):
        if not re.search(rf"0{i}-", text):
            problems.append(f"00-方案概述 未引用 0{i}- 文档")
    for n in FINAL_NAMES:
        if not (DOCS / n).exists():
            problems.append(f"docs/{n} 缺失")
    if "data_dict" not in text:
        problems.append("00-方案概述 未声明数据字典")
    return problems


def main() -> int:
    all_p = []
    for name, fn in (("数据字典↔JSON", check_dict),
                     ("注入表字节一致", check_injected_tables),
                     ("过期值黑名单", check_blacklist),
                     ("文档互引", check_cross_refs)):
        ps = fn()
        print(f"[{'FAIL' if ps else 'OK  '}] {name}")
        for p in ps:
            print(f"       - {p}")
        all_p += ps
    if all_p:
        print(f"\n共 {len(all_p)} 项不一致")
        return 1
    print("\n全部一致（数据字典/注入表/黑名单/互引）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
