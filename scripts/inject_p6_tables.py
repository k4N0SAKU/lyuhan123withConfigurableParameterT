"""把 p6_tables.md 的密文矩阵/C2 表注入 docs/04 与 docs/05（P6：禁止手抄数字）。"""
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
tables = (repo / "benchmarks/results/p6_tables.md").read_text(encoding="utf-8")


def _section(marker: str, next_marker: str) -> str:
    start = tables.index(marker)
    end = tables.index(next_marker, start)
    return tables[start:end].rstrip()


# ---- docs/04：密文矩阵表 ----
doc4 = repo / "docs/04-性能与基线对比.md"
s = doc4.read_text(encoding="utf-8")
ph4 = "<!-- P6-TABLES:cipher_matrix（由 run_full_bench.py --parts tables 生成后粘贴） -->"
if ph4 in s:
    note4 = ("\n\n（上表由 run_full_bench.py --parts tables 自动生成并注入"
             "——禁止手抄数字，D8）")
    s = s.replace(ph4, _section("## 2.", "## 3.") + note4)
    doc4.write_text(s, encoding="utf-8")
    print("docs/04 injected")
else:
    print("docs/04: placeholder gone (already injected)" if "自动生成并注入" in s
          else "docs/04: placeholder missing")

# ---- docs/05：C2 A/B 表（P6-R2：时延列曾手抄脱钩——纳入脚本注入） ----
doc5 = repo / "docs/05-创新点报告.md"
s5 = doc5.read_text(encoding="utf-8")
ph5 = "<!-- P6-TABLES:c2_selector -->"
if ph5 not in s5:
    # 首次注入：把 §1.3 手抄表替换为占位符
    start = s5.index("| 配置 | token 一致率 | 体积")
    end = s5.index("### 1.4", start)
    s5 = s5[:start] + ph5 + "\n\n" + s5[end:]
sec5 = _section("## 5.", "层级升级 trace")
# 去掉生成表标题行，保留表体
body = "\n".join(sec5.splitlines()[1:]).rstrip()
s5 = s5.replace(ph5, body + "\n\n（上表由 run_full_bench.py --parts tables 自动生成并注入"
                          "——禁止手抄数字，D8；层级升级 trace 与 FP32 参照体积见"
                          " p6_tables.md §5）")
doc5.write_text(s5, encoding="utf-8")
print("docs/05 C2 table injected")
