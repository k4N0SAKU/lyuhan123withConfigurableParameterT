"""数据字典生成器（P7 陷阱 3：集中数据字典，文档数字的单一权威来源）。

从 benchmarks/results/*.json 提取文档引用的头条数字，写入 docs/data_dict.json；
每条带 source 指针（file::dot.path；max/mean 等统计量标注 derived 计算式）。
生成时自校验：按 source 回读源文件比对数值——字典与 JSON 不一致即报错。

用法：python scripts/gen_data_dict.py [--check]
  --check：只校验已生成字典与源 JSON 一致（不写文件）——供
  scripts/check_docs_consistency.py 调用。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
OUT = REPO / "docs" / "data_dict.json"

E: dict = {}   # name -> {value, unit, source}


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _put(key: str, value, unit: str, source: str) -> None:
    E[key] = {"value": value, "unit": unit, "source": source}


def _resolve(source: str):
    """解析 file::path 指针（path 段含 '/' 时为 dict key）；返回 value。"""
    f, path = source.split("::", 1)
    cur = json.loads((REPO / f).read_text(encoding="utf-8"))
    for seg in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(seg)]
        else:
            cur = cur[seg]
    return cur


def build() -> dict:
    p6 = _load("p6_full_bench.json")
    # ---- 代码规模 ----
    t = p6["code_size"]["total"]
    _put("code_size.code_lines", t["code"], "lines", "benchmarks/results/p6_full_bench.json::code_size.total.code")
    _put("code_size.files", t["files"], "files", "benchmarks/results/p6_full_bench.json::code_size.total.files")
    _put("code_size.comment_lines", t["comment"], "lines", "benchmarks/results/p6_full_bench.json::code_size.total.comment")
    # ---- 密文矩阵 ----
    for k in (4, 8, 12):
        c = p6["cipher_matrix"][f"modeB_layer{k}"]
        base = f"benchmarks/results/p6_full_bench.json::cipher_matrix.modeB_layer{k}"
        _put(f"matrix.layer{k}.rounds", c["rounds_done"], "rounds", f"{base}.rounds_done")
        _put(f"matrix.layer{k}.wall_p50_ms", round(c["wall_ms"]["p50"]), "ms", f"{base}.wall_ms.p50")
        _put(f"matrix.layer{k}.wall_p95_ms", round(c["wall_ms"]["p95"]), "ms", f"{base}.wall_ms.p95")
        _put(f"matrix.layer{k}.wall_max_ms", round(max(r["wall_ms"] for r in c["records"])), "ms",
             f"{base}.records[max(wall_ms)]")
        _put(f"matrix.layer{k}.mem_p50_mib", round(c["peak_delta_bytes"]["p50"] / 2**20, 1), "MiB",
             f"{base}.peak_delta_bytes.p50/2^20")
        _put(f"matrix.layer{k}.mem_max_mib", round(max(r["peak_delta_bytes"] for r in c["records"]) / 2**20, 1), "MiB",
             f"{base}.records[max(peak_delta_bytes)]/2^20")
        _put(f"matrix.layer{k}.conversions_per_round", c["conversions_per_round"], "count", f"{base}.conversions_per_round")
    # ---- 明文基线（P6 20 轮正口径） ----
    for wl, jkey in (("bert", "bert_latency_full12"), ("gpt2", "gpt2_latency")):
        rt = p6["plaintext"][jkey]["round_total"]
        _put(f"plaintext.{wl}.p50_ms", round(rt["p50"], 1), "ms",
             f"benchmarks/results/p6_full_bench.json::plaintext.{jkey}.round_total.p50")
        _put(f"plaintext.{wl}.p95_ms", round(rt["p95"], 1), "ms",
             f"benchmarks/results/p6_full_bench.json::plaintext.{jkey}.round_total.p95")
    acc = p6["plaintext"]["bert_accuracy_full12_quantized"]
    _put("plaintext.bert_acc_full12_quantized", round(acc["accuracy"] * 100, 2), "%",
         "benchmarks/results/p6_full_bench.json::plaintext.bert_accuracy_full12_quantized.accuracy*100")
    # ---- P0 权威基线（跨阶段对比锚点） ----
    import glob as _g
    bf = sorted(_g.glob(str(RESULTS / "bert_plaintext_baseline_*.json")))[-1]
    gf = sorted(_g.glob(str(RESULTS / "gpt2_plaintext_baseline_*.json")))[-1]
    for name, f in (("bert_authoritative", Path(bf)), ("gpt2_authoritative", Path(gf))):
        d = json.loads(f.read_text(encoding="utf-8"))
        rel = str(Path(f).relative_to(REPO)).replace("\\", "/")
        _put(f"baseline.{name}.p50_ms", round(d["summary"]["round_total"]["p50"], 1), "ms",
             f"{rel}::summary.round_total.p50")
        _put(f"baseline.{name}.p95_ms", round(d["summary"]["round_total"]["p95"], 1), "ms",
             f"{rel}::summary.round_total.p95")
    # ---- 定点化（P0） ----
    qf = sorted(_g.glob(str(RESULTS / "quantize_eval_*.json")))[-1]
    q = json.loads(Path(qf).read_text(encoding="utf-8"))
    qrel = str(Path(qf).relative_to(REPO)).replace("\\", "/")
    _put("quantize.gpt2_token_agreement", round(q["gpt2"]["token_agreement"], 4), "ratio",
         f"{qrel}::gpt2.token_agreement")
    _put("quantize.bert_drop_pp", round(q["bert"]["drop_pp"], 2), "pp",
         f"{qrel}::bert.drop_pp")
    _put("quantize.bert_fp32_acc", round(q["bert"]["fp32_acc"] * 100, 2), "%", f"{qrel}::bert.fp32_acc*100")
    _put("quantize.bert_quantized_acc", round(q["bert"]["quantized_acc"] * 100, 2), "%", f"{qrel}::bert.quantized_acc*100")
    # ---- 管线精度（P3 四口径） ----
    af = sorted(_g.glob(str(RESULTS / "pipeline_accuracy_*.json")))[-1]
    a = json.loads(Path(af).read_text(encoding="utf-8"))
    arel = str(Path(af).relative_to(REPO)).replace("\\", "/")
    for key in ("fp32_full_12layer", "pipeline_cipher"):
        node = a[key]
        if "accuracy" in node:
            _put(f"accuracy.{key}", round(node["accuracy"] * 100, 2), "%", f"{arel}::{key}.accuracy*100")
        elif "label" in node:
            _put(f"accuracy.{key}_present", True, "bool", f"{arel}::{key}")
    # ---- C2 创新点 ----
    for name, c in p6["c2_selector"].items():
        if isinstance(c, dict) and "token_agreement_vs_fp32" in c:
            base = f"benchmarks/results/p6_full_bench.json::c2_selector.{name}"
            _put(f"c2.{name}.agreement", round(c["token_agreement_vs_fp32"], 4), "ratio", f"{base}.token_agreement_vs_fp32")
            _put(f"c2.{name}.size_mib", round(c["size_bytes"] / 2**20, 1), "MiB", f"{base}.size_bytes/2^20")
            _put(f"c2.{name}.p50_ms", round(c["latency"]["p50"]), "ms", f"{base}.latency.p50")
    _put("c2.fp32_ref_size_mib", round(p6["c2_selector"]["fp32_reference_size_bytes"] / 2**20, 1), "MiB",
         "benchmarks/results/p6_full_bench.json::c2_selector.fp32_reference_size_bytes/2^20")
    # ---- C3 创新点 ----
    for dim in ("d256", "d512"):
        cell = p6["c3_ab"][dim]
        for impl in ("diagonal_bsgs", "rowsum"):
            _put(f"c3.{dim}.{impl}.time_s", cell[impl]["time_s"], "s",
                 f"benchmarks/results/p6_full_bench.json::c3_ab.{dim}.{impl}.time_s")
            _put(f"c3.{dim}.{impl}.rotations", cell[impl]["rotations"], "count",
                 f"benchmarks/results/p6_full_bench.json::c3_ab.{dim}.{impl}.rotations")
        ratio = round(cell["rowsum"]["time_s"] / cell["diagonal_bsgs"]["time_s"], 2)
        _put(f"c3.{dim}.ratio_rowsum_over_diag", ratio, "x",
             f"benchmarks/results/p6_full_bench.json::c3_ab.{dim}.*(rowsum.time_s/diagonal.time_s)")
    # ---- 流量与节点内存 ----
    for link, m in p6["traffic"]["per_connection_bidirectional"].items():
        safe = link.replace("/", "_")
        _put(f"traffic.{safe}.bytes_sent", m["bytes_sent"], "bytes",
             f"benchmarks/results/p6_full_bench.json::traffic.per_connection_bidirectional.{link}.bytes_sent")
    for nid, m in p6["node_memory"]["per_node_peak_rss"].items():
        _put(f"node_memory.{nid}.peak_rss_mib", round(m["peak_rss_bytes"] / 2**20, 1), "MiB",
             f"benchmarks/results/p6_full_bench.json::node_memory.per_node_peak_rss.{nid}.peak_rss_bytes/2^20")
    # ---- 攻击判定 ----
    av = _load("attack_verdicts.json")
    s_ = av["summary"]
    for k in ("defended", "demonstrated_boundary", "failed"):
        _put(f"attacks.{k}", s_[k], "count", "benchmarks/results/attack_verdicts.json::summary." + k)
    _put("attacks.total_cases", sum(s_.values()), "count",
         "benchmarks/results/attack_verdicts.json::summary(sum)")
    # ---- P4 协议基准 ----
    p4 = _load("p4_protocol_bench.json")
    _put("p4.establish_p50_ms", round(p4["establish"]["establish_total"]["p50"]), "ms",
         "benchmarks/results/p4_protocol_bench.json::establish.establish_total.p50")
    _put("p4.roundtrip_64k_p50_ms", round(p4["channel_roundtrip"]["payload_65536B"]["p50"]), "ms",
         "benchmarks/results/p4_protocol_bench.json::channel_roundtrip.payload_65536B.p50")
    _put("p4.modeb_entry_p50_ms", round(p4["conversion_mode_b_full_slot"]["entry_make_masked_ct"]["p50"]), "ms",
         "benchmarks/results/p4_protocol_bench.json::conversion_mode_b_full_slot.entry_make_masked_ct.p50")
    # ---- estimator ----
    se = _load("security_estimator.json")
    cdec = se["C_decision_2p16_1720bit"]
    _put("estimator.C_2p16_best_rop_log2", cdec["best_rop_log10_2"], "log2",
         "benchmarks/results/security_estimator.json::C_decision_2p16_1720bit.best_rop_log10_2")
    _put("estimator.C_meets_128bit", cdec["meets_128bit_classical"], "bool",
         "benchmarks/results/security_estimator.json::C_decision_2p16_1720bit.meets_128bit_classical")
    # ---- 开销倍数（派生，带两口径声明） ----
    m12 = p6["cipher_matrix"]["modeB_layer12"]["wall_ms"]["p50"]
    new_anchor = p6["plaintext"]["bert_latency_full12"]["round_total"]["p50"]
    auth_anchor = json.loads(Path(bf).read_text(encoding="utf-8"))["summary"]["round_total"]["p50"]
    _put("overhead.12L_vs_plaintext_p6anchor", round(m12 / new_anchor), "x",
         f"derived: p6 matrix.layer12.wall_p50 / plaintext.bert.p50({new_anchor}ms)")
    _put("overhead.12L_vs_plaintext_authoritative", round(m12 / auth_anchor), "x",
         f"derived: p6 matrix.layer12.wall_p50 / P0 权威 baseline({auth_anchor}ms)")
    return {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "数据字典——00~06 文档头条数字的单一权威来源（P7 陷阱 3）；"
                    "source=file::dot.path（含 derived 的计算式）；"
                    "校验：python scripts/check_docs_consistency.py",
            "entries": E}


def verify(data: dict) -> list:
    """按 source 回读源文件核对（file 指针为简单路径的条目）；返回不一致列表。"""
    bad = []
    for k, e in data["entries"].items():
        src = e["source"]
        # 派生/聚合/缩放指针不回读；数值条目按存储精度取容差
        if (src.startswith("derived:") or "/2^" in src or "*" in src
                or "[max(" in src or "sum" in src.split("::")[-1]):
            continue
        try:
            got = _resolve(src)
        except Exception as exc:
            bad.append((k, src, f"resolve error {exc}"))
            continue
        if isinstance(e["value"], bool) or isinstance(got, bool):
            if got != e["value"]:
                bad.append((k, src, f"{e['value']} != {got}"))
        elif isinstance(e["value"], (int, float)) and isinstance(got, (int, float)):
            # 存储值为 round(got, d)——容差取半单位（整数 0.5，一位小数 0.05 等）
            dec = len(str(e["value"]).split(".")[1]) if "." in str(e["value"]) else 0
            tol = 0.5 * (10 ** -dec) + 1e-9
            if abs(float(got) - float(e["value"])) > tol:
                bad.append((k, src, f"{e['value']} != {got}"))
        elif got != e["value"]:
            bad.append((k, src, f"{e['value']} != {got}"))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args()
    data = build()
    bad = verify(data)
    if bad:
        print("字典与源 JSON 不一致：")
        for k, src, msg in bad:
            print(f"  [{k}] {src} -> {msg}")
        return 1
    if args.check:
        on_disk = json.loads(OUT.read_text(encoding="utf-8"))
        # 逐条比对（忽略 generated_utc）
        diff = [k for k in data["entries"]
                if k not in on_disk.get("entries", {})
                or on_disk["entries"][k] != data["entries"][k]]
        missing = [k for k in on_disk.get("entries", {}) if k not in data["entries"]]
        if diff or missing:
            print(f"字典过期：新增/变更 {diff[:5]}，多出 {missing[:5]}——请重跑 gen_data_dict.py")
            return 1
        print(f"data_dict OK（{len(data['entries'])} 条，与源 JSON 一致）")
        return 0
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written -> docs/data_dict.json（{len(E)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
