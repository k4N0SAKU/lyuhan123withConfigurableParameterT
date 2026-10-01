"""端到端精度回归（P3-R1 A 项；P3 核心出口条件）。

三口径 × 层数曲线（docs/01 §6.3 截断口径，全部真实执行，D8）：
  1. fp32_full        ：FP32 全模型（12 层）——参考上限；
  2. fp32_truncated   ：FP32 截断明文（n_layer ∈ {1,2}，与管线同构：公开移位
                        softmax、FFN 截断 768、numpy GELU）——密文化的公平对照；
  3. q16_truncated    ：截断明文 + **激活 Q16 量化**（每个转换点
                        round(v·2¹⁶)/2¹⁶，模拟管线的定点化噪声）；
  4. pipeline_cipher  ：完整密文管线（Q16 + CKKS 噪声 + RCR 非线性；真实 CKKS
                        线性段/转换掩码），在 eval 子集上运行。

判据（P3-R1）：pipeline_cipher 与 fp32_truncated 同 n_layer 的 accuracy 下降
≤ 3 个百分点。GELU 阶数曲线：deg15 vs deg9 多项式（data/minimax 系数）在
q16_truncated 口径下的 accuracy 差。

口径声明：FP32/量化口径跑 eval 全量 200 条；密文管线跑前 20 条子集（单条
~32-60s CPU，20×2 配置约 35 分钟——评审时效约束，子集规模写入 JSON）。
稳定性：torch 线程锁定 8（P6 协议先行条款）+ seed 固定。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA = REPO_ROOT / "data" / "minimax" / "poly_approx.json"

THREADS = 8
MAX_DROP_PP = 3.0


def load_eval(path: Path, limit: int = 0):
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
        if ln.strip():
            _, lab, txt = ln.split("\t")
            rows.append((int(lab), txt))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cipher-samples", type=int, default=20,
                        help="密文管线口径的评测子集规模")
    parser.add_argument("--n-layers", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()

    torch.set_num_threads(THREADS)
    torch.manual_seed(42)
    np.random.seed(42)

    from src.model.loader import BertSentimentPipeline, LABEL_MAP
    from src.model.pipeline import ModeBPipeline, PipelineConfig
    from src.model.ops.nonlinear_approx import ref_gelu, ref_softmax_row

    plain = BertSentimentPipeline()
    m = plain.model
    L = 8
    eval_rows = load_eval(REPO_ROOT / "data" / "sentiment" / "eval.tsv")
    report = {
        "schema": "a122-pipeline-accuracy/1",
        "kind": "pipeline_accuracy",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {"seq_tokens": L, "n_layers": args.n_layers,
                   "eval_rows_total": len(eval_rows),
                   "cipher_subset": args.cipher_samples,
                   "torch_threads": THREADS,
                   "stability_protocol": "线程锁定+固定 seed（P6 协议先行条款）"},
        "thresholds": {"max_drop_pp_vs_fp32_truncated": MAX_DROP_PP},
    }

    minimax = json.loads(DATA.read_text(encoding="utf-8"))
    gelu_coeffs = {k: minimax[k]["coeffs"] for k in ("gelu_deg15", "gelu_deg9")}

    def poly_gelu(x, coeffs):
        return np.polynomial.polynomial.polyval(x, coeffs)

    def truncated_forward(texts, n_layer, quantize: bool, gelu_fn):
        """与管线同构的明文截断前向；quantize=True 时每个转换点 Q16 量化。"""
        preds = []
        for _, text in texts:
          with torch.inference_mode():
            enc = plain.tokenizer(text, truncation=True, max_length=L,
                                  return_tensors="pt")
            x = m.bert.embeddings(enc["input_ids"])[0].numpy()
            if quantize:
                x = np.round(x * (1 << 16)) / (1 << 16)
            for li in range(n_layer):
                layer = m.bert.encoder.layer[li]
                q = layer.attention.self.query(torch.tensor(x, dtype=torch.float32))
                k = layer.attention.self.key(torch.tensor(x, dtype=torch.float32))
                v = layer.attention.self.value(torch.tensor(x, dtype=torch.float32))
                ao = np.zeros((L, 768))
                qs, ks, vs = q.detach().numpy(), k.detach().numpy(), v.detach().numpy()
                for h in range(12):
                    qh, kh, vh = (qs[:, h*64:(h+1)*64], ks[:, h*64:(h+1)*64],
                                  vs[:, h*64:(h+1)*64])
                    ao[:, h*64:(h+1)*64] = ref_softmax_row(qh @ kh.T / 8.0 - 8.0) @ vh
                attn = layer.attention.output.dense(
                    torch.tensor(ao, dtype=torch.float32).unsqueeze(0))[0].numpy()
                h1v = attn + x
                if quantize:
                    h1v = np.round(h1v * (1 << 16)) / (1 << 16)
                with torch.inference_mode():
                    h1 = layer.attention.output.LayerNorm(
                        torch.tensor(h1v, dtype=torch.float32).unsqueeze(0))[0].numpy()
                inter = layer.intermediate.dense(
                    torch.tensor(h1, dtype=torch.float32)).numpy()[:, :768]
                g = gelu_fn(inter)
                if quantize:
                    g = np.round(g * (1 << 16)) / (1 << 16)
                W2 = layer.output.dense.weight.detach().numpy()[:, :768]
                ffn2 = g @ W2.T + layer.output.dense.bias.detach().numpy()
                z_raw = ffn2 + h1
                if quantize:
                    z_raw = np.round(z_raw * (1 << 16)) / (1 << 16)
                with torch.inference_mode():
                    z = layer.output.LayerNorm(
                        torch.tensor(z_raw, dtype=torch.float32).unsqueeze(0))[0].numpy()
                x = z
            with torch.inference_mode():
                cls = torch.tensor(x[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                logits = m.classifier(m.bert.pooler(cls))
            preds.append(int(logits.argmax()))
        return preds

    def accuracy(preds, rows):
        return sum(1 for p, (y, _) in zip(preds, rows)
                   if p == y) / len(rows)

    # 口径 1：FP32 全模型（200 条全量）
    t0 = time.perf_counter()
    with torch.inference_mode():
        accs = []
        for y, text in eval_rows:
            enc = plain.tokenizer(text, truncation=True, max_length=L,
                                  return_tensors="pt")
            logits = m(**enc).logits[0]
            accs.append(int(logits.argmax()))
    report["fp32_full_12layer"] = {"n": len(eval_rows),
                                   "accuracy": accuracy(accs, eval_rows),
                                   "wall_s": round(time.perf_counter() - t0, 1)}
    print(f"fp32_full: acc={report['fp32_full_12layer']['accuracy']:.4f}")

    # 口径 2/3：截断明文（FP32 / Q16）+ GELU 阶数曲线（200 条全量）
    report["fp32_truncated"] = {}
    report["q16_truncated"] = {}
    report["gelu_order_curve"] = {}
    for n_layer in args.n_layers:
        p = truncated_forward([(y, t) for y, t in eval_rows], n_layer, False,
                              lambda x: ref_gelu(x))
        report["fp32_truncated"][n_layer] = {"n": len(eval_rows),
                                             "accuracy": accuracy(p, eval_rows)}
        pq = truncated_forward([(y, t) for y, t in eval_rows], n_layer, True,
                               lambda x: ref_gelu(x))
        report["q16_truncated"][n_layer] = {"n": len(eval_rows),
                                            "accuracy": accuracy(pq, eval_rows)}
        print(f"n_layer={n_layer}: fp32={report['fp32_truncated'][n_layer]['accuracy']:.4f} "
              f"q16={report['q16_truncated'][n_layer]['accuracy']:.4f}")
    # GELU 阶数曲线（q16 口径，n_layer=1）
    for name, fn in (("deg15", lambda x: poly_gelu(x, gelu_coeffs["gelu_deg15"])),
                     ("deg9", lambda x: poly_gelu(x, gelu_coeffs["gelu_deg9"]))):
        p = truncated_forward([(y, t) for y, t in eval_rows], 1, True, fn)
        report["gelu_order_curve"][name] = {"n": len(eval_rows),
                                            "accuracy": accuracy(p, eval_rows)}
        print(f"gelu_{name}: acc={report['gelu_order_curve'][name]['accuracy']:.4f}")

    # 口径 4：密文管线（子集）
    report["pipeline_cipher"] = {}
    subset = eval_rows[:args.cipher_samples]
    for n_layer in args.n_layers:
        pipe = ModeBPipeline(plain, PipelineConfig(n_layer=n_layer, seq_tokens=L))
        preds, walls, convs, gates = [], [], [], []
        t0 = time.perf_counter()
        for i, (y, text) in enumerate(subset):
            r = pipe.classify(text)
            preds.append(int(r["label"] == "正面"))
            walls.append(r["wall_s"])
            convs.append(r["conversions"])
            gates.append(r["mpc_gates"])
            if (i + 1) % 5 == 0:
                print(f"  cipher n_layer={n_layer}: {i+1}/{len(subset)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
        acc = sum(1 for p, (y, _) in zip(preds, subset) if p == y) / len(subset)
        fp32_n = report["fp32_truncated"][n_layer]["accuracy"]
        drop_pp = (report["fp32_truncated"][n_layer]["accuracy"] - acc) * 100
        report["pipeline_cipher"][n_layer] = {
            "n": len(subset), "accuracy": acc,
            "drop_pp_vs_fp32_truncated": round(drop_pp, 2),
            "wall_s_mean": round(float(np.mean(walls)), 1),
            "conversions_mean": round(float(np.mean(convs)), 1),
            "mpc_gates_mean": int(np.mean(gates)),
            "pass": drop_pp <= MAX_DROP_PP,
        }
        print(f"cipher n_layer={n_layer}: acc={acc:.4f} drop={drop_pp:.2f}pp "
              f"{'PASS' if drop_pp <= MAX_DROP_PP else 'FAIL'}")

    all_pass = all(v["pass"] for v in report["pipeline_cipher"].values())
    report["verdict"] = {"pass": bool(all_pass), "max_drop_pp": MAX_DROP_PP}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = (RESULTS_DIR / "pipeline_accuracy_"
           f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    from src.common.perf import write_report
    write_report(report, out)
    print(f"report -> {out}")
    print("VERDICT:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
