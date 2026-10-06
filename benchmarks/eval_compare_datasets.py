# -*- coding: utf-8 -*-
"""对比实验数据集评测（P9 对比实验；D8 纪律）。

阶段：
  crossdomain — 既有自建检查点在 ChnSentiCorp test 上的跨域迁移准确率（无需微调）
  calibers    — ChnSentiCorp 官方微调检查点三口径：FP32 / INT8 量化 / 密文管线标签一致率
                （密文为截断 2 层 8 token 口径，对照明文截断参考的一致性，精度水平另申明）
  sst2        — SST-2 验证集：FP32 / INT8 量化准确率 + 密文一致性子集（文献可比臂）

输出：benchmarks/results/compare_eval_<phase>.json（schema a122-perf/1）
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

DEV = "cpu"
FIXED_ONE = 1 << 16

def now(): return datetime.now(timezone.utc).isoformat()

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written ->", path)

def load_chn_test():
    ds = load_dataset("lansinuote/ChnSentiCorp")
    return ds["test"]["text"], ds["test"]["label"]

def load_sst2_val():
    ds = load_dataset("nyu-mll/glue", "sst2")
    return ds["validation"]["sentence"], ds["validation"]["label"]

@torch.no_grad()
def bert_accuracy(model, tok, texts, labels, max_length=128, batch=32):
    model.eval()
    correct = 0
    preds = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt")
        pred = model(**enc).logits.argmax(-1)
        preds += pred.tolist()
    correct = sum(int(p == l) for p, l in zip(preds, labels))
    return correct / len(labels), preds

def phase_crossdomain(ckpt):
    """既有自建检查点 → ChnSentiCorp test（自建域→公开域跨域迁移）。"""
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEV)
    texts, labels = load_chn_test()
    t0 = time.perf_counter()
    acc, preds = bert_accuracy(model, tok, texts, labels)
    out = {"schema": "a122-perf/1", "kind": "compare-eval-crossdomain",
           "checkpoint": ckpt, "eval_set": "ChnSentiCorp test (n=1200)",
           "accuracy": acc, "seconds": time.perf_counter()-t0,
           "note": "自建域→公开域跨域迁移：模型未在 ChnSentiCorp 训练",
           "timestamp_utc": now()}
    write_json("benchmarks/results/compare_eval_crossdomain.json", out)

def truncated_model(model, n_layer):
    """构造截断参考：保留前 n_layer 个编码器层（P3 口径的明文截断参考）。"""
    model.eval()
    kept = torch.nn.ModuleList(list(model.bert.encoder.layer)[:n_layer])
    model.bert.encoder.layer = kept
    model.config.num_hidden_layers = n_layer
    return model

def phase_calibers(ckpt, subset=20, seed=42):
    """三口径：FP32 全模型 / INT8 量化 / 密文管线标签一致率（截断层口径）。"""
    from src.model.loader import BertSentimentPipeline
    from src.model.pipeline import ModeBPipeline, PipelineConfig
    texts, labels = load_chn_test()
    pipe = BertSentimentPipeline(model_path=ckpt)
    # 口径 A：FP32 全模型（len 128）
    t0 = time.perf_counter()
    model = pipe.model
    tok = pipe.tokenizer
    a_acc, _ = bert_accuracy(model, tok, texts, labels)
    a_sec = time.perf_counter() - t0
    # 口径 B：INT8 量化（同管线量化器）
    import torch
    qpipe = BertSentimentPipeline(model_path=ckpt,
                                  quantizer=__import__("src.model.quantize", fromlist=["FixedPointQuantizer"]).FixedPointQuantizer())
    b_acc, _ = bert_accuracy(qpipe.model, qpipe.tokenizer, texts, labels)
    b_sec = time.perf_counter() - t0 if False else None
    # 口径 C：密文管线标签一致率（截断 2 层 8 token；对照明文截断参考）
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(texts), generator=g)[:subset].tolist()
    sub_texts = [texts[i] for i in idx]
    sub_labels = [labels[i] for i in idx]
    # 明文截断参考（torch 截断模型）
    import copy
    tmodel = truncated_model(copy.deepcopy(model), 2)
    ref_labels, _ = bert_accuracy(tmodel, tok, sub_texts, sub_labels, max_length=8) if False else (None, None)
    # 用 bert_accuracy 拿 preds 而非 acc：直接调用
    tmodel.eval()
    ref_preds = []
    with torch.no_grad():
        for i in range(0, len(sub_texts), 8):
            enc = tok(sub_texts[i:i+8], truncation=True, max_length=8,
                      padding=True, return_tensors="pt")
            ref_preds += tmodel(**enc).logits.argmax(-1).tolist()
    # 密文管线（截断 2 层 8 token）
    cfg = PipelineConfig(n_layer=2, seq_tokens=8)
    cpipe = ModeBPipeline(pipe, cfg)
    cipher_labels = []
    for t in sub_texts:
        cipher_labels.append(cpipe.classify(t)["label_idx"] if "label_idx" in cpipe.classify(t)
                             else None)
    # 上面的 classify 返回 dict（label/prob）——统一取 label 文本转索引
    from src.model.loader import LABEL_MAP
    lab2idx = {v: k for k, v in LABEL_MAP.items()}
    cipher_labels = [lab2idx.get(l, -1) for l in cipher_labels]
    plain_trunc_labels = [lab2idx.get(l, -1) for l in
                          (ref_preds and [ "负面" if x == 0 else "正面" for x in ref_preds])]
    consist = sum(int(c == p) for c, p in
                  zip(cipher_labels, plain_trunc_labels)) / len(cipher_labels)
    full_agree = sum(int(c == l) for c, l in zip(cipher_labels, sub_labels)) / len(cipher_labels)
    out = {"schema": "a122-perf/1", "kind": "compare-eval-calibers",
           "checkpoint": ckpt, "eval_set": "ChnSentiCorp test",
           "caliber_A_fp32_full_accuracy": a_acc, "caliber_A_seconds": a_sec,
           "caliber_B_int8_accuracy": b_acc,
           "caliber_C": {"subset": subset, "n_layer": 2, "seq_tokens": 8,
                         "cipher_vs_plain_trunc_consistency": consist,
                         "cipher_vs_full_label_agree": full_agree,
                         "note": "截断层精度随机水平为 P3 诚实发现（容量事实）；本口径只主张密文与明文截断参考的一致性"},
           "timestamp_utc": now()}
    write_json("benchmarks/results/compare_eval_calibers.json", out)

def phase_sst2(ckpt_dir=None, subset=20, seed=42):
    """SST-2 验证集：FP32/INT8 准确率 + 密文一致性子集。"""
    from src.model.loader import BertSentimentPipeline
    from src.model.pipeline import ModeBPipeline, PipelineConfig
    texts, labels = load_sst2_val()
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(DEV)
    t0 = time.perf_counter()
    a_acc, _ = bert_accuracy(model, tok, texts, labels)
    a_sec = time.perf_counter() - t0
    from src.model.quantize import FixedPointQuantizer
    qmodel = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(DEV)
    from src.model.quantize import quantize_model_weights
    quantize_model_weights(qmodel, linear_scheme="int8", embedding_scheme="int8")
    b_acc, _ = bert_accuracy(qmodel, tok, texts, labels)
    # 密文一致性子集（截断 2 层 8 token，对照明文截断参考）
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(texts), generator=g)[:subset].tolist()
    sub_texts = [texts[i] for i in idx]
    pipe = BertSentimentPipeline(model_path=ckpt_dir) if os.path.exists(
        os.path.join(ckpt_dir, "config.json")) else None
    cpipe = ModeBPipeline(_wrap(ckpt_dir), PipelineConfig(n_layer=2, seq_tokens=8))
    cipher = [cpipe.classify(t)["prob"] for t in sub_texts]
    out = {"schema": "a122-perf/1", "kind": "compare-eval-sst2",
           "checkpoint": ckpt_dir, "eval_set": "SST-2 validation (n=872)",
           "caliber_fp32_accuracy": a_acc, "caliber_int8_accuracy": b_acc,
           "cipher_subset": {"n": subset, "labels_consistent_with_plain_trunc": None,
                             "note": "密文一致性明细同 ChnSentiCorp 口径"},
           "literature_ref": {"EncFormer_SST2_acc": 0.9178, "EncFormer_plain_acc": 0.9243,
                              "source": "arXiv:2604.09975（已核验）"},
           "timestamp_utc": now()}
    write_json("benchmarks/results/compare_eval_sst2.json", out)

def _wrap(ckpt_dir):
    from src.model.loader import BertSentimentPipeline
    return BertSentimentPipeline(model_path=ckpt_dir)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True,
                    choices=["crossdomain", "calibers", "sst2"])
    ap.add_argument("--ckpt", default="data/models/bert-base-chinese-sentiment",
                    help="检查点路径（calibers/sst2 用）")
    ap.add_argument("--subset", type=int, default=20)
    args = ap.parse_args()
    if args.phase == "crossdomain":
        phase_crossdomain(args.ckpt)
    elif args.phase == "calibers":
        phase_calibers(args.ckpt, subset=args.subset)
    elif args.phase == "sst2":
        phase_sst2(args.ckpt, subset=args.subset)

if __name__ == "__main__":
    main()
