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
    import copy
    import torch
    from src.model.loader import BertSentimentPipeline, LABEL_MAP
    from src.model.pipeline import ModeBPipeline, PipelineConfig
    from src.model.quantize import FixedPointQuantizer
    texts, labels = load_chn_test()
    lab2idx = {v: k for k, v in LABEL_MAP.items()}

    # 口径 A：FP32 全模型（len 128）
    t0 = time.perf_counter()
    pipe = BertSentimentPipeline(model_path=ckpt)
    tok = pipe.tokenizer
    a_acc, _ = bert_accuracy(pipe.model, pipe.tokenizer, texts, labels)
    a_sec = time.perf_counter() - t0

    # 口径 B：INT8 权重量化（同管线量化器）
    qpipe = BertSentimentPipeline(model_path=ckpt,
                                  quantizer=FixedPointQuantizer())
    b_acc, _ = bert_accuracy(qpipe.model, qpipe.tokenizer, texts, labels)
    del qpipe
    import gc; gc.collect()

    # 口径 C：密文管线标签一致率（截断 2 层 8 token；对照明文截断参考）
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(texts), generator=g)[:subset].tolist()
    sub_texts = [texts[i] for i in idx]
    sub_labels = [labels[i] for i in idx]
    tmodel = truncated_model(copy.deepcopy(pipe.model), 2)
    ref_preds = []
    with torch.no_grad():
        for i in range(0, len(sub_texts), 8):
            enc = tok(sub_texts[i:i+8], truncation=True, max_length=8,
                      padding=True, return_tensors="pt")
            ref_preds += tmodel(**enc).logits.argmax(-1).tolist()
    ref_labels = ["负面" if x == 0 else "正面" for x in ref_preds]
    import gc as _gc; _gc.collect()
    cpipe = ModeBPipeline(pipe, PipelineConfig(n_layer=2, seq_tokens=8))
    cipher_labels = [lab2idx.get(cpipe.classify(t)["label"], -1) for t in sub_texts]
    plain_labels = [lab2idx.get(l, -1) for l in ref_labels]
    consist = sum(int(c == pl) for c, pl in
                  zip(cipher_labels, plain_labels)) / len(cipher_labels)
    full_agree = sum(int(c == l) for c, l in
                     zip(cipher_labels, sub_labels)) / len(cipher_labels)
    out = {"schema": "a122-perf/1", "kind": "compare-eval-calibers",
           "checkpoint": ckpt, "eval_set": "ChnSentiCorp test",
           "caliber_A_fp32_full_accuracy": a_acc, "caliber_A_seconds": round(a_sec, 1),
           "caliber_B_int8_accuracy": b_acc,
           "caliber_C": {"subset": subset, "n_layer": 2, "seq_tokens": 8,
                         "cipher_vs_plain_trunc_consistency": consist,
                         "cipher_vs_full_label_agree": full_agree,
                         "note": "截断层精度随机水平为 P3 诚实发现（容量事实）；本口径只主张密文与明文截断参考的一致性"},
           "timestamp_utc": now()}
    write_json("benchmarks/results/compare_eval_calibers.json", out)

def phase_sst2(ckpt_dir, subset=20, seed=42):
    """SST-2 验证集（n=872）：FP32/INT8 准确率 + 密文管线一致性子集。"""
    import copy
    import torch
    from datasets import load_dataset as ld
    if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        os.makedirs(ckpt_dir, exist_ok=True)
        m = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
        t = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        m.save_pretrained(ckpt_dir); t.save_pretrained(ckpt_dir)
    from src.model.loader import BertSentimentPipeline, LABEL_MAP
    from src.model.pipeline import ModeBPipeline, PipelineConfig
    from src.model.quantize import FixedPointQuantizer, quantize_model_weights
    texts = ld("nyu-mll/glue", "sst2")["validation"]["sentence"]
    labels = ld("nyu-mll/glue", "sst2")["validation"]["label"]
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(DEV)
    t0 = time.perf_counter()
    a_acc, _ = bert_accuracy(model, tok, texts, labels)
    a_sec = time.perf_counter() - t0
    qmodel = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(DEV)
    quantize_model_weights(qmodel, linear_scheme="int8", embedding_scheme="int8")
    b_acc, _ = bert_accuracy(qmodel, tok, texts, labels)
    lab2idx = {v: k for k, v in LABEL_MAP.items()}
    pipe = BertSentimentPipeline(model_path=ckpt_dir)
    tmodel = truncated_model(copy.deepcopy(pipe.model), 2)
    import random
    rnd = random.Random(seed)
    # 密文管线不做 padding：筛选分词后 ≥ seq_tokens 的样例（隐含约束如实执行）
    tok0 = AutoTokenizer.from_pretrained(ckpt_dir)
    eligible = [i for i in range(len(texts))
                if len(tok0(texts[i], truncation=True, max_length=64)["input_ids"]) >= 8]
    rnd.shuffle(eligible)
    idx = eligible[:subset]
    sub_texts = [texts[i] for i in idx]
    tmodel.eval()
    ref_preds = []
    with torch.no_grad():
        for i in range(0, len(sub_texts), 8):
            enc = tok(sub_texts[i:i+8], truncation=True, max_length=8,
                      padding=True, return_tensors="pt")
            ref_preds += tmodel(**enc).logits.argmax(-1).tolist()
    cpipe = ModeBPipeline(pipe, PipelineConfig(n_layer=2, seq_tokens=8))
    cipher_labels = [lab2idx.get(cpipe.classify(t)["label"], -1) for t in sub_texts]
    plain_labels = [lab2idx.get("负面" if x == 0 else "正面", -1) for x in ref_preds]
    consist = sum(int(c == pl) for c, pl in
                  zip(cipher_labels, plain_labels)) / len(cipher_labels)
    out = {"schema": "a122-perf/1", "kind": "compare-eval-sst2",
           "checkpoint": ckpt_dir, "eval_set": "SST-2 validation (n=872)",
           "caliber_fp32_accuracy": a_acc, "caliber_int8_accuracy": b_acc,
           "caliber_cipher": {"subset": subset, "n_layer": 2, "seq_tokens": 8,
                              "cipher_vs_plain_trunc_consistency": consist},
           "literature_ref": {"EncFormer_SST2_acc": 0.9178, "EncFormer_plain_acc": 0.9243,
                              "source": "arXiv:2604.09975（已核验）"},
           "timestamp_utc": now()}
    write_json("benchmarks/results/compare_eval_sst2.json", out)

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
        phase_sst2(os.path.join("data", "models", "bert-base-uncased-sst2"),
                   subset=args.subset)

if __name__ == "__main__":
    main()
