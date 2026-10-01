"""微调 BERT-base-chinese 情感分类头（P0：为分类精度实验提供 FP32 检查点）。

产出：data/models/bert-base-chinese-sentiment/（含分类头的完整模型，供
BertSentimentPipeline 直接加载）。

可复现性：固定随机种子（默认 42），训练超参写死在 config 里并写入检查点目录
的 train_meta.json；CPU 训练约 1~3 分钟。

用法：python -m benchmarks.finetune_bert [--epochs 4] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "sentiment"
BASE_MODEL = REPO_ROOT / "data" / "models" / "bert-base-chinese"
OUT_DIR = REPO_ROOT / "data" / "models" / "bert-base-chinese-sentiment"


def load_tsv(path: Path) -> list:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not ln.strip():
            continue
        sid, label, text = ln.split("\t")
        rows.append((int(label), text))
    return rows


class SentimentDataset(Dataset):
    def __init__(self, rows: list, tokenizer, max_len: int = 64) -> None:
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        label, text = self.rows[idx]
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for enc, labels in loader:
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            pred = logits.argmax(dim=-1).cpu()
            correct += int((pred == torch.tensor(labels)).sum())
            total += len(labels)
    return correct / total


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    train_rows = load_tsv(DATA_DIR / "train.tsv")
    eval_rows = load_tsv(DATA_DIR / "eval.tsv")
    train_loader = DataLoader(SentimentDataset(train_rows, tokenizer),
                              batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(SentimentDataset(eval_rows, tokenizer), batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained(str(BASE_MODEL), num_labels=2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc, best_epoch = 0.0, -1
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for enc, labels in train_loader:
            labels_t = torch.tensor(labels, device=device)
            out = model(**{k: v.to(device) for k, v in enc.items()}, labels=labels_t)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(out.loss)
        acc = evaluate_accuracy(model, eval_loader, device)
        print(f"epoch {epoch}: train_loss={total_loss / len(train_loader):.4f} "
              f"eval_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)

    meta = {
        "base_model": "bert-base-chinese",
        "dataset": {"train": str(DATA_DIR / "train.tsv"), "eval": str(DATA_DIR / "eval.tsv"),
                    "train_rows": len(train_rows), "eval_rows": len(eval_rows)},
        "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "seed": args.seed, "max_len": 64},
        "best_eval_acc": best_acc,
        "best_epoch": best_epoch,
        "device": device,
        "train_wall_s": round(time.perf_counter() - t0, 1),
    }
    (OUT_DIR / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved checkpoint -> {OUT_DIR}")
    print(f"best eval_acc={best_acc:.4f} (epoch {best_epoch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
