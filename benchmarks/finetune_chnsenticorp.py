# -*- coding: utf-8 -*-
"""ChnSentiCorp 官方切分微调（P9 对比实验；替换自建 600 条）。

D8：训练配置、损失曲线、验证准确率全部写入 benchmarks/results/finetune_chnsenticorp.json。
产出检查点：data/models/bert-base-chinese-sentiment-chn/（供三口径与跨域评测）。
"""
import json, os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset

SEED = 42
BASE = "data/models/bert-base-chinese"
OUT_DIR = "data/models/bert-base-chinese-sentiment-chn"
EPOCHS, BATCH, LR, MAXLEN = 2, 16, 2e-5, 64

torch.manual_seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)
ds = load_dataset("lansinuote/ChnSentiCorp")

def encode(split):
    enc = tok([t for t in ds[split]["text"]], truncation=True, max_length=MAXLEN,
              padding="max_length", return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"],
                         torch.tensor(ds[split]["label"]))

train_ds = encode("train")
val_ds = encode("validation")
g = torch.Generator().manual_seed(SEED)
loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=g)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
total = len(loader) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(total * 0.06), total)
model.train()

t0 = time.perf_counter()
loss_curve, steps = [], 0
for ep in range(EPOCHS):
    for ids, mask, labels in loader:
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        steps += 1
        loss_curve.append(round(loss.item(), 4))
        if steps % 100 == 0:
            el = (time.perf_counter() - t0)
            print(f"step {steps}/{total} loss {loss.item():.4f} elapsed {el:.0f}s", flush=True)
train_s = time.perf_counter() - t0

model.eval()
def acc(split_ds):
    ldr = DataLoader(split_ds, batch_size=32)
    correct = 0
    with torch.no_grad():
        for ids, mask, labels in ldr:
            pred = model(input_ids=ids, attention_mask=mask).logits.argmax(-1)
            correct += int((pred == labels).sum())
    return correct / len(split_ds)

val_acc = acc(val_ds)
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
out = {
    "schema": "a122-perf/1", "kind": "finetune-chnsenticorp",
    "config": {"base": BASE, "dataset": "lansinuote/ChnSentiCorp train(9600)",
               "epochs": EPOCHS, "batch": BATCH, "lr": LR, "max_length": MAXLEN,
               "seed": SEED, "steps": steps},
    "val_accuracy": val_acc, "val_size": len(val_ds),
    "train_seconds": train_s, "loss_curve_head": loss_curve[:5],
    "checkpoint": OUT_DIR,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
json.dump(out, open("benchmarks/results/finetune_chnsenticorp.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"DONE val_acc={val_acc:.4f} train={train_s:.0f}s ckpt={OUT_DIR}")
