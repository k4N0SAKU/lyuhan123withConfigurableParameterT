"""离线三元组生成器进程（可信离线假设——D7，如实写入 docs/02 §1.3）。

用法：
    python -m src.nodes.offline_triple_gen --store data/beaver/store.json \\
        --batch 100000

角色边界（S8）：本进程**可信且离线**，不与在线参与方交互；生成的完整三元组
只存在于生成器侧，部署时分别为 P1/P2 写独立份额库存（模拟实现见
beaver.split_triple 的说明）。每次生成附 SM3 完整性校验和。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gmssl import sm3

from src.crypto.beaver import DEFAULT_MODULUS
from src.crypto.secret_sharing import sample_fresh_mask


def generate_batch(store_path: str, batch_size: int,
                   modulus: int = DEFAULT_MODULUS) -> int:
    """生成 batch_size 个三元组追加写入库存，返回本次生成数。"""
    rows = []
    path = Path(store_path)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("triples", [])
    start = len(rows)
    masks = sample_fresh_mask(batch_size * 3, modulus)   # 批量均匀采样
    for i in range(batch_size):
        a, b = masks[3 * i], masks[3 * i + 1]
        rows.append({"gate_id": start + i, "a": a, "b": b,
                     "c": (a * b) % modulus, "used": False})
    checksum = sm3.sm3_hash(
        list("".join(json.dumps(t, sort_keys=True) for t in rows).encode("utf-8")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"triples": rows, "checksum": checksum}),
                    encoding="utf-8")
    return batch_size


def main(argv: list | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--batch", type=int, default=100000)
    args = parser.parse_args(argv)
    n = generate_batch(args.store, args.batch)
    total = len(json.loads(Path(args.store).read_text(encoding="utf-8"))["triples"])
    print(f"generated {n} triples -> {args.store} (total {total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
