# -*- coding: utf-8 -*-
"""多进程 BSGS 线性层 PoC（P9-①；GIL PoC 已证线程路线不可行——收益 1.0×）。

原理：linear_cipher 的 24 个 giant 组彼此独立（各自消耗 baby_rot 全集）——
按组分发给 W 个工作进程并行求值。交接设计（避免 IPC 放大）：
- baby 旋转（32 条密文）由主进程算一次，**序列化落临时目录**（32×1.64MB），
  工作进程一次性读入并缓存——避免每组重算旋转（否则总旋转量 768 vs 54）；
- 权重 (4.7MB) 仅初始化传一次，工作进程本地预建全部对角线（含组内预旋转）；
- 每任务返回 1 条序列化组累加密文（1.64MB × 24 组回传 ≈40MB，管道可承受）。

实测回答：**扣除全部 IPC/落盘开销后，单线性层多进程真实加速比是多少？**
口径：真实 BERT 权重 768×768（Q 权重）、L=2 token、mode-b。
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.crypto.ckks_ops import CKKSContext, PARAMS_MODE_B
from src.model.ops.packing import (DEFAULT_SPEC, build_diagonal_plaintexts,
                                   layout_input)

_WORKER = {}


def _init_worker(keys_dir: str, weight: "np.ndarray") -> None:
    """工作进程初始化：自建 public_only 上下文（含 galois）+ 本地预建对角线。"""
    _WORKER["ctx"] = CKKSContext(PARAMS_MODE_B, public_only=True,
                                 keys_dir=keys_dir)
    _WORKER["diags"] = build_diagonal_plaintexts(weight, DEFAULT_SPEC)
    _WORKER["baby_rot"] = None                       # 惰性加载（主进程落盘后）


def _load_baby_rot(baby: int) -> None:
    if _WORKER.get("baby_rot"):
        return
    ctx = _WORKER["ctx"]
    d = DEFAULT_SPEC.block_elems
    cache_dir = Path(os.environ["A122_POC_ROT_DIR"])
    _WORKER["baby_rot"] = [
        ctx.load_ct_bytes((cache_dir / f"rot_{k}.bin").read_bytes())
        for k in range(baby)]


def _giant_group(j: int) -> bytes:
    """单个 giant 组（含组内预旋转）：total_j = Σ_k baby_rot[k]·roll(diags[j·baby+k], j·baby)。"""
    _load_baby_rot(int(os.environ["A122_POC_BABY"]))
    ctx = _WORKER["ctx"]
    diags = _WORKER["diags"]
    baby = int(os.environ["A122_POC_BABY"])
    rolled = [np.roll(np.asarray(diags[j * baby + k]), j * baby).tolist()
              for k in range(baby)]
    acc = ctx.multiply_plain(_WORKER["baby_rot"][0], rolled[0], rescale=False)
    for k in range(1, baby):
        acc = ctx.add(acc, ctx.multiply_plain(_WORKER["baby_rot"][k], rolled[k],
                                              rescale=False))
    return ctx.serialize_ct_bytes(acc)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from src.model.loader import BertSentimentPipeline
    plain = BertSentimentPipeline()
    layer = plain.model.bert.encoder.layer[0]
    W = layer.attention.self.query.weight.detach().numpy()
    spec = DEFAULT_SPEC
    d = spec.block_elems
    baby = max(1, int(d ** 0.5))
    while d % baby != 0:
        baby += 1
    giant = d // baby

    keys_dir = os.path.join("build", "poc_mp_keys")
    os.makedirs(keys_dir, exist_ok=True)
    ctx_full = CKKSContext(PARAMS_MODE_B, public_only=False)
    ctx_full.save_keys(keys_dir, with_secret=False)
    ctx = CKKSContext(PARAMS_MODE_B, public_only=True, keys_dir=keys_dir)

    vals = layout_input([[0.1 * (i % 7)] * d for i in range(2)], 2, spec)
    ct = ctx.encrypt_vector(vals)
    diags = build_diagonal_plaintexts(W, spec)

    # ---- 顺序基线（分段计时）----
    t0 = time.perf_counter()
    baby_rot = [ct] + [ctx.rotate(ct, k) for k in range(1, baby)]
    t_baby = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    acc = []
    for j in range(giant):
        rolled = [np.roll(np.asarray(diags[j * baby + k]), j * baby).tolist()
                  for k in range(baby)]
        total = ctx.multiply_plain(baby_rot[0], rolled[0], rescale=False)
        for k in range(1, baby):
            total = ctx.add(total, ctx.multiply_plain(baby_rot[k], rolled[k],
                                                      rescale=False))
        acc.append(total)
    t_giant = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    result = acc[0]
    for j in range(1, giant):
        result = ctx.add(result, ctx.rotate(acc[j], j * baby))
    out = ctx.rescale_next(result)
    t_final = (time.perf_counter() - t0) * 1000
    seq_total = t_baby + t_giant + t_final
    print(f"顺序分段: baby旋转 {t_baby:.0f}ms + giant组 {t_giant:.0f}ms + 汇总 {t_final:.0f}ms"
          f" | 合计 ≈ {seq_total:.0f}ms")

    # ---- 多进程并行（baby 旋转落盘交接 → giant 组按 worker 分发）----
    ctx_mp = mp.get_context("spawn")
    n_workers = min(8, os.cpu_count() or 4)
    rot_dir = tempfile.mkdtemp(prefix="a122_poc_rot_")
    os.environ["A122_POC_ROT_DIR"] = rot_dir
    os.environ["A122_POC_BABY"] = str(baby)
    with ctx_mp.Pool(n_workers, initializer=_init_worker,
                     initargs=(keys_dir, W)) as pool:
        t0 = time.perf_counter()
        for k in range(baby):
            Path(rot_dir, f"rot_{k}.bin").write_bytes(
                ctx.serialize_ct_bytes(baby_rot[k]))
        t_write = (time.perf_counter() - t0) * 1000
        pool.map(_giant_group, [0])                  # 预热（spawn+ctx+对角线，一次性）
        t0 = time.perf_counter()
        baby_rot = [ct] + [ctx.rotate(ct, k) for k in range(1, baby)]
        t_baby2 = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        for k in range(baby):
            Path(rot_dir, f"rot_{k}.bin").write_bytes(
                ctx.serialize_ct_bytes(baby_rot[k]))
        t_write2 = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        acc_bytes = pool.map(_giant_group, list(range(giant)))
        t_giant2 = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        result = ctx.load_ct_bytes(acc_bytes[0])
        for j in range(1, giant):
            result = ctx.add(result, ctx.rotate(ctx.load_ct_bytes(acc_bytes[j]),
                                                j * baby))
        out2 = ctx.rescale_next(result)
        t_final2 = (time.perf_counter() - t0) * 1000
    total_mp = t_baby2 + t_write2 + t_giant2 + t_final2
    print(f"并行分段: baby旋转 {t_baby2:.0f}ms + 旋转落盘 {t_write2:.0f}ms"
          f" + giant组(含IPC) {t_giant2:.0f}ms + 汇总 {t_final2:.0f}ms"
          f" | 合计 ≈ {total_mp:.0f}ms")
    print(f"加速比（稳态）: {seq_total / total_mp:.2f}×"
          f"（workers={n_workers}, giant={giant}, baby={baby}）")
    return 0


if __name__ == "__main__":
    main()
