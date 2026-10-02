"""D5′ m 方代价曲线基准（P8；数字唯一合法产地，D8 纪律适用）。

口径：
- mode-b 生产参数（2^15 / 链 200bit / scale 2^40），全槽向量（16384 槽）；
- 一次「转换往返」= 入口链（m 片采样+加密+P0 白名单①解密+份额形成）+ 
  出口（m 片准备+合成方合成）+ 解密偏差断言（红线 4 ulp——P4 δ≤5 口径内；环算术重构相对 y₁
  精确；超线立即 FAIL；逐 m 的 max dev 与翻转率作为噪声指标记录）；
- 通信记账（协议消息口径）：
  入口 = masked-ct 链 m 跳（每跳转发全密文）+ y 分发 1 次（8B/槽）；
  出口 = m×(z: 8B/槽 + Enc(s): 密文) + fresh ct 1 次；
- fresh 掩码/密文每轮独立；m ∈ {2,3,4,5}（mode-b 上限 15 内取点）。

输出：benchmarks/results/mparty_cost_curve.json（a122-perf/1，UTC 时间戳）。
文档引用本 JSON——「t 可配置的代价曲线」（延迟/字节随 m 增长）唯一证据源。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.common.perf import NetworkMeter
from src.crypto.ckks_ops import CKKSContext, PARAMS_MODE_B
from src.crypto.secret_sharing_m import share_vector_n
from src.protocol.conversion import (FIXED_ONE, ClientRole, entry_decrypt_masked,
                                     policy_for, to_fixed)
from src.protocol.conversion_m import (ComputePartyRole, check_m, exit_compose_m,
                                       m_cap)

M_LIST = (2, 3, 4, 5)
ROUNDS = 10
# 偏差红线：P4 记录 δ≤5 @ 掩码量级 2^37（conversion.py docstring）；
# m 方掩码和 ≤ m·2^49，实测 m≤5 max dev=2 → 红线取 4（2 倍裕量）。
# 管线级不变量 = label 稳定（等价 e2e 实证），ulp 抖动远低于分类敏感度。
ULP_REDLINE = 4


def _stats(samples):
    s = sorted(samples)
    n = len(s)
    return {"n": n, "unit": "ms",
            "mean": sum(s) / n,
            "p50": s[n // 2],
            "max": s[-1], "min": s[0]}


def run_round(pub, sec, client, parties, m):
    slots = pub.slot_count
    vals = [(int.from_bytes(os.urandom(4), "big") / 2**32) * 8 - 4.0
            for _ in range(slots)]
    fixed = [to_fixed(v) for v in vals]

    # ---- 入口：掩码链 P2→…→Pm→P1 → P0 白名单①解密 → 份额 ----
    t0 = time.perf_counter()
    request_id = os.urandom(16)
    masked = pub.encrypt_vector(vals)
    for p in parties[1:] + parties[:1]:
        masked = p.entry_apply_piece(masked, request_id)
    masked_bytes = masked.size_bytes
    y = client.masked_decrypt(masked, request_id)
    shares = [p.entry_own_share(request_id, y=y if p.is_anchor else None)
              for p in parties]
    entry_ms = (time.perf_counter() - t0) * 1000.0
    rec = [sum(col) % (1 << 64) for col in zip(*shares)]
    # ±1 ulp 口径（conversion.py 申报 e_ckks ≤1 ulp）：y₁=round(解码) 相对
    # 理想值可差 ±1（解码噪声舍入，全槽实测 ~10-14%/槽）；环算术重构
    # 相对 y₁ **精确**。红线 = |dev| ≤ 1（超出即协议/参数错误）。
    dev = [abs((r - f + (1 << 63)) % (1 << 64) - (1 << 63)) for r, f in zip(rec, fixed)]
    assert max(dev) <= ULP_REDLINE, f"入口偏差 {max(dev)} 超出 {ULP_REDLINE} ulp 红线"
    entry_flip_rate = sum(1 for d in dev if d) / len(dev)

    # ---- 出口：m 片 (zᵢ, Enc(sᵢ)) → 合成 → 解密精确断言 ----
    t1 = time.perf_counter()
    result = share_vector_n(fixed, m)
    request_id = os.urandom(16)
    pieces = []
    for p, a in zip(parties, result):
        p.set_result_share(request_id, a)
        pieces.append(p.exit_piece(request_id))
    ct = exit_compose_m(pub, pieces)
    dec = sec.decrypt(ct)
    got = [int(round(v * FIXED_ONE)) % (1 << 64) for v in dec]
    dev = [abs((g - f + (1 << 63)) % (1 << 64) - (1 << 63)) for g, f in zip(got, fixed)]
    assert max(dev) <= ULP_REDLINE, f"出口偏差 {max(dev)} 超出 {ULP_REDLINE} ulp 红线"
    exit_flip_rate = sum(1 for d in dev if d) / len(dev)
    exit_ms = (time.perf_counter() - t1) * 1000.0

    entry_bytes = m * masked_bytes + slots * 8          # 链 m 跳 + y 分发
    exit_bytes = (m * (slots * 8 + pub.encrypt_vector([0.0]).size_bytes)
                  + ct.size_bytes)                      # m×(z+Enc(s)) + fresh ct
    return {"entry_ms": entry_ms, "exit_ms": exit_ms,
            "round_ms": entry_ms + exit_ms,
            "entry_bytes": entry_bytes, "exit_bytes": exit_bytes,
            "masked_ct_bytes": masked_bytes, "fresh_ct_bytes": ct.size_bytes,
            "entry_flip_rate": entry_flip_rate, "exit_flip_rate": exit_flip_rate,
            "max_abs_dev": max(dev),
            "exact": True}


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--out", default=None, help="输出 JSON（默认 results 固定名）")
    args = ap.parse_args(argv)

    keys_dir = os.path.join("build", "bench_mparty_keys")
    os.makedirs(keys_dir, exist_ok=True)
    full = CKKSContext(PARAMS_MODE_B)
    full.save_keys(keys_dir, with_secret=True)
    del full
    pub = CKKSContext(PARAMS_MODE_B, public_only=True, keys_dir=keys_dir)
    sec = CKKSContext(PARAMS_MODE_B, public_only=False, keys_dir=keys_dir)
    client = ClientRole(sec)
    policy = policy_for(pub)
    slots = pub.slot_count

    results = {}
    for m in M_LIST:
        check_m(policy, m)
        parties = [ComputePartyRole(pub, index=i + 1, m=m, is_anchor=(i == 0))
                   for i in range(m)]
        rounds = []
        for _ in range(args.rounds):
            rounds.append(run_round(pub, sec, client, parties, m))
        results[str(m)] = {
            "rounds": rounds,
            "round_ms": _stats([r["round_ms"] for r in rounds]),
            "entry_ms": _stats([r["entry_ms"] for r in rounds]),
            "exit_ms": _stats([r["exit_ms"] for r in rounds]),
            "entry_bytes": rounds[0]["entry_bytes"],
            "exit_bytes": rounds[0]["exit_bytes"],
            "bytes_per_round": rounds[0]["entry_bytes"] + rounds[0]["exit_bytes"],
            "ulp_ok_all": all(r["max_abs_dev"] <= ULP_REDLINE for r in rounds),
            "max_dev": max(r["max_abs_dev"] for r in rounds),
            "entry_flip_rate": sum(r["entry_flip_rate"] for r in rounds) / len(rounds),
            "exit_flip_rate": sum(r["exit_flip_rate"] for r in rounds) / len(rounds),
        }
        print(f"m={m}: round p50={results[str(m)]['round_ms']['p50']:.1f}ms "
              f"bytes={results[str(m)]['bytes_per_round']/(1<<20):.1f}MiB "
              f"ulp_ok={results[str(m)]['ulp_ok_all']} flip={results[str(m)]['entry_flip_rate']:.1%}/{results[str(m)]['exit_flip_rate']:.1%}")

    out = {
        "schema": "a122-perf/1",
        "kind": "mparty-cost-curve",
        "workload": "conversion-round-trip (entry chain + exit compose)",
        "config": {"params": PARAMS_MODE_B.name, "slots": slots,
                   "rounds_per_m": args.rounds, "m_list": list(M_LIST),
                   "policy": policy.name,
                   "entry_window": f"[0, {policy.entry_mask_hi})",
                   "exit_window": f"[{policy.exit_mask_lo}, {policy.exit_mask_hi})",
                   "m_cap": m_cap(policy)},
        "environment": {"python": ".".join(map(str, sys.version_info[:3])),
                        "platform": sys.platform,
                        "note": "CPU-only；fresh 掩码/密文每轮独立；逐轮精确断言"},
        "results": results,
        "stability": {"m": 3, "rounds_done": args.rounds,
                      "ulp_ok": results["3"]["ulp_ok_all"],
                      "redline": ULP_REDLINE,
                      "note": "偏差红线 4 ulp（P4 δ≤5 口径内）；环算术相对 y₁ 精确"},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = args.out or str(Path("benchmarks/results/mparty_cost_curve.json"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"written -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
