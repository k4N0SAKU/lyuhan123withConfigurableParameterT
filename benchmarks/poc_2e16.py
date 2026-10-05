# -*- coding: utf-8 -*-
"""2^16 评审路径 PoC（P9-1b）：SEC_LEVEL_TYPE.NONE 下 N=2^16 参数的可用性实证。

登记口径（docs/01 §5.3）：2^15 主线 vs 2^16-NONE 的取舍在 estimator 数据齐后
由团队决策；本 PoC 只实证**参数可用性**（SEAL 接受性/密文尺寸/加解密精确性/
转换次数算术），不切换主线、不含旋转（Galois 全集 keygen ≈45min，最小集方案
见 docs/01）。安全性 provenance：estimator C≈2^130.2≥128（历史实测，docs/01）。
输出：benchmarks/results/poc_2e16.json（a122-perf/1）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crypto.ckks_ops import PARAMS_MODE_B, PARAMS_MODE_B_2E16, CKKSContext
from src.protocol.conversion import FIXED_ONE


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    ctx = CKKSContext(PARAMS_MODE_B_2E16, public_only=False,
                      sec_level="NONE", galois=False)
    keygen_ms = (time.perf_counter() - t0) * 1000.0
    slots = ctx.slot_count

    vals = [(int.from_bytes(os.urandom(4), "big") / 2**32) * 8 - 4.0
            for _ in range(slots)]
    t1 = time.perf_counter()
    ct = ctx.encrypt_vector(vals)
    enc_ms = (time.perf_counter() - t1) * 1000.0
    ct_size = ct.size_bytes

    t2 = time.perf_counter()
    ct2 = ctx.add(ct, ct)                       # 转换入口同款同态加（无旋转依赖）
    add_ms = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    dec = ctx.decrypt(ct2)
    dec_ms = (time.perf_counter() - t3) * 1000.0
    # ±ulp 口径（与 bench_mparty 同款红线 4）：比较基准 = round(2·v·2^16)
    # （对 2 倍明文的直接定点化）；偏差分布入册（实测 max|dev|=1、0.1%）
    got = [int(round(v * FIXED_ONE)) % (1 << 64) for v in dec]
    want = [int(round(2 * v * FIXED_ONE)) % (1 << 64) for v in vals]
    devs = [abs((g - w + (1 << 63)) % (1 << 64) - (1 << 63)) for g, w in zip(got, want)]
    exact = max(devs) <= 4
    max_dev = max(devs)
    flip_rate = sum(1 for d in devs if d) / len(devs)

    out = {
        "schema": "a122-perf/1", "kind": "poc-2e16",
        "config": {"params": PARAMS_MODE_B_2E16.name, "poly": PARAMS_MODE_B_2E16.poly_modulus_degree,
                   "chain": list(PARAMS_MODE_B_2E16.coeff_mod_bit_sizes),
                   "scale_log2": PARAMS_MODE_B_2E16.scale_log2, "slots": slots,
                   "sec_level": "NONE", "galois_keys": False,
                   "security_provenance": "lattice-estimator C≈2^130.2≥128（docs/01 §5.3 登记值）"},
        "results": {
            "seal_accepted": True, "keygen_ms_no_galois": keygen_ms,
            "ct_size_bytes": ct_size, "ct_size_mib": ct_size / 1048576,
            "encrypt_ms": enc_ms, "homomorphic_add_ms": add_ms,
            "decrypt_ms": dec_ms, "roundtrip_ulp_ok": exact,
            "roundtrip_max_dev": max_dev, "roundtrip_flip_rate": flip_rate,
            "conversions_note": "L=16 单密文 ⇒ 每层域转换 96 次（mode-b 主线 192 次的一半，docs/01 §5.3）",
        },
        "baseline_mode_b": {"slots": 1 << 14, "ct_size_mib": 1.64,
                            "conversions_per_layer_at_L16": 192},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    p = Path("benchmarks/results/poc_2e16.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"SEAL accepted N=2^16 (NONE): slots={slots} ct={ct_size/1048576:.2f}MiB "
          f"keygen(无Galois)={keygen_ms:.0f}ms max_dev={max_dev} flip={flip_rate:.2%}")
    print(f"written -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
