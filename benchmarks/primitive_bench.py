"""原语级微基准（P2 任务 7）：国密 / CKKS / 打包线性层，产出 a122-perf/1 JSON。

用法（仓库根目录）：
    python -m benchmarks.primitive_bench --suite gm --rounds 20
    python -m benchmarks.primitive_bench --suite ckks --rounds 20
    python -m benchmarks.primitive_bench --suite packing --rounds 5   # 真实 shape，较慢

口径：时间分段经 PerfSession（P50/P95 跨轮统计）；CKKS 基准在 PARAMS_MODE_B
（2^15 主线参数）上运行；packing 基准为 GPT-2 真实 shape（d=768，L=2 token）
的单层密文矩阵乘，对角线明文编码计入 setup 不计入被测段。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA_DIR = REPO_ROOT / "data" / "demo"


def bench_gm(rounds: int) -> dict:
    from src.common.perf import PerfSession
    from src.crypto.gm_cipher import (CTX_AUTH, sm2_ecdh, sm2_generate_keypair,
                                      sm2_sign, sm2_verify, sm3_hash,
                                      sm4_gcm_decrypt, sm4_gcm_encrypt)

    payload = os.urandom(1024)                       # 1KB 标准载荷
    key = os.urandom(16)
    nonce = os.urandom(12)
    pub, priv = sm2_generate_keypair()
    pub2, priv2 = sm2_generate_keypair()
    sig = sm2_sign(priv, payload, context=CTX_AUTH)
    ct, tag = sm4_gcm_encrypt(key, payload, aad=b"bench", nonce=nonce)

    def run(ctx):
        with ctx.timer.segment("sm3_hash_1kb"):
            sm3_hash(payload)
        with ctx.timer.segment("sm4_gcm_encrypt_1kb"):
            sm4_gcm_encrypt(key, payload, aad=b"bench", nonce=nonce)
        with ctx.timer.segment("sm4_gcm_decrypt_1kb"):
            sm4_gcm_decrypt(key, ct, tag, aad=b"bench", nonce=nonce)
        with ctx.timer.segment("sm2_sign"):
            sm2_sign(priv, payload, context=CTX_AUTH)
        with ctx.timer.segment("sm2_verify"):
            sm2_verify(pub, payload, sig, context=CTX_AUTH)
        with ctx.timer.segment("sm2_ecdh"):
            sm2_ecdh(priv, pub2)

    session = PerfSession(workload="gm-primitives", mode="micro-bench",
                          config={"payload_bytes": 1024, "nonce": "12B 显式"},
                          environment=_env(), keep_rounds=False)
    return session.run(run, rounds)


def bench_ckks(rounds: int) -> dict:
    from src.common.perf import PerfSession
    from src.crypto.ckks_ops import CKKSContext, PARAMS_MODE_B

    ctx = CKKSContext(PARAMS_MODE_B)     # P0 侧口径：解密段需要私钥
    rng = np.random.default_rng(2026)
    x = rng.normal(0, 1.0, 256).tolist()             # ~1KB（256×float64）
    w = rng.normal(0, 0.05, 256).tolist()
    ct_w = ctx.encrypt_vector(w)
    ct_in = ctx.encrypt_vector(x)
    rot_out = [0.0] * 256

    def run(ctxb):
        with ctxb.timer.segment("ckks_encrypt_256floats"):
            c = ctx.encrypt_vector(x)
        with ctxb.timer.segment("ckks_decrypt_256floats"):
            ctx.decrypt(c)
        with ctxb.timer.segment("ckks_multiply_plain_256"):
            ctx.multiply_plain(ct_in, w)
        with ctxb.timer.segment("ckks_multiply_ct_256"):
            ctx.multiply_ct(ct_in, ct_w)
        with ctxb.timer.segment("ckks_rotate_256"):
            ctx.rotate(ct_in, 1)
        _ = rot_out

    session = PerfSession(workload="ckks-primitives", mode="micro-bench",
                          config={"params": PARAMS_MODE_B.name,
                                  "poly_modulus_degree": PARAMS_MODE_B.poly_modulus_degree,
                                  "elements": 256, "scale_log2": 40},
                          environment=_env(), keep_rounds=False)
    return session.run(run, rounds)


def bench_packing(rounds: int) -> dict:
    """GPT-2 真实 shape 线性层：d=768，L=2 token（单密文）。"""
    from src.common.perf import PerfSession
    from src.crypto.ckks_ops import CKKSContext, PARAMS_MODE_B
    from src.model.ops.packing import (DEFAULT_SPEC, build_diagonal_plaintexts,
                                       layout_input, linear_cipher)

    ctx = CKKSContext(PARAMS_MODE_B)     # P0 侧口径：解密段需要私钥
    rng = np.random.default_rng(768)
    W = rng.normal(0, 0.05, (768, 768))
    tokens = [rng.normal(0, 0.5, 768) for _ in range(2)]
    layout = layout_input(tokens, 2, DEFAULT_SPEC)
    diags = build_diagonal_plaintexts(W, DEFAULT_SPEC)   # setup：不计入被测段
    ct_in = ctx.encrypt_vector(layout)

    def run(ctxb):
        with ctxb.timer.segment("packing_linear_768x768_L2"):
            out = linear_cipher(ctx, ct_in, W, 2, spec=DEFAULT_SPEC,
                                diagonals=diags)
        with ctxb.timer.segment("decrypt_output_1536slots"):
            ctx.decrypt(out)[:1536]

    # P2-R1 评审 E 项：轮数下限提升至 5（单层 4.15s，20 轮需 ~83s×开销，
    # 报告 JSON 的 config 段注明"慢测试豁免口径：5 轮 < 框架标准 20 轮"）
    session = PerfSession(workload="packing-linear", mode="micro-bench",
                          config={"spec": "d=768 gap=2 slots=2^14",
                                  "tokens": 2, "bsgs_rotations": 54,
                                  "plaintext_mults": 768,
                                  "rounds_exemption": "5 轮（慢基准豁免 20 轮标准，"
                                                      "单轮 4.15s）"},
                          environment=_env(), keep_rounds=False)
    return session.run(run, rounds)


def _env() -> dict:
    from src.common.envinfo import collect_environment
    return collect_environment(probe_gpu=True)


def main(argv: list | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=["gm", "ckks", "packing", "all"])
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args(argv)

    from src.common.perf import write_report
    suites = (["gm", "ckks", "packing"] if args.suite == "all" else [args.suite])
    runners = {"gm": bench_gm, "ckks": bench_ckks, "packing": bench_packing}
    for name in suites:
        rounds = args.rounds if name != "packing" else max(5, min(args.rounds, 5))
        report = runners[name](rounds)
        report["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"primitives_{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        write_report(report, out)
        try:
            print(f"[{name}] rounds={report['rounds_completed']} -> {out.name}")
        except ValueError:
            print(f"[{name}] rounds={report['rounds_completed']} -> {out}")
        for seg, block in report["summary"]["segments"].items():
            print(f"   {seg}: p50={block['p50']:.3f}ms p95={block['p95']:.3f}ms n={block['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
