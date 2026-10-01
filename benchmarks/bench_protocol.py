"""P4 协议层基准（D8：所有性能结论必须来自本脚本产出的 JSON）。

度量项（对应 P4 出口条件与工作记录引用口径）：
- establish：三链路握手（双向认证 + SM2DH + KDF 派生）耗时分段（AUTH/KEYNEG）；
- channel_roundtrip：GCM 消息往返（按 payload 规模 1KB / 64KB 两档）；
- ratchet：一轮密钥轮换（含 KEY_ROTATE 往返）耗时；
- audit：1000 事件链追加与全链核验耗时；
- conversion：D5 真实路径入口/出口在 PARAMS_MODE_B 全槽（16384）规模耗时
  与字节数（[I] 项批量合并的基线口径：单密文逐次 vs 同请求 3 密文批量）。

用法：python -m benchmarks.bench_protocol [--rounds 5] [--toy]
产出：benchmarks/results/p4_protocol_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from src.common.envinfo import collect_environment
from src.common.perf import _stats_block, write_report
from src.crypto.ckks_ops import CKKSContext, CKKSParams, PARAMS_MODE_B
from src.crypto.gm_cipher import sm2_generate_keypair
from src.nodes.base import LoopbackLink
from src.protocol.auth import NodeIdentity, ca_issue_cert
from src.protocol.audit_log import AuditLog, verify_audit_entries
from src.protocol.conversion import (entry_decrypt_masked, entry_make_masked_ct,
                                     exit_infernode_compose,
                                     exit_keynode_prepare, policy_for)
from src.protocol.keylifecycle import KeyManager
from src.protocol.messages import MessageEnvelope, CommonHeader
from src.protocol.session import SecureChannel, Session

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "benchmarks" / "results" / "p4_protocol_bench.json"
NOW = 1_700_000_000_000


def _identity(ca_priv, node_id):
    pub, priv = sm2_generate_keypair()
    return NodeIdentity(node_id=node_id, role=0, static_pub=pub,
                        static_priv=priv,
                        cert=ca_issue_cert(ca_priv, node_id, pub, now_ms=NOW))


def _mk_pair(ca_priv, name="p0", **kw):
    la, lb = LoopbackLink.create_pair()
    sid = os.urandom(16)
    cha = SecureChannel(Session(sid, 0), _identity(ca_priv, name + "-a"),
                        KeyManager(), clock_ms=lambda: NOW, **kw)
    chb = SecureChannel(Session(sid, 1), _identity(ca_priv, name + "-b"),
                        KeyManager(), clock_ms=lambda: NOW, **kw)
    return cha, chb, la, lb


def _timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return (time.perf_counter() - t0) * 1000.0, out


def bench_establish(rounds: int) -> dict:
    """三链路式握手基准：单链路（双向认证 + SM2DH + KDF 派生）双线程全程。"""
    ca_pub, ca_priv = sm2_generate_keypair()
    total_ms = []
    for _ in range(rounds):
        cha, chb, la, lb = _mk_pair(ca_priv)
        import threading
        th = [threading.Thread(target=cha.establish, args=(la, ca_pub, True)),
              threading.Thread(target=chb.establish, args=(lb, ca_pub, False))]
        t0 = time.perf_counter()
        for t in th:
            t.start()
        for t in th:
            t.join()
        total_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "note": "单链路 establish 全程墙钟（双向认证 + SM2DH + KDF 派生，"
                "双线程并发）；认证/协商分段细分见 tests/unit/test_session",
        "establish_total": _stats_block(total_ms),
    }


def bench_channel_roundtrip(cha, chb, payload_sizes, rounds) -> dict:
    out = {}
    for size in payload_sizes:
        samples = []
        for i in range(rounds):
            env = MessageEnvelope(header=CommonHeader(msg_type=0x0115),
                                  payload_type="HeartbeatPayload",
                                  payload={"i": i, "pad": "a" * size})
            ms, _ = _timed(cha.send_message, env)
            ms2, _ = _timed(chb.recv_message)
            samples.append(ms + ms2)
        out[f"payload_{size}B"] = _stats_block(samples)
    return out


def bench_conversion(rounds: int, params) -> dict:
    """D5 真实路径在指定参数集的入口/出口耗时（全槽）。[I] 项批量基线：
    batch=3 模拟同层 QKV 三密文合并往返（省 2 次 P0 往返）。"""
    full = CKKSContext(params)
    d = tempfile.mkdtemp(prefix="a122_bench_keys_")
    full.save_keys(d, with_secret=True)
    del full
    sec = CKKSContext(params, public_only=False, keys_dir=d)
    pub = CKKSContext(params, public_only=True, keys_dir=d)
    n_slots = pub.slot_count
    values = [(i % 13 - 6) * 0.25 for i in range(n_slots)]
    ct = pub.encrypt_vector(values)
    entry_ms, exit_ms = [], []
    entry_bytes, exit_bytes = 0, 0
    for _ in range(rounds):
        ms, (masked, r) = _timed(entry_make_masked_ct, pub, ct)
        entry_ms.append(ms)
        y1 = entry_decrypt_masked(sec, masked)
        entry_bytes = masked.size_bytes + len(y1) * 8
        vals_fixed = [(i + 1) * 4096 for i in range(16)]      # 1/16 步进小值
        a1 = [(v + 1) % (1 << 64) for v in vals_fixed]        # 分享 a1 = x+1
        ms2, (z, enc_s) = _timed(exit_keynode_prepare, pub, a1)
        exit_ms.append(ms2)
        exit_bytes = len(enc_s) + len(z) * 8
        a2 = [(1 << 64) - 1] * 16                             # a2 = −1 ⇒ x=a1+a2
        ct2 = exit_infernode_compose(pub, enc_s, z, a2)
    got = sec.decrypt(ct2)[:16]
    expect = [i / 16.0 for i in range(1, 17)]     # a1=x+1、a2=−1 ⇒ v=x/16
    err = max(abs(g - e) for g, e in zip(got, expect))
    return {
        "params": params.name, "slots": n_slots,
        "entry_make_masked_ct": _stats_block(entry_ms),
        "exit_keynode_prepare": _stats_block(exit_ms),
        "entry_wire_bytes_approx": entry_bytes,
        "exit_wire_bytes_approx": exit_bytes,
        "exit_reconstruct_max_err": err,
        "policy": {"value_bound": policy_for(pub).value_bound,
                   "entry_mask_hi": policy_for(pub).entry_mask_hi,
                   "exit_window": [policy_for(pub).exit_mask_lo,
                                   policy_for(pub).exit_mask_hi]},
        "note": "入口含 Enc(r) 加密+mod_switch+同态加；出口含 Enc(s) 加密+"
                "密文域合成。单密文逐次口径；批量合并收益见工作记录 [I] 项",
    }


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args(argv)

    ca_pub, ca_priv = sm2_generate_keypair()
    report = {
        "schema": "a122-perf/1",
        "kind": "performance",
        "workload": "p4_protocol_lifecycle",
        "mode": "p4",
        "config": {"rounds": args.rounds,
                   "ratchet": {"mode": "messages", "threshold": 4}},
        "environment": collect_environment(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report["establish"] = bench_establish(args.rounds)

    # 通道往返 + ratchet（带内口径：阈值 4，第 5 条消息的发送触发
    # KEY_ROTATE+密钥切换，接收侧完成前向派生——整轮墙钟即轮换成本）
    cha, chb, la, lb = _mk_pair(ca_priv, ratchet_mode="messages",
                                ratchet_threshold=4)
    import threading
    th = [threading.Thread(target=cha.establish, args=(la, ca_pub, True)),
          threading.Thread(target=chb.establish, args=(lb, ca_pub, False))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    report["channel_roundtrip"] = bench_channel_roundtrip(
        cha, chb, [1024, 65536], args.rounds)
    rot = []
    for _ in range(args.rounds):
        for i in range(4):
            cha.send_message(MessageEnvelope(
                header=CommonHeader(msg_type=0x0115),
                payload_type="HeartbeatPayload", payload={"i": i}))
            env = chb.recv_message()
            while env.payload_type == "KeyRotatePayload":
                env = chb.recv_message()
        t0 = time.perf_counter()
        cha.send_message(MessageEnvelope(
            header=CommonHeader(msg_type=0x0115),
            payload_type="HeartbeatPayload", payload={"rot": 1}))
        env = chb.recv_message()
        while env.payload_type == "KeyRotatePayload":
            env = chb.recv_message()
        rot.append((time.perf_counter() - t0) * 1000)
    report["ratchet_note"] = ("轮换计时口径：触发阈值后的整轮（KEY_ROTATE 帧 + "
                              "新钥业务帧 + 接收侧前向派生）墙钟；触发前的"
                              "4 条计数消息不计入")
    report["ratchet_epoch"] = {"send": cha.km.current_epoch("send"),
                               "recv": chb.km.current_epoch("recv")}
    report["ratchet_round"] = _stats_block(rot)

    # 审计链
    clock = [NOW]
    log = AuditLog(actor="bench", clock_ms=lambda: (clock.__setitem__(
        0, clock[0] + 1), clock[0])[1])
    t0 = time.perf_counter()
    for i in range(1000):
        log.append("P1", "BENCH_EVENT", {"i": i})
    append_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    ok, errs = verify_audit_entries(log.entries)
    verify_ms = (time.perf_counter() - t0) * 1000
    report["audit_chain_1000"] = {"append_total_ms": append_ms,
                                  "verify_total_ms": verify_ms,
                                  "verify_ok": ok,
                                  "events": len(log.entries)}

    # D5 真实路径（mode-b 全槽 + 玩具参数对照）
    report["conversion_mode_b_full_slot"] = bench_conversion(
        max(3, args.rounds // 2), PARAMS_MODE_B)
    toy = CKKSParams(name="p4-protocol-toy", poly_modulus_degree=1 << 12,
                     coeff_mod_bit_sizes=(30, 30, 40), scale_log2=30,
                     slots=1 << 11)
    report["conversion_toy"] = bench_conversion(args.rounds, toy)

    dest = _destroy_stats(cha)
    report["destroy"] = dest
    write_report(report, OUT)
    print(f"written -> {OUT.relative_to(REPO_ROOT)}")
    print(f"establish p50: {report['establish']['establish_total']['p50']:.1f} ms; "
          f"roundtrip 64KB p50: "
          f"{report['channel_roundtrip']['payload_65536B']['p50']:.1f} ms; "
          f"mode-b entry p50: "
          f"{report['conversion_mode_b_full_slot']['entry_make_masked_ct']['p50']:.1f} ms")
    return 0


def _destroy_stats(cha) -> dict:
    t0 = time.perf_counter()
    detail = cha.destroy("bench_end")
    return {"destroy_total_ms": (time.perf_counter() - t0) * 1000,
            "keys_destroyed": detail["keys_destroyed"],
            "all_zeroed": detail["all_zeroed"]}


if __name__ == "__main__":
    raise SystemExit(main())
