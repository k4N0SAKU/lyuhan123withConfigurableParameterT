"""三方节点编排（P4：会话建立 → 推理数据面 → ratchet → 销毁 → 审计核验）。

场景编排为**同步驱动**（每链路 FIFO，握手/数据面脚本严格交替；异步消息循环
与对抗性交互属 P5 攻击套件）。链路拓扑（docs/01 §2）：
    p0_client ↔ p1_keynode（17401 方向语义）
    p0_client ↔ p2_infernode（17402）
    p1_keynode ↔ p2_infernode（17403）
节点日志按节点分文件（logs/<node>.jsonl）。

多进程模式：_mp_node_worker 为模块级函数（Windows spawn 安全），每进程加载
自己的身份文件、执行命令表、回传报告——CKKS 密钥经供给目录分发（P0 独享
私钥目录）。
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.crypto.gm_cipher import CTX_RESULT, sm2_sign, sm2_verify
from src.nodes.base import BaseNode, LoopbackLink, MpLink, save_report
from src.nodes.client import ClientNode
from src.nodes.infernode import InferNode
from src.nodes.keynode import KeyNode
from src.protocol import messages as M
from src.protocol.audit_log import verify_audit_entries
from src.protocol.conversion import pack_ints, unpack_ints
from src.protocol.session import make_envelope, payload_sign_base

LINK_NAMES = ("p0-p1", "p0-p2", "p1-p2")
_NODE_IDS = {"p0-p1": ("p0_client", "p1_keynode"),
             "p0-p2": ("p0_client", "p2_infernode"),
             "p1-p2": ("p1_keynode", "p2_infernode")}


def build_local_nodes(prov_dir: str, params, log_dir: Optional[str] = None,
                      clock_ms=None) -> Tuple[ClientNode, KeyNode, InferNode]:
    """进程内构建三节点（含各自审计文件；不建链路）。params 为 CKKSParams。"""
    from src.nodes.provision import load_ca_pub, load_identity
    ca_pub = load_ca_pub(prov_dir)
    log = Path(log_dir) if log_dir else None
    audit_path = (lambda nid: str(log / f"{nid}.jsonl")) if log else (lambda nid: None)

    client = ClientNode(load_identity(prov_dir, "p0_client"), ca_pub,
                        str(Path(prov_dir) / "ckks_secret"), params,
                        audit_path=audit_path("p0_client"), clock_ms=clock_ms)
    keynode = KeyNode(load_identity(prov_dir, "p1_keynode"), ca_pub,
                      str(Path(prov_dir) / "ckks"), params,
                      audit_path=audit_path("p1_keynode"), clock_ms=clock_ms)
    infer = InferNode(load_identity(prov_dir, "p2_infernode"), ca_pub,
                      str(Path(prov_dir) / "ckks"), params,
                      audit_path=audit_path("p2_infernode"), clock_ms=clock_ms)
    return client, keynode, infer


def establish_all(client: ClientNode, keynode: KeyNode, infer: InferNode,
                  ratchet_mode: str = "messages",
                  ratchet_threshold: int = 64) -> Dict[str, dict]:
    """三条链路的双向认证 + SM2DH 协商（每链路一双线程，join 等待）。"""
    endpoints = {n.node_id: n for n in (client, keynode, infer)}
    threads: List[threading.Thread] = []
    for name in LINK_NAMES:
        a_id, b_id = _NODE_IDS[name]
        na, nb = endpoints[a_id], endpoints[b_id]
        la, lb = LoopbackLink.create_pair(name)
        cha = na.make_channel(name, os.urandom(16),
                              ratchet_mode=ratchet_mode,
                              ratchet_threshold=ratchet_threshold)
        chb = nb.make_channel(name, cha.session.session_id,
                              ratchet_mode=ratchet_mode,
                              ratchet_threshold=ratchet_threshold)
        t1 = threading.Thread(target=cha.establish, args=(la, na.ca_pub, True))
        t2 = threading.Thread(target=chb.establish, args=(lb, nb.ca_pub, False))
        threads += [t1, t2]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    if any(t.is_alive() for t in threads):
        raise TimeoutError("握手线程超时")
    return {n.node_id: n.state_summary()
            for n in (client, keynode, infer)}


def run_inference_round(client: ClientNode, keynode: KeyNode,
                        infer: InferNode, values: List[float],
                        const_mult: int = 2) -> dict:
    """一次数据面往返（真实通道 + 真实转换协议；MPC 中间为 S8 模拟）。

    时序（docs/01 §4.3 缩微）：INFER_REQUEST(P0→P2) → 转换入口
    CONVERT_MASKED_CT(P2→P0) → P0 白名单①解密 → CONVERT_SHARE(P0→P1) →
    [MPC 中间：公开常数乘 const_mult] → RECRYPT_SHARES(P1→P2) → 密文域合成 →
    INFER_RESULT(P2→P0，SM2 签名) → P0 白名单②解密。"""
    request_id = os.urandom(16)
    t0 = time.perf_counter()

    # 1) P0：加密输入 + INFER_REQUEST
    ct_in = client.encrypt_vector(values)
    req = M.InferRequestPayload(request_id=request_id, mode=1,
                                ciphertext=client._secret_ctx.serialize_ct_bytes(ct_in),
                                model_id="p4-toy")
    client.channels["p0-p2"].send_message(
        make_envelope(M.MsgType.INFER_REQUEST, req))

    # 2) P2：转换入口（密文域加 Enc(r)）→ CONVERT_MASKED_CT
    env = infer.channels["p0-p2"].recv_message()
    if env.payload_type != "InferRequestPayload":
        raise ValueError(f"期待 INFER_REQUEST，得 {env.payload_type}")
    req_in = M.InferRequestPayload(**{k: M._decode_json_value(v)
                                      for k, v in env.payload.items()})
    ct = infer.ctx.load_ct_bytes(req_in.ciphertext)
    masked = infer.entry_mask(ct, request_id)
    conv = M.ConvertMaskedCtPayload(request_id=request_id,
                                    masked_ct=infer.ctx.serialize_ct_bytes(masked),
                                    level=masked.level)
    infer.channels["p0-p2"].send_message(
        make_envelope(M.MsgType.CONVERT_MASKED_CT, conv))

    # 3) P0：白名单①掩码解密 → CONVERT_SHARE(P0→P1)
    env = client.channels["p0-p2"].recv_message()
    if env.payload_type != "ConvertMaskedCtPayload":
        raise ValueError(f"期待 CONVERT_MASKED_CT，得 {env.payload_type}")
    conv_in = M.ConvertMaskedCtPayload(**{k: M._decode_json_value(v)
                                          for k, v in env.payload.items()})
    ct_masked = client._secret_ctx.load_ct_bytes(conv_in.masked_ct)
    y1 = client.masked_decrypt(ct_masked, request_id)
    share_msg = M.ConvertSharePayload(request_id=request_id,
                                      share=pack_ints(y1), holder=1)
    client.channels["p0-p1"].send_message(
        make_envelope(M.MsgType.CONVERT_SHARE, share_msg))

    # 4) P2 本地：−r 分享 + MPC 中间常数乘
    p2_share = infer.share_const_mult(infer.entry_p2_share(request_id),
                                      const_mult)

    # 5) P1：收 y₁ → 出口准备（常数乘后的 a₁）→ RECRYPT_SHARES(P1→P2)
    env = keynode.channels["p0-p1"].recv_message()
    if env.payload_type != "ConvertSharePayload":
        raise ValueError(f"期待 CONVERT_SHARE，得 {env.payload_type}")
    share_in = M.ConvertSharePayload(**{k: M._decode_json_value(v)
                                        for k, v in env.payload.items()})
    keynode.take_entry_share(request_id, unpack_ints(share_in.share))
    z, enc_s = keynode.exit_prepare(request_id, const_mult=const_mult)
    rec = M.RecryptSharesPayload(request_id=request_id, enc_mask=enc_s,
                                 masked_share_ints=pack_ints(z))
    keynode.channels["p1-p2"].send_message(
        make_envelope(M.MsgType.RECRYPT_SHARES, rec))

    # 6) P2：密文域合成 fresh 顶层密文 → INFER_RESULT（SM2 签名，CTX_RESULT）
    env = infer.channels["p1-p2"].recv_message()
    if env.payload_type != "RecryptSharesPayload":
        raise ValueError(f"期待 RECRYPT_SHARES，得 {env.payload_type}")
    rec_in = M.RecryptSharesPayload(**{k: M._decode_json_value(v)
                                       for k, v in env.payload.items()})
    ct_fresh = infer.exit_compose(request_id, rec_in.enc_mask,
                                  unpack_ints(rec_in.masked_share_ints),
                                  p2_share)
    payload = dataclasses.asdict(M.InferResultPayload(
        request_id=request_id,
        ciphertext=infer.ctx.serialize_ct_bytes(ct_fresh),
        cipher_meta={"level": ct_fresh.level}))
    payload["sig"] = sm2_sign(infer.identity.static_priv,
                              payload_sign_base(payload), context=CTX_RESULT)
    infer.channels["p0-p2"].send_message(M.MessageEnvelope(
        header=M.CommonHeader(msg_type=int(M.MsgType.INFER_RESULT)),
        payload_type="InferResultPayload", payload=payload))

    # 7) P0：验签（P2 静态公钥）+ 白名单②最终解密
    env = client.channels["p0-p2"].recv_message()
    if env.payload_type != "InferResultPayload":
        raise ValueError(f"期待 INFER_RESULT，得 {env.payload_type}")
    p = dict(env.payload)
    sig = bytes(M._decode_json_value(p.pop("sig")))
    if not sm2_verify(infer.identity.static_pub,
                      payload_sign_base(p), sig, context=CTX_RESULT):
        raise ValueError("INFER_RESULT 签名验证失败")
    result_in = M.InferResultPayload(**{k: M._decode_json_value(v)
                                        for k, v in p.items()})
    ct_out = client._secret_ctx.load_ct_bytes(result_in.ciphertext)
    got = client.final_decrypt(ct_out, request_id)
    n = min(len(values), len(got))
    err = max(abs(g - v * const_mult) for g, v in zip(got[:n], values[:n]))
    return {"request_id": request_id.hex(), "const_mult": const_mult,
            "max_abs_err": err, "values_out": got[:n],
            "wall_ms": (time.perf_counter() - t0) * 1000.0}


def force_ratchet(client: ClientNode, keynode: KeyNode,
                  n_messages: int = 3) -> dict:
    """P0→P1 方向发送心跳触发/执行 ratchet，返回双端纪元观测。

    接收循环跳过 KEY_ROTATE 控制帧（通道内部已完成前向派生），直至本迭代
    的心跳到达——保证轮换帧不滞留、双端纪元一致。"""
    for i in range(n_messages):
        client.channels["p0-p1"].send_message(M.MessageEnvelope(
            header=M.CommonHeader(msg_type=int(M.MsgType.HEARTBEAT)),
            payload_type="HeartbeatPayload", payload={"i": i}))
        while True:
            env = keynode.channels["p0-p1"].recv_message()
            if env.payload_type != "KeyRotatePayload":
                break
    return {"p0_send_epoch": client.channels["p0-p1"].km.current_epoch("send"),
            "p1_recv_epoch": keynode.channels["p0-p1"].km.current_epoch("recv")}


def run_lifecycle(prov_dir: str, params=None, log_dir: Optional[str] = None,
                  ratchet_threshold: int = 64) -> dict:
    """完整生命周期 e2e：建立 → 推理 → ratchet → 销毁 → 审计核验。"""
    if params is None:
        from src.crypto.ckks_ops import PARAMS_MODE_B
        params = PARAMS_MODE_B
    client, keynode, infer = build_local_nodes(prov_dir, params, log_dir)
    report: dict = {"phases": {}}
    report["phases"]["establish"] = establish_all(
        client, keynode, infer, ratchet_threshold=ratchet_threshold)
    report["phases"]["inference"] = run_inference_round(
        client, keynode, infer, [0.5, -1.25, 2.0, 3.75, -0.125, 1.0])
    report["phases"]["ratchet"] = force_ratchet(client, keynode)
    report["phases"]["destroy"] = {
        n.node_id: {k: {"keys_destroyed": v["keys_destroyed"],
                        "all_zeroed": v["all_zeroed"]}
                    for k, v in n.destroy_all("lifecycle_e2e").items()}
        for n in (client, keynode, infer)}
    # 销毁后收发必须拒绝（F6）
    report["phases"]["post_destroy_reject"] = {}
    for n in (client, keynode, infer):
        for name, chan in n.channels.items():
            try:
                chan.send_message(M.MessageEnvelope(
                    header=M.CommonHeader(msg_type=int(M.MsgType.HEARTBEAT)),
                    payload_type="HeartbeatPayload", payload={}))
                report["phases"]["post_destroy_reject"][
                    f"{n.node_id}/{name}"] = "SENT(BUG)"
            except Exception as exc:
                report["phases"]["post_destroy_reject"][
                    f"{n.node_id}/{name}"] = f"rejected: {type(exc).__name__}"
    # 审计核验 + 落盘
    audit_ok, audit_errors = {}, {}
    for n in (client, keynode, infer):
        ok, errs = verify_audit_entries(n.audit.entries)
        audit_ok[n.node_id] = ok
        audit_errors[n.node_id] = errs
        n.flush_audit()
    report["audit_chain_verified"] = audit_ok
    report["audit_errors"] = audit_errors
    report["trajectories"] = {
        n.node_id: {name: [dict(zip(("ts_ms", "from", "to"), h))
                           for h in c.session.history]
                    for name, c in n.channels.items()}
        for n in (client, keynode, infer)}
    report["node_reports"] = {n.node_id: n.export_report()
                              for n in (client, keynode, infer)}
    return report


# ---- 多进程模式（Windows spawn 安全；CKKS 可选——冒烟场景不含密文计算） ----

def _mp_node_worker(node_id: str, prov_dir: str, links_spec: List[dict],
                    node_links: Dict[str, Dict[str, tuple]],
                    commands: "mp.Queue", results: "mp.Queue") -> None:
    """节点进程主循环：加载身份 → 执行命令表（establish/ping/destroy/report）。"""
    from src.nodes.provision import load_ca_pub, load_identity
    ca_pub = load_ca_pub(prov_dir)
    identity = load_identity(prov_dir, node_id)
    node = BaseNode(node_id, identity.role, identity, ca_pub)

    chans = {}
    for spec in links_spec:
        if node_id not in spec["ends"]:
            continue
        q_send, q_recv = node_links[node_id][spec["name"]]
        chan = node.make_channel(spec["name"], bytes(spec["session_id"]))
        chans[spec["name"]] = (chan, MpLink(q_send, q_recv),
                               spec["ends"][0] == node_id)

    report = {"node_id": node_id, "commands": []}
    # 各节点峰值内存口径：本进程 RSS 后台自采样（psutil，20ms）
    import threading
    import psutil
    _proc = psutil.Process()
    _rss_peak = [0]

    def _sample_rss():
        import time
        while not _stop.is_set():
            try:
                _rss_peak[0] = max(_rss_peak[0], _proc.memory_info().rss)
            except Exception:
                pass
            time.sleep(0.02)

    from threading import Event
    _stop = Event()
    threading.Thread(target=_sample_rss, daemon=True).start()
    while True:
        cmd, arg = commands.get(timeout=120)
        if cmd == "establish":
            for name, (chan, q, is_initiator) in chans.items():
                chan.establish(q, ca_pub, initiator=is_initiator,
                               timeout_s=180.0)   # spawn 启动耗时长于单机线程
            report["commands"].append({"cmd": "establish", "ok": True})
        elif cmd == "ping":
            # 每条本端链路各一轮 GCM 往返（对端同命令天然配对）
            for name, (chan, q, _is_init) in chans.items():
                chan.send_message(M.MessageEnvelope(
                    header=M.CommonHeader(msg_type=int(M.MsgType.HEARTBEAT)),
                    payload_type="HeartbeatPayload",
                    payload={"data": f"{arg}-{name}"}))
                env = chan.recv_message(timeout_s=60)
                while env.payload_type == "KeyRotatePayload":
                    env = chan.recv_message(timeout_s=60)
                report["commands"].append(
                    {"cmd": "ping", "link": name, "recv": env.payload})
        elif cmd == "destroy":
            report["destroy"] = {k: v["all_zeroed"]
                                 for k, v in node.destroy_all("mp_smoke").items()}
            report["commands"].append({"cmd": "destroy", "ok": True})
        elif cmd == "report":
            _stop.set()
            merged = node.export_report()
            merged.update(report)
            merged["peak_rss_bytes"] = _rss_peak[0]
            results.put(merged)
            return


def run_multiprocess_smoke(prov_dir: str, out_path: Optional[str] = None) -> dict:
    """三真进程编排冒烟：认证 → 会话 → GCM 消息 → 销毁（不含 CKKS）。"""
    ctx = mp.get_context("spawn")
    node_links: Dict[str, Dict[str, tuple]] = {}
    for name in LINK_NAMES:
        qa, qb = ctx.Queue(), ctx.Queue()      # qa: a→b，qb: b→a
        a_id, b_id = _NODE_IDS[name]
        node_links.setdefault(a_id, {})[name] = (qa, qb)
        node_links.setdefault(b_id, {})[name] = (qb, qa)
    links_spec = [{"name": name, "ends": list(_NODE_IDS[name]),
                   "session_id": list(os.urandom(16))}
                  for name in LINK_NAMES]
    procs, cmd_qs, res_q = [], {}, ctx.Queue()
    for node_id in ("p0_client", "p1_keynode", "p2_infernode"):
        cq = ctx.Queue()
        procs.append(ctx.Process(target=_mp_node_worker,
                                 args=(node_id, prov_dir, links_spec,
                                       node_links, cq, res_q)))
        cmd_qs[node_id] = cq
    for p in procs:
        p.start()
    for cq in cmd_qs.values():
        cq.put(("establish", None))
    for node_id, cq in cmd_qs.items():
        cq.put(("ping", f"hello-from-{node_id}"))
    for cq in cmd_qs.values():
        cq.put(("destroy", None))
    for cq in cmd_qs.values():
        cq.put(("report", None))
    reports = {}
    for _ in range(3):
        r = res_q.get(timeout=180)
        reports[r["node_id"]] = r
    for p in procs:
        p.join(timeout=30)
    out = {"reports": reports,
           "all_processes_ended": not any(p.is_alive() for p in procs)}
    if out_path:
        save_report(out, out_path)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prov-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--toy", action="store_true",
                        help="玩具 CKKS 参数（与 provision --toy 配套）")
    args = parser.parse_args()
    params = None
    if args.toy:
        from src.crypto.ckks_ops import PARAMS_P4_TOY
        params = PARAMS_P4_TOY
    rep = run_lifecycle(args.prov_dir, params=params,
                        log_dir=str(Path(args.prov_dir) / "logs"))
    print(f"audit_chain_verified: {rep['audit_chain_verified']}")
    print(f"inference max_abs_err: "
          f"{rep['phases']['inference']['max_abs_err']:.3e}")
    if args.out:
        save_report(rep, args.out)
