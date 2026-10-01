"""m 方节点编排（P8；D5′ 协议族的通道承载 e2e，零改动 orchestrator.py）。

拓扑通式（1 + m 方；m=3 首发部署）：
    基础链路：p0-p1（y 分发 + 掩码链末跳 P1→P0）、p0-p2（INFER 请求/结果）、
             p1-p2（出口片 P1→合成方）
    掩码链：  p2→p3→…→pm（链路 p(i-1)-pi）→ p1（链路 p1-pm）
    出口星型：pᵢ→p2（i=3 经 p2-p3 反向；i≥4 经 p2-pi）
    m=3 共 5 条链路。

数据面时序（run_inference_round_m，docs/01 §4.3′ 缩微）：
    INFER_REQUEST(P0→P2) → 掩码链逐跳 CONVERT_MASKED_CT（各计算方加自己的
    Enc(rᵢ)）→ P0 白名单①解密 y → CONVERT_SHARE(P0→P1 锚点, holder=1) →
    各方份额本地常数乘 → 出口星型 RECRYPT_SHARES(Pᵢ→P2) → 密文域合成 →
    INFER_RESULT(P2→P0, SM2 签名) → P0 白名单②解密。
同步驱动（每链路 FIFO 严格交替）与原编排同口径；MPC 中间为 S8 模拟。
"""
from __future__ import annotations

import dataclasses
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.crypto.gm_cipher import CTX_RESULT, sm2_sign, sm2_verify
from src.nodes.base import BaseNode, LoopbackLink, save_report
from src.nodes.client import ClientNode
from src.nodes.maskparty import MaskPartyNode
from src.protocol import messages as M
from src.protocol.audit_log import verify_audit_entries
from src.protocol.conversion import pack_ints, unpack_ints
from src.protocol.conversion_m import exit_compose_m
from src.protocol.session import make_envelope, payload_sign_base


def _nid(k: int) -> str:
    """计算方索引 → 节点 id（1=锚点密钥节点、2=合成方推理节点、3+=掩码方）。"""
    return {0: "p0_client", 1: "p1_keynode", 2: "p2_infernode"}.get(
        k, f"p{k}_compute")


def links_for_m(m: int) -> List[Tuple[str, str, str]]:
    """1+m 方链路表：(链路名, 端 a, 端 b)。m=3 → 5 条。"""
    if m < 2:
        raise ValueError("m 必须 ≥ 2")
    links = [("p0-p1", _nid(0), _nid(1)),
             ("p0-p2", _nid(0), _nid(2)),
             ("p1-p2", _nid(1), _nid(2))]
    for k in range(3, m + 1):                    # 掩码链 p2→p3→…→pm
        links.append((f"p{k-1}-p{k}", _nid(k - 1), _nid(k)))
    if m >= 3:                                   # 链末跳 pm→p1
        links.append((f"p1-p{m}", _nid(1), _nid(m)))
    for k in range(4, m + 1):                    # 出口星型 pi→p2（i≥4）
        links.append((f"p2-p{k}", _nid(2), _nid(k)))
    return links


def build_local_nodes_m(prov_dir: str, m: int, params,
                        log_dir: Optional[str] = None
                        ) -> Tuple[ClientNode, List[MaskPartyNode]]:
    """进程内构建 1+m 方节点（含各自审计文件；不建链路）。"""
    from src.nodes.provision import load_ca_pub, load_identity
    ca_pub = load_ca_pub(prov_dir)
    log = Path(log_dir) if log_dir else None
    audit_path = (lambda nid: str(log / f"{nid}.jsonl")) if log else (lambda nid: None)
    from src.nodes.provision import load_identity
    client = ClientNode(load_identity(prov_dir, _nid(0)), ca_pub,
                        str(Path(prov_dir) / "ckks_secret"), params,
                        audit_path=audit_path(_nid(0)))
    parties = []
    for idx in range(1, m + 1):
        parties.append(MaskPartyNode(
            load_identity(prov_dir, _nid(idx)), ca_pub,
            str(Path(prov_dir) / "ckks"), params, index=idx, m=m,
            audit_path=audit_path(_nid(idx))))
    return client, parties


def establish_all_m(client: ClientNode, parties: List[MaskPartyNode],
                    ratchet_mode: str = "messages",
                    ratchet_threshold: int = 64) -> Dict[str, dict]:
    """全部链路的双向认证 + SM2DH 协商（每链路一双线程，join 等待）。"""
    m = len(parties)
    links = links_for_m(m)
    endpoints = {n.node_id: n for n in [client] + parties}
    threads: List[threading.Thread] = []
    for name, a_id, b_id in links:
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
    return {n.node_id: n.state_summary() for n in [client] + parties}


def _decode(env, cls):
    if env.payload_type != cls.__name__:
        raise ValueError(f"期待 {cls.__name__}，得 {env.payload_type}")
    return cls(**{k: M._decode_json_value(v) for k, v in env.payload.items()})


def run_inference_round_m(client: ClientNode, parties: List[MaskPartyNode],
                          values: List[float], const_mult: int = 2) -> dict:
    """一次数据面往返（真实通道 + D5′ m 方转换协议；MPC 中间为 S8 模拟）。

    常数乘后 P0 解密须与 values·const_mult **精确一致**（环算术 +
    生产参数化精确性——协议正确性的端到端断言）。"""
    m = len(parties)
    request_id = os.urandom(16)
    t0 = __import__("time").perf_counter()

    # 1) P0：加密输入 + INFER_REQUEST（发合成方 P2）
    ct_in = client.encrypt_vector(values)
    req = M.InferRequestPayload(request_id=request_id, mode=1,
                                ciphertext=client._secret_ctx.serialize_ct_bytes(ct_in),
                                model_id="p4-toy")
    client.channels["p0-p2"].send_message(
        make_envelope(M.MsgType.INFER_REQUEST, req))

    # 2) 掩码链：P2→P3→…→Pm→P1 逐跳加片（ConvertMaskedCtPayload 复用）
    chain = parties[1:] + [parties[0]]
    name_of = {}
    for name, a, b in links_for_m(m):
        name_of[(a, b)] = name
        name_of[(b, a)] = name                  # 链路双向往返共用链路名
    hop = 0
    for i, node in enumerate(chain):
        if i == 0:
            env = node.channels["p0-p2"].recv_message()
            req_in = _decode(env, M.InferRequestPayload)
            ct_cur = node.ctx.load_ct_bytes(req_in.ciphertext)
        else:
            prev = chain[i - 1]
            lname = name_of[(prev.node_id, node.node_id)]
            env = node.channels[lname].recv_message()
            conv_in = _decode(env, M.ConvertMaskedCtPayload)
            ct_cur = node.ctx.load_ct_bytes(conv_in.masked_ct)
        masked = node.entry_apply_piece(ct_cur, request_id)
        hop += 1
        # 下一跳：链内下家；链末（锚点 P1）→ P0
        if i + 1 < len(chain):
            nxt = chain[i + 1]
            lname = name_of[(node.node_id, nxt.node_id)]
        else:
            lname = "p0-p1"
        conv = M.ConvertMaskedCtPayload(request_id=request_id,
                                        masked_ct=node.ctx.serialize_ct_bytes(masked),
                                        level=masked.level, seq_in_layer=hop)
        node.channels[lname].send_message(
            make_envelope(M.MsgType.CONVERT_MASKED_CT, conv))

    # 3) P0：白名单①掩码解密 → CONVERT_SHARE(P0→锚点 P1, holder=1)
    env = client.channels["p0-p1"].recv_message()
    conv_in = _decode(env, M.ConvertMaskedCtPayload)
    ct_masked = client._secret_ctx.load_ct_bytes(conv_in.masked_ct)
    y = client.masked_decrypt(ct_masked, request_id)
    share_msg = M.ConvertSharePayload(request_id=request_id,
                                      share=pack_ints(y), holder=1)
    client.channels["p0-p1"].send_message(
        make_envelope(M.MsgType.CONVERT_SHARE, share_msg))

    # 4) 各计算方：份额形成（锚点 y−r₁，他方 −rᵢ）+ 本地常数乘
    env = parties[0].channels["p0-p1"].recv_message()
    share_in = _decode(env, M.ConvertSharePayload)
    for node in parties:
        share = node.entry_own_share(
            request_id, y=unpack_ints(share_in.share) if node.is_anchor else None)
        node.set_result_share(request_id, node.share_const_mult(share, const_mult))

    # 5) 出口星型：各方 (zᵢ, Enc(sᵢ)) → 合成方 P2
    composer = parties[1]
    recv_links = ["p1-p2"] + [f"p2-p{k}" for k in range(4, m + 1)]
    if m >= 3:
        recv_links.insert(1, "p2-p3")            # P3 经链路反向送达
    for idx, node in enumerate(parties):
        if node is composer:
            continue
        z, enc_s = node.exit_piece(request_id)
        rec = M.RecryptSharesPayload(request_id=request_id, enc_mask=enc_s,
                                     masked_share_ints=pack_ints(z))
        send_link = {parties[0].node_id: "p1-p2"}.get(
            node.node_id, f"p2-p{node.party_index}")
        node.channels[send_link].send_message(
            make_envelope(M.MsgType.RECRYPT_SHARES, rec))
    pieces = []
    for lname in recv_links:
        env = composer.channels[lname].recv_message()
        rec_in = _decode(env, M.RecryptSharesPayload)
        pieces.append((unpack_ints(rec_in.masked_share_ints), rec_in.enc_mask))
    z2, enc_s2 = composer.exit_piece(request_id)  # 合成方自己的片（同一入参形态）
    pieces.append((z2, enc_s2))

    # 6) 合成方：密文域合成 fresh 顶层密文 → INFER_RESULT（SM2 签名）
    ct_fresh = exit_compose_m(composer.ctx, pieces)
    payload = dataclasses.asdict(M.InferResultPayload(
        request_id=request_id,
        ciphertext=composer.ctx.serialize_ct_bytes(ct_fresh),
        cipher_meta={"level": ct_fresh.level}))
    payload["sig"] = sm2_sign(composer.identity.static_priv,
                              payload_sign_base(payload), context=CTX_RESULT)
    composer.channels["p0-p2"].send_message(M.MessageEnvelope(
        header=M.CommonHeader(msg_type=int(M.MsgType.INFER_RESULT)),
        payload_type="InferResultPayload", payload=payload))

    # 7) P0：验签 + 白名单②最终解密（精确性断言）
    env = client.channels["p0-p2"].recv_message()
    p = dict(env.payload)
    sig = bytes(M._decode_json_value(p.pop("sig")))
    if not sm2_verify(composer.identity.static_pub,
                      payload_sign_base(p), sig, context=CTX_RESULT):
        raise ValueError("INFER_RESULT 签名验证失败")
    result_in = M.InferResultPayload(**{k: M._decode_json_value(v)
                                        for k, v in p.items()})
    ct_out = client._secret_ctx.load_ct_bytes(result_in.ciphertext)
    got = client.final_decrypt(ct_out, request_id)
    n = min(len(values), len(got))
    err = max(abs(g - v * const_mult) for g, v in zip(got[:n], values[:n]))
    return {"request_id": request_id.hex(), "m": m, "const_mult": const_mult,
            "max_abs_err": err, "values_out": got[:n],
            "mask_chain_hops": hop,
            "wall_ms": (__import__("time").perf_counter() - t0) * 1000.0}


def force_ratchet_m(client: ClientNode, anchor: MaskPartyNode,
                    n_messages: int = 3) -> dict:
    """P0→P1 心跳触发/执行 ratchet（与原版同口径，节点换 m 方锚点）。"""
    for i in range(n_messages):
        client.channels["p0-p1"].send_message(M.MessageEnvelope(
            header=M.CommonHeader(msg_type=int(M.MsgType.HEARTBEAT)),
            payload_type="HeartbeatPayload", payload={"i": i}))
        while True:
            env = anchor.channels["p0-p1"].recv_message()
            if env.payload_type != "KeyRotatePayload":
                break
    return {"p0_send_epoch": client.channels["p0-p1"].km.current_epoch("send"),
            "p1_recv_epoch": anchor.channels["p0-p1"].km.current_epoch("recv")}


def run_lifecycle_m(prov_dir: str, m: int = 3, params=None,
                    log_dir: Optional[str] = None,
                    ratchet_threshold: int = 64) -> dict:
    """完整 m 方生命周期 e2e：建立 → 推理 → ratchet → 销毁 → 审计核验。"""
    if params is None:
        from src.crypto.ckks_ops import PARAMS_MODE_B
        params = PARAMS_MODE_B
    client, parties = build_local_nodes_m(prov_dir, m, params, log_dir)
    nodes: List[BaseNode] = [client] + parties
    report: dict = {"phases": {}, "m": m}
    report["phases"]["establish"] = establish_all_m(
        client, parties, ratchet_threshold=ratchet_threshold)
    report["phases"]["inference"] = run_inference_round_m(
        client, parties, [0.5, -1.25, 2.0, 3.75, -0.125, 1.0])
    report["phases"]["ratchet"] = force_ratchet_m(client, parties[0])
    report["phases"]["destroy"] = {
        n.node_id: {k: {"keys_destroyed": v["keys_destroyed"],
                        "all_zeroed": v["all_zeroed"]}
                    for k, v in n.destroy_all("lifecycle_m_e2e").items()}
        for n in nodes}
    report["phases"]["post_destroy_reject"] = {}
    for n in nodes:
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
    audit_ok, audit_errors = {}, {}
    for n in nodes:
        ok, errs = verify_audit_entries(n.audit.entries)
        audit_ok[n.node_id] = ok
        audit_errors[n.node_id] = errs
        n.flush_audit()
    report["audit_chain_verified"] = audit_ok
    report["audit_errors"] = audit_errors
    report["node_reports"] = {n.node_id: n.export_report() for n in nodes}
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prov-dir", required=True)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    from src.crypto.ckks_ops import PARAMS_P4_TOY
    rep = run_lifecycle_m(args.prov_dir, m=args.m, params=PARAMS_P4_TOY,
                          log_dir=str(Path(args.prov_dir) / "logs"))
    print(f"m={rep['m']} audit_chain_verified: {rep['audit_chain_verified']}")
    print(f"inference max_abs_err: "
          f"{rep['phases']['inference']['max_abs_err']:.3e}")
    if args.out:
        save_report(rep, args.out)
