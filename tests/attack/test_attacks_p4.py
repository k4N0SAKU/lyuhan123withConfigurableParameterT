"""P4 攻击预埋测试（正式攻击套件在 P5；此处建立框架与基线拒绝路径）。

覆盖任务书攻击预埋三项 + F5/F6 出口条件：
- 篡改消息字节 → GCM 失败（MAC_INVALID）
- 重放旧消息 → 序列号拒绝（SEQ_REPLAY）
- 伪造证书 → 认证拒绝（CERT_INVALID）
- 销毁后旧密钥解密失败（F6）
- 审计链篡改可检测（F5）
- D5 白名单外解密拒绝
"""
from __future__ import annotations

import os
import threading

import pytest

from src.crypto.gm_cipher import sm2_generate_keypair, sm2_sign
from src.nodes.base import LoopbackLink
from src.protocol.auth import (CTX_CA, AuthError, NodeIdentity, ca_issue_cert,
                               cert_canonical_bytes)
from src.protocol.keylifecycle import KeyManager
from src.protocol.messages import CertExchangePayload, CommonHeader, MessageEnvelope
from src.protocol.session import (ChannelError, SecureChannel, Session,
                                  SessionState)

NOW = 1_700_000_000_000


class _Clock:
    def __init__(self):
        self.ms = NOW

    def __call__(self):
        return self.ms


def _identity(ca_priv, node_id):
    pub, priv = sm2_generate_keypair()
    return NodeIdentity(node_id=node_id, role=0, static_pub=pub,
                        static_priv=priv,
                        cert=ca_issue_cert(ca_priv, node_id, pub, now_ms=NOW))



def _guarded_establish(chan, link, ca_pub, initiator):
    """线程内吞异常（告警纪律——P5-R1：queue.Empty/对端先退出均属预期路径）。"""
    try:
        chan.establish(link, ca_pub, initiator=initiator)
    except Exception:
        pass

def _establish_pair(ca_pub, id_a, id_b, alerts_a=None, alerts_b=None):
    la, lb = LoopbackLink.create_pair()
    sid = os.urandom(16)
    cb = (lambda e, d: alerts_b.append((e, d))) if alerts_b is not None else None
    ca_ = (lambda e, d: alerts_a.append((e, d))) if alerts_a is not None else None
    cha = SecureChannel(Session(sid, 0), id_a, KeyManager(),
                        clock_ms=_Clock(), audit=ca_)
    chb = SecureChannel(Session(sid, 1), id_b, KeyManager(),
                        clock_ms=_Clock(), audit=cb)
    t1 = threading.Thread(target=_guarded_establish,
                          args=(cha, la, ca_pub, True))
    t2 = threading.Thread(target=_guarded_establish,
                          args=(chb, lb, ca_pub, False))
    t1.start(); t2.start(); t1.join(60); t2.join(60)
    return cha, chb, la, lb


def _msg(i=0):
    return MessageEnvelope(header=CommonHeader(msg_type=0x0115),
                           payload_type="HeartbeatPayload", payload={"i": i})


def _capture(link):
    """捕获链路上最近发出的帧（测试注入重放/篡改样本用）。"""
    return bytes(link._send_q.queue[-1])


class _Mitm:
    """链路中间人：在途改写首个目标帧（篡改测试用；fresh seq 才能到达 MAC 检查）。"""

    def __init__(self, inner, offsets):
        self.inner = inner
        self.offsets = list(offsets)

    def send(self, data):
        self.inner.send(data)

    def recv(self, timeout_s: float = 30.0):
        d = self.inner.recv(timeout_s)
        if self.offsets:
            ba = bytearray(d)
            for off in self.offsets:
                ba[off] ^= 0xFF
            d = bytes(ba)
            self.offsets = []
        return d


class TestTamper:
    def test_tamper_ciphertext_byte_mac_invalid(self):
        """F4：篡改密文任一字节 → MAC_INVALID + 审计告警（中间人在途改写）。"""
        ca_pub, ca_priv = sm2_generate_keypair()
        a, b = _identity(ca_priv, "p0"), _identity(ca_priv, "p1")
        alerts = []
        cha, chb, la, lb = _establish_pair(ca_pub, a, b, alerts_b=alerts)
        chb.link = _Mitm(lb, [70])              # 翻转密文段 1 字节
        cha.send_message(_msg(1))
        with pytest.raises(ChannelError) as ei:
            chb.recv_message()
        assert ei.value.error_code == 4         # MAC_INVALID
        assert any(e[0] == "SECURITY_ALERT" and
                   e[1].get("code") == 4 for e in alerts)

    def test_tamper_header_seq_detected(self):
        """头部虽明文，但受 AAD 覆盖——在途改 seq 后 tag 校验失败。"""
        ca_pub, ca_priv = sm2_generate_keypair()
        a, b = _identity(ca_priv, "p0"), _identity(ca_priv, "p1")
        cha, chb, la, lb = _establish_pair(ca_pub, a, b)
        chb.link = _Mitm(lb, [27])              # header 内 seq 字节
        cha.send_message(_msg(1))
        with pytest.raises(ChannelError) as ei:
            chb.recv_message()
        assert ei.value.error_code == 4


class TestReplay:
    def test_replay_rejected_by_seq(self):
        ca_pub, ca_priv = sm2_generate_keypair()
        a, b = _identity(ca_priv, "p0"), _identity(ca_priv, "p1")
        cha, chb, la, lb = _establish_pair(ca_pub, a, b)
        cha.send_message(_msg(1))
        captured = _capture(la)
        chb.recv_message()
        lb._recv_q.put(captured)                # 向 B 的入站队列原样重放
        with pytest.raises(ChannelError) as ei:
            chb.recv_message()
        assert ei.value.error_code == 5         # SEQ_REPLAY

    def test_replay_across_session_rejected(self):
        """跨会话重放：session_id 不同 → nonce 绑定校验失败（MAC_INVALID 路径）。"""
        ca_pub, ca_priv = sm2_generate_keypair()
        a, b = _identity(ca_priv, "p0"), _identity(ca_priv, "p1")
        cha1, chb1, la1, lb1 = _establish_pair(ca_pub, a, b)
        cha2, chb2, la2, lb2 = _establish_pair(ca_pub, a, b)
        cha1.send_message(_msg(1))
        captured = _capture(la1)
        chb1.recv_message()
        lb2._recv_q.put(captured)               # 把会话 1 的帧注入会话 2 入站队列
        with pytest.raises(ChannelError):
            chb2.recv_message()


class TestForgeCert:
    @staticmethod
    def _victim_identity(ca_pub, ca_priv, node_id="p1_keynode"):
        pub, priv = sm2_generate_keypair()
        return NodeIdentity(node_id=node_id, role=1, static_pub=pub,
                            static_priv=priv,
                            cert=ca_issue_cert(ca_priv, node_id, pub, now_ms=NOW))

    def test_forged_ca_sig_rejected_at_handshake(self):
        """F3：伪造证书（自签冒充 CA 签发）在握手期即被拒——会话到不了 ACTIVE。"""
        ca_pub, ca_priv = sm2_generate_keypair()
        evil_pub, evil_priv = sm2_generate_keypair()
        cert = CertExchangePayload(subject_id="p0_client", subject_pub=evil_pub,
                                   not_before_ms=NOW - 60_000,
                                   not_after_ms=NOW + 365 * 24 * 3600 * 1000)
        cert.ca_sig = sm2_sign(evil_priv, cert_canonical_bytes(cert),
                               context=CTX_CA)   # 自签而非 CA 签
        evil = NodeIdentity(node_id="p0_client", role=0, static_pub=evil_pub,
                            static_priv=evil_priv, cert=cert)
        victim = self._victim_identity(ca_pub, ca_priv)
        la, lb = LoopbackLink.create_pair()
        cha = SecureChannel(Session(os.urandom(16), 0), evil, KeyManager(),
                            clock_ms=_Clock())
        chb = SecureChannel(Session(cha.session.session_id, 1), victim,
                            KeyManager(), clock_ms=_Clock())
        def _init():
            try:
                cha.establish(la, ca_pub, initiator=True)
            except Exception:
                pass                       # 发起方在伪造证书被拒后超时/失败属预期

        t1 = threading.Thread(target=_init, daemon=True)

        def _resp():
            try:
                chb.establish(lb, ca_pub, initiator=False)
            except (ChannelError, AuthError, Exception):
                pass                       # 应答方按预期在证书/签名验证处失败
                # （queue.Empty 含于 Exception——对端先退出时接收超时属预期路径）

        t1.start()
        t2 = threading.Thread(target=_resp)
        t2.start()
        t1.join(30)
        t2.join(30)
        assert chb.session.state != SessionState.ACTIVE

    def test_handshake_never_completes_with_rogue_ca(self):
        """F3 出口条件：持另一 CA 根签发证书的对端，握手无法完成。"""
        ca_pub, ca_priv = sm2_generate_keypair()
        a = _identity(ca_priv, "p0")
        rogue_pub, rogue_priv = sm2_generate_keypair()
        rogue_cert = ca_issue_cert(rogue_priv, "p1_keynode", rogue_pub,
                                   now_ms=NOW)   # 另一个"CA"（ rogue_priv）签发
        b = NodeIdentity(node_id="p1_keynode", role=1, static_pub=rogue_pub,
                         static_priv=rogue_priv, cert=rogue_cert)
        la, lb = LoopbackLink.create_pair()
        cha = SecureChannel(Session(os.urandom(16), 0), a, KeyManager(),
                            clock_ms=_Clock())
        chb = SecureChannel(Session(cha.session.session_id, 1), b,
                            KeyManager(), clock_ms=_Clock())
        box = {}

        def _run():
            try:
                cha.establish(la, ca_pub, initiator=True)
                box["a"] = "active"
            except (ChannelError, AuthError) as exc:
                box["a"] = type(exc).__name__

        t1 = threading.Thread(target=_run)
        def _resp2():
            try:
                chb.establish(lb, ca_pub, initiator=False)
            except Exception:
                pass                       # rogue-CA 拒绝/对端先退出均属预期

        t2 = threading.Thread(target=_resp2, daemon=True)
        t1.start(); t2.start(); t1.join(30); t2.join(30)
        assert box.get("a") != "active"          # 发起方验对端证书失败


class TestDestroyAndAudit:
    def test_destroy_then_old_key_unusable_in_system(self):
        """F6：销毁后系统持有的旧密钥不可再用于解密/收发。

        Python 语义申报：测试进程内的 bytes 拷贝无法回收覆写——断言对象是
        系统侧密钥材料（registry 清零 + channel_key 拒绝 + 通道拒绝）。"""
        ca_pub, ca_priv = sm2_generate_keypair()
        a, b = _identity(ca_priv, "p0"), _identity(ca_priv, "p1")
        cha, chb, la, lb = _establish_pair(ca_pub, a, b)
        from src.protocol.keylifecycle import KeyLifecycleError
        from src.crypto.gm_cipher import sm4_gcm_encrypt
        sm4_gcm_encrypt(cha.km.channel_key("send"), b"x", nonce=b"\x03" * 12)
        cha.destroy("test")
        with pytest.raises(Exception):
            cha.send_message(_msg(9))            # 会话期拒绝
        assert all(v == 0 for v in cha.km._registry["chan_send_e0"])
        with pytest.raises(KeyLifecycleError):
            cha.km.channel_key("send")           # 系统侧解密路径已失效

    def test_audit_tamper_detectable(self, tmp_path):
        from src.protocol.audit_log import AuditLog, verify_audit_entries
        log = AuditLog(actor="P1", clock_ms=lambda: 1)
        log.append("P1", "SESSION_ACTIVE", {"peer": "p0"})
        log.append("P1", "KEYS_DESTROYED", {"reason": "e2e"})
        path = tmp_path / "p1.jsonl"
        path.write_text(log.export_json(), encoding="utf-8")
        # 攻击者篡改落盘日志一行
        lines = path.read_text(encoding="utf-8").splitlines()
        d = __import__("json").loads(lines[1])
        d["detail"]["reason"] = "innocent"
        lines[1] = __import__("json").dumps(d)
        path.write_text("\n".join(lines), encoding="utf-8")
        loaded = AuditLog.from_json(path.read_text(encoding="utf-8"))
        ok, errs = verify_audit_entries(loaded.entries)
        assert not ok and any("篡改" in e for e in errs)
