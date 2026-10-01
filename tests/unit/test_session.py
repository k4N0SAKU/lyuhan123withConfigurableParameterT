"""会话与通道测试（P4；SM2DH 一致性 / GCM 通道 / 防重放 / ratchet）。"""
from __future__ import annotations

import os
import threading

import pytest

from src.common.netmeter import NetworkMeter
from src.crypto.gm_cipher import sm2_generate_keypair
from src.nodes.base import LoopbackLink
from src.protocol.auth import NodeIdentity, ca_issue_cert
from src.protocol.keylifecycle import KeyManager
from src.protocol.messages import MessageEnvelope, CommonHeader
from src.protocol.session import (SecureChannel, Session, SessionState,
                                  _gen_ephemeral, confirm_tag,
                                  sm2_key_agreement)

NOW = 1_700_000_000_000
SID = b"\x33" * 16


class _Clock:
    def __init__(self, start=NOW):
        self.ms = start

    def __call__(self):
        return self.ms


@pytest.fixture(scope="module")
def ca():
    pub, priv = sm2_generate_keypair()
    return {"pub": pub, "priv": priv}


def _identity(ca, node_id):
    pub, priv = sm2_generate_keypair()
    return NodeIdentity(node_id=node_id, role=0,
                        static_pub=pub, static_priv=priv,
                        cert=ca_issue_cert(ca["priv"], node_id, pub, now_ms=NOW))


class _Recording:
    """链路包装：记录本端发出的全部帧（测试捕获重放/篡改样本用）。"""

    def __init__(self, inner):
        self.inner = inner
        self.sent = []

    def send(self, data):
        self.sent.append(data)
        self.inner.send(data)

    def recv(self, timeout_s: float = 30.0):
        return self.inner.recv(timeout_s)


def _pair(ca, id_a, id_b, clock=None, **kw):
    la, lb = LoopbackLink.create_pair()
    session_id = os.urandom(16)
    cha = SecureChannel(Session(session_id, 0), id_a, KeyManager(),
                        clock_ms=clock or _Clock(), **kw)
    chb = SecureChannel(Session(session_id, 1), id_b, KeyManager(),
                        clock_ms=clock or _Clock(), **kw)
    return cha, chb, la, lb


def _establish(cha, chb, la, lb, ca):
    t1 = threading.Thread(target=cha.establish, args=(la, ca["pub"], True))
    t2 = threading.Thread(target=chb.establish, args=(lb, ca["pub"], False))
    t1.start(); t2.start(); t1.join(60); t2.join(60)


def _ping(i=0):
    return MessageEnvelope(header=CommonHeader(msg_type=0x0115),
                           payload_type="HeartbeatPayload", payload={"i": i})


class TestSM2DH:
    def test_both_sides_agree(self):
        from src.crypto.gm_cipher import sm2_generate_keypair as kp
        pub_a, priv_a = kp()
        pub_b, priv_b = kp()
        ra, Ra = _gen_ephemeral()
        rb, Rb = _gen_ephemeral()
        s_a = sm2_key_agreement(priv_a, pub_b, ra, Ra, Rb, pub_a, pub_b)
        s_b = sm2_key_agreement(priv_b, pub_a, rb, Rb, Ra, pub_a, pub_b)
        assert s_a == s_b and len(s_a) == 32
        assert confirm_tag(s_a) == confirm_tag(s_b)

    def test_wrong_peer_detected_by_confirm_tag(self):
        """MitM 替换静态公钥 → 确认标签不一致（协商不可静默降级）。"""
        from src.crypto.gm_cipher import sm2_generate_keypair as kp
        pub_a, priv_a = kp()
        pub_b, priv_b = kp()
        pub_e, _ = kp()
        ra, Ra = _gen_ephemeral()
        rb, Rb = _gen_ephemeral()
        s_a = sm2_key_agreement(priv_a, pub_e, ra, Ra, Rb, pub_a, pub_e)  # A 被骗
        s_b = sm2_key_agreement(priv_b, pub_a, rb, Rb, Ra, pub_a, pub_b)
        assert confirm_tag(s_a) != confirm_tag(s_b)




class _Mitm:
    """链路中间人：在途改写首个目标帧（篡改测试用；fresh seq 才能到达 MAC 检查）。"""

    def __init__(self, inner, offsets):
        self.inner = inner
        self.offsets = offsets          # 待翻转字节位集合（首帧生效一次）

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

class TestChannel:
    def test_establish_and_roundtrip(self, ca):
        a, b = _identity(ca, "p0_client"), _identity(ca, "p1_keynode")
        cha, chb, la, lb = _pair(ca, a, b)
        _establish(cha, chb, la, lb, ca)
        assert cha.session.state == SessionState.ACTIVE
        assert chb.km.audit_state()["directions"]["send"]["epoch"] == 0
        seq0 = cha.current_seq("send")
        cha.send_message(_ping(1))
        env = chb.recv_message()
        assert env.payload == {"i": 1}
        # seq 全程单调连续（握手帧计入）；recv 侧仅记会话期帧
        assert cha.current_seq("send") == seq0 + 1
        assert chb.current_seq("recv") == seq0   # 首个会话帧 seq = 发送侧握手后基线
        chb.send_message(_ping(2))
        env2 = cha.recv_message()
        assert env2.payload == {"i": 2}
        # 双端 transcript 链一致（同序处理同批帧）
        assert cha.session.transcript_hash() == chb.session.transcript_hash()

    def test_meter_counts_full_frames(self, ca):
        """F9：字节计数含协议头（整帧口径）。"""
        a, b = _identity(ca, "p0"), _identity(ca, "p1")
        cha, chb, la, lb = _pair(ca, a, b)
        meter_a, meter_b = NetworkMeter(), NetworkMeter()
        cha.meter = meter_a
        chb.meter = meter_b
        _establish(cha, chb, la, lb, ca)
        n0 = meter_a.bytes_sent
        n = cha.send_message(_ping(1))
        assert meter_a.bytes_sent - n0 >= n          # 整帧口径（含协议头）
        chb.recv_message()                           # ping 到达后再对账
        assert meter_b.messages_recv == cha.meter.messages_sent

    def test_replayed_frame_rejected(self, ca):
        """F6 攻击预埋：重放旧帧 → SEQ_REPLAY。"""
        a, b = _identity(ca, "p0"), _identity(ca, "p1")
        cha, chb, la, lb = _pair(ca, a, b)
        _establish(cha, chb, la, lb, ca)
        rec = _Recording(la)
        cha.link = rec
        cha.send_message(_ping(1))
        chb.recv_message()
        lb._recv_q.put(rec.sent[0])            # 攻击者向 B 的入站队列注入重放
        from src.protocol.session import ChannelError
        with pytest.raises(ChannelError) as ei:
            chb.recv_message()
        assert ei.value.error_code == 5        # SEQ_REPLAY

    def test_tampered_frame_rejected(self, ca):
        """F4 攻击预埋：篡改密文任一字节 → MAC_INVALID + 审计告警。"""
        a, b = _identity(ca, "p0"), _identity(ca, "p1")
        alerts = []
        cha, chb, la, lb = _pair(ca, a, b, )
        chb._audit = lambda e, d: alerts.append((e, d))
        _establish(cha, chb, la, lb, ca)
        rec = _Recording(la)
        cha.link = rec
        chb.link = _Mitm(lb, [-20])            # 翻转密文中部一字节
        cha.send_message(_ping(1))             # 帧在途被 _Mitm 篡改
        from src.protocol.session import ChannelError
        with pytest.raises(ChannelError) as ei:
            chb.recv_message()
        assert ei.value.error_code == 4        # MAC_INVALID
        assert any(e[0] == "SECURITY_ALERT" for e in alerts)

    def test_stale_timestamp_rejected(self, ca):
        """F6：时间戳超容差窗 → TS_EXPIRED（注入时钟，陷阱 1）。"""
        clock = _Clock()
        a, b = _identity(ca, "p0"), _identity(ca, "p1")
        cha, chb, la, lb = _pair(ca, a, b, clock=clock)
        _establish(cha, chb, la, lb, ca)
        cha.send_message(_ping(1))
        clock.ms += 120_000                    # 越过 30s 容差窗
        from src.protocol.session import ChannelError
        with pytest.raises(ChannelError) as ei:
            chb.recv_message()
        assert ei.value.error_code == 6        # TS_EXPIRED


class TestRatchet:
    def test_message_count_trigger(self, ca):
        """ratchet 触发（消息数口径）：阈值 3，第 4 条业务消息前完成轮换。"""
        a, b = _identity(ca, "p0"), _identity(ca, "p1")
        cha, chb, la, lb = _pair(ca, a, b, ratchet_mode="messages",
                                 ratchet_threshold=3)
        _establish(cha, chb, la, lb, ca)
        for i in range(4):
            cha.send_message(_ping(i))
            env = chb.recv_message()
            while env.payload_type == "KeyRotatePayload":   # 控制帧已被通道内处理
                env = chb.recv_message()
            assert env.payload == {"i": i}
        assert cha.km.current_epoch("send") == 1
        assert chb.km.current_epoch("recv") == 1
        # 轮换后 transcript 链仍双端一致
        assert cha.session.transcript_hash() == chb.session.transcript_hash()

    def test_bytes_trigger(self, ca):
        a, b = _identity(ca, "p0"), _identity(ca, "p1")
        cha, chb, la, lb = _pair(ca, a, b, ratchet_mode="bytes",
                                 ratchet_threshold=300)
        _establish(cha, chb, la, lb, ca)
        for i in range(4):
            cha.send_message(MessageEnvelope(
                header=CommonHeader(msg_type=0x0115),
                payload_type="HeartbeatPayload",
                payload={"i": i, "pad": "x" * 80}))
            chb.recv_message()
        assert cha.km.current_epoch("send") >= 1
        assert chb.km.current_epoch("recv") >= 1

    def test_state_transitions_guarded(self):
        s = Session(SID, role=0)
        from src.protocol.session import StateError
        with pytest.raises(StateError):
            s.transition(SessionState.ACTIVE)      # IDLE→ACTIVE 非法
        s.transition(SessionState.NEGOTIATED)
        s.transition(SessionState.ACTIVE)
        s.transition(SessionState.ROTATING)
        s.transition(SessionState.ACTIVE)
        s.transition(SessionState.DESTROYED)
        with pytest.raises(StateError):
            s.transition(SessionState.ACTIVE)      # 终态不可逆
        assert len(s.history) == 5
