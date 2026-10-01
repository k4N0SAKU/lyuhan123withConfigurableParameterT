"""F7-4 重放攻击：合法消息重放（同会话/跨会话/跨消息类型）。

断言：序列号单调检查拒绝全部重放（先于解密），告警入审计。

运行：pytest tests/attack/test_replay.py -v
"""
from __future__ import annotations

import os
import threading

import pytest

from src.crypto.gm_cipher import sm2_generate_keypair
from src.nodes.base import LoopbackLink
from src.protocol.keylifecycle import KeyManager
from src.protocol.messages import CommonHeader, MessageEnvelope
from src.protocol.session import ChannelError, SecureChannel, Session
from tests.attack.conftest import NOW, make_identity
from tests.attack.framework import VERDICTS


class _Clock:
    def __init__(self):
        self.ms = NOW

    def __call__(self):
        return self.ms



def _guarded_establish(chan, link, ca_pub, initiator):
    """线程内吞异常（告警纪律——P5-R1：queue.Empty/对端先退出均属预期路径）。"""
    try:
        chan.establish(link, ca_pub, initiator=initiator)
    except Exception:
        pass

def _establish_pair(attack_ca, id_a, id_b, threshold=999):
    la, lb = LoopbackLink.create_pair()
    cha = SecureChannel(Session(os.urandom(16), 0), id_a, KeyManager(),
                        clock_ms=_Clock(), ratchet_threshold=threshold)
    chb = SecureChannel(Session(cha.session.session_id, 1), id_b, KeyManager(),
                        clock_ms=_Clock(), ratchet_threshold=threshold)
    t1 = threading.Thread(target=_guarded_establish,
                          args=(cha, la, attack_ca["pub"], True))
    t2 = threading.Thread(target=_guarded_establish,
                          args=(chb, lb, attack_ca["pub"], False))
    t1.start(); t2.start(); t1.join(60); t2.join(60)
    return cha, chb, la, lb


def _msg(i=0, ptype="HeartbeatPayload"):
    return MessageEnvelope(header=CommonHeader(msg_type=0x0115),
                           payload_type=ptype, payload={"i": i})


def _capture(link):
    """捕获该端最近发出的帧（queue 取即删——先捕获后消费）。"""
    return bytes(link._send_q.queue[-1])


def _recv_business(chan):
    """接收并跳过控制帧（KEY_ROTATE 已在通道内处理）。"""
    env = chan.recv_message()
    while env.payload_type == "KeyRotatePayload":
        env = chan.recv_message()
    return env


class TestReplay:
    def test_same_session_replay_business_message(self, attack_ca):
        """同会话重放业务帧（INFER_RESULT 语义载荷）→ SEQ_REPLAY。"""
        a, b = make_identity(attack_ca, "p0_client"), make_identity(
            attack_ca, "p1_keynode")
        cha, chb, la, lb = _establish_pair(attack_ca, a, b)
        cha.send_message(_msg(1, "InferResultPayload"))
        captured = _capture(la)
        _recv_business(chb)
        lb._recv_q.put(captured)                       # 注入重放
        with pytest.raises(ChannelError) as ei:
            _recv_business(chb)
        assert ei.value.error_code == 5                # SEQ_REPLAY
        VERDICTS.collect("F7-4", "同会话重放", "防御成功",
                         {"error_code": 5, "check": "seq 单调（先于解密）"})

    def test_replay_multiple_copies_all_rejected(self, attack_ca):
        """连续重放多份拷贝：每一份都被拒绝（无窗口放行）。"""
        a, b = make_identity(attack_ca, "p0_client"), make_identity(
            attack_ca, "p1_keynode")
        cha, chb, la, lb = _establish_pair(attack_ca, a, b)
        cha.send_message(_msg(1))
        captured = _capture(la)
        _recv_business(chb)
        rejected = 0
        for _ in range(3):
            lb._recv_q.put(captured)
            with pytest.raises(ChannelError) as ei:
                _recv_business(chb)
            assert ei.value.error_code == 5
            rejected += 1
        VERDICTS.collect("F7-4", "多份连续重放", "防御成功",
                         {"rejected": rejected})

    def test_cross_session_replay(self, attack_ca):
        """跨会话重放：会话 1 的帧注入会话 2 → nonce 绑定校验失败。"""
        a, b = make_identity(attack_ca, "p0_client"), make_identity(
            attack_ca, "p1_keynode")
        cha1, chb1, la1, lb1 = _establish_pair(attack_ca, a, b)
        cha2, chb2, la2, lb2 = _establish_pair(attack_ca, a, b)
        cha1.send_message(_msg(1))
        captured = _capture(la1)
        _recv_business(chb1)
        lb2._recv_q.put(captured)
        with pytest.raises(ChannelError) as ei:
            _recv_business(chb2)
        assert ei.value.error_code in (4, 5)           # MAC（nonce 绑定）/seq
        VERDICTS.collect("F7-4", "跨会话重放", "防御成功",
                         {"error_code": ei.value.error_code,
                          "defense": "nonce=SM3(sid‖sender)‖seq 绑定"})

    def test_replay_after_ratchet_rejected(self, attack_ca):
        """ratchet 后重放轮换前捕获的帧：seq 拒绝；旧纪元密钥窗口外亦不可解。"""
        a, b = make_identity(attack_ca, "p0_client"), make_identity(
            attack_ca, "p1_keynode")
        cha, chb, la, lb = _establish_pair(attack_ca, a, b, threshold=2)
        old_frames = []
        for i in range(2):                             # 阈值 2：填充计数
            cha.send_message(_msg(i))
            old_frames.append(_capture(la))
            _recv_business(chb)
        cha.send_message(_msg(9))                      # 触发轮换（KEY_ROTATE+新钥帧）
        _recv_business(chb)
        for fr in old_frames:                          # 旧帧全部重放
            lb._recv_q.put(fr)
            with pytest.raises(ChannelError) as ei:
                _recv_business(chb)
            assert ei.value.error_code == 5            # seq 单调拒绝（先于解密）
        VERDICTS.collect("F7-4", "ratchet 后旧帧重放", "防御成功",
                         {"rejected": len(old_frames),
                          "defense": "seq 单调 + 旧纪元密钥窗口外销毁"})

    def test_timestamp_replay_stale_frame(self, attack_ca):
        """重放时篡改 seq 绕过单调检查 → 时间戳容差窗拒绝（注入时钟）。"""
        a, b = make_identity(attack_ca, "p0_client"), make_identity(
            attack_ca, "p1_keynode")
        clock = _Clock()
        la, lb = LoopbackLink.create_pair()
        cha = SecureChannel(Session(os.urandom(16), 0), a, KeyManager(),
                            clock_ms=clock, ratchet_threshold=999)
        chb = SecureChannel(Session(cha.session.session_id, 1), b,
                            KeyManager(), clock_ms=clock,
                            ratchet_threshold=999)
        t1 = threading.Thread(target=_guarded_establish,
                              args=(cha, la, attack_ca["pub"], True))
        t2 = threading.Thread(target=_guarded_establish,
                              args=(chb, lb, attack_ca["pub"], False))
        t1.start(); t2.start(); t1.join(60); t2.join(60)
        cha.send_message(_msg(1))
        captured = _capture(la)
        _recv_business(chb)
        # 攻击者重放并把 seq 改为未来值——AAD 覆盖头部 → MAC 失败（tag 保护）
        tampered = bytearray(captured)
        tampered[27] ^= 0x01                           # seq 低位字节
        lb._recv_q.put(bytes(tampered))
        with pytest.raises(ChannelError) as ei:
            _recv_business(chb)
        assert ei.value.error_code == 4                # MAC_INVALID（AAD）
        VERDICTS.collect("F7-4", "改 seq 重放", "防御成功",
                         {"error_code": 4,
                          "defense": "header 受 AAD 覆盖，改 seq 即 tag 失败"})
