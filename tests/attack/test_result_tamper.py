"""F7-3 结果篡改攻击：篡改输出密文/签名/元数据，断言客户端验签拒绝。

攻击者模型：恶意 P2（持本端通道密钥——可发送任意合法通道帧，但无 P0 私钥
与 CA 越权）。防御：INFER_RESULT 载荷级 SM2 签名（CTX_RESULT）+ 通道 AEAD。

运行：pytest tests/attack/test_result_tamper.py -v
"""
from __future__ import annotations

import dataclasses
import os
import threading

import pytest

from src.crypto.gm_cipher import (CTX_RESULT, sm2_sign, sm2_verify)
from src.nodes.base import LoopbackLink
from src.protocol import messages as M
from src.protocol.keylifecycle import KeyManager
from src.protocol.session import (ChannelError, SecureChannel, Session,
                                  make_envelope, payload_sign_base)
from tests.attack.conftest import NOW, make_identity
from tests.attack.framework import VERDICTS


class _Clock:
    def __init__(self):
        self.ms = NOW

    def __call__(self):
        return self.ms


@pytest.fixture(scope="module")
def active_session(attack_ca, tmp_path_factory):
    """已建立的 P0↔P2 会话（玩具 CKKS 公钥材料签发真实结果帧）。"""
    prov = str(tmp_path_factory.mktemp("tamper_prov"))
    from src.nodes.provision import provision_demo
    from src.crypto.ckks_ops import PARAMS_P4_TOY
    provision_demo(prov, ckks_params=PARAMS_P4_TOY)
    from src.nodes.orchestrator import build_local_nodes, establish_all
    client, keynode, infer = build_local_nodes(prov, PARAMS_P4_TOY,
                                               log_dir=str(prov) + "/logs")
    establish_all(client, keynode, infer, ratchet_threshold=999)
    return client, infer


def _honest_result(infer, values):
    """P2 真实签发的 INFER_RESULT 载荷（dict，含 sig）。"""
    ct = infer.ctx.encrypt_vector(values)
    payload = dataclasses.asdict(M.InferResultPayload(
        request_id=os.urandom(16),
        ciphertext=infer.ctx.serialize_ct_bytes(ct),
        cipher_meta={"level": 0}))
    payload["sig"] = sm2_sign(infer.identity.static_priv,
                              payload_sign_base(payload), context=CTX_RESULT)
    return payload


def _send_result(infer, payload):
    """结果帧从 P2 侧发出（INFER_RESULT 方向 P2→P0）。"""
    infer.channels["p0-p2"].send_message(M.MessageEnvelope(
        header=M.CommonHeader(msg_type=int(M.MsgType.INFER_RESULT)),
        payload_type="InferResultPayload", payload=payload))


def _client_receive_and_verify(client, infer_identity):
    """P0 侧接收+验签（与 orchestrator.run_inference_round 同一逻辑路径）。"""
    env = client.channels["p0-p2"].recv_message()
    assert env.payload_type == "InferResultPayload"
    p = dict(env.payload)
    sig = bytes(M._decode_json_value(p.pop("sig")))
    ok = sm2_verify(infer_identity.static_pub, payload_sign_base(p), sig,
                    context=CTX_RESULT)
    return ok, p


class TestResultTamper:
    def test_tampered_ciphertext_rejected(self, active_session):
        """篡改输出密文（载荷内）→ 验签失败 → 客户端拒绝。"""
        client, infer = active_session
        payload = _honest_result(infer, [1.0, 2.0])
        ct = bytearray(payload["ciphertext"])
        ct[len(ct) // 2] ^= 0xFF                        # 翻转密文中部
        payload["ciphertext"] = bytes(ct)
        _send_result(infer, payload)
        ok, _ = _client_receive_and_verify(client, infer.identity)
        assert ok is False
        VERDICTS.collect("F7-3", "篡改输出密文", "防御成功",
                         {"sig_verify": False, "client_action": "拒绝"})

    def test_tampered_signature_rejected(self, active_session):
        """伪造/篡改签名字段 → 验签失败。"""
        client, infer = active_session
        payload = _honest_result(infer, [1.0, 2.0])
        sig = bytearray(payload["sig"])
        sig[0] ^= 0x01
        payload["sig"] = bytes(sig)
        _send_result(infer, payload)
        ok, _ = _client_receive_and_verify(client, infer.identity)
        assert ok is False
        VERDICTS.collect("F7-3", "篡改签名", "防御成功",
                         {"sig_verify": False})

    def test_tampered_meta_rejected(self, active_session):
        """篡改密文元数据（level 等签名字段）→ 验签失败。"""
        client, infer = active_session
        payload = _honest_result(infer, [1.0, 2.0])
        payload["cipher_meta"] = {"level": 7}
        _send_result(infer, payload)
        ok, _ = _client_receive_and_verify(client, infer.identity)
        assert ok is False
        VERDICTS.collect("F7-3", "篡改元数据", "防御成功",
                         {"sig_verify": False})

    def test_forged_result_by_p0_keyholder_rejected(self, active_session):
        """攻击者无 P2 私钥：自行构造结果帧（任意密文+伪造签名）→ 拒绝。"""
        client, infer = active_session
        forged = dataclasses.asdict(M.InferResultPayload(
            request_id=os.urandom(16), ciphertext=os.urandom(512),
            cipher_meta={"level": 0}))
        forged["sig"] = os.urandom(64)                  # 伪造签名
        _send_result(infer, forged)
        ok, _ = _client_receive_and_verify(client, infer.identity)
        assert ok is False
        VERDICTS.collect("F7-3", "无钥伪造整帧", "防御成功",
                         {"sig_verify": False})

    def test_honest_result_accepted(self, active_session):
        """对照：诚实 P2 的结果帧通过验签（防御不误伤）。"""
        client, infer = active_session
        payload = _honest_result(infer, [3.0, 4.0])
        _send_result(infer, payload)
        ok, p = _client_receive_and_verify(client, infer.identity)
        assert ok is True
        VERDICTS.collect("F7-3", "诚实结果对照", "防御成功",
                         {"sig_verify": True})
