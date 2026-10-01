"""协议消息序列化测试（P4；S7 锚点：规范序列化 + 字节保真往返）。"""
from __future__ import annotations

import dataclasses
import os

import pytest

from src.protocol import messages as M


def _sample_payloads():
    return [
        M.HelloPayload(node_id="p0_client", role=0, nonce=os.urandom(32)),
        M.CertExchangePayload(subject_id="p1", subject_pub=os.urandom(65),
                              not_before_ms=1, not_after_ms=2,
                              ca_sig=os.urandom(64)),
        M.AuthChallengePayload(challenge_nonce=os.urandom(32), expires_ms=99),
        M.AuthResponsePayload(challenge_nonce=os.urandom(32),
                              responder_nonce=os.urandom(32), sig=os.urandom(64)),
        M.KeyNegotiatePayload(ephemeral_pub=os.urandom(65), epoch=3,
                              sig=os.urandom(64)),
        M.KeyRotatePayload(epoch_new=4, transcript_hash=os.urandom(32),
                           sig=os.urandom(64)),
        M.ConvertMaskedCtPayload(request_id=os.urandom(16),
                                 masked_ct=os.urandom(37), level=2),
        M.ConvertSharePayload(request_id=os.urandom(16), share=os.urandom(24),
                              holder=1),
        M.RecryptSharesPayload(request_id=os.urandom(16), enc_mask=os.urandom(41),
                               masked_share_ints=os.urandom(32), level_target=0),
        M.ErrorPayload(error_code=5, failed_seq=7, detail="seq 回退"),
    ]


class TestEnvelopeSerialization:
    @pytest.mark.parametrize("payload", _sample_payloads(),
                             ids=lambda p: type(p).__name__)
    def test_roundtrip_byte_fidelity(self, payload):
        env = M.MessageEnvelope(header=M.CommonHeader(
            msg_type=int(M.MsgType.HELLO), session_id=os.urandom(16), seq=9,
            timestamp_ms=1_700_000_000_000), payload_type=type(payload).__name__,
            payload=dataclasses.asdict(payload), auth_value=os.urandom(64))
        env.header.payload_len = len(M.payload_to_json(env.payload))
        env2 = M.deserialize_envelope(M.serialize_envelope(env))
        assert env2.header.seq == 9
        assert env2.header.session_id == env.header.session_id
        assert env2.payload_type == type(payload).__name__
        for k, v in env2.payload.items():
            expect = getattr(payload, k)
            got = M._decode_json_value(v)
            if isinstance(expect, tuple):        # JSON 无元组语义 → list 往返
                got = tuple(got)
            assert got == expect, k
        assert env2.auth_value == env.auth_value

    def test_canonical_is_deterministic(self):
        p = M.HelloPayload(node_id="x", nonce=b"\x01" * 32)
        e1 = M.MessageEnvelope(payload_type="HelloPayload",
                               payload=dataclasses.asdict(p))
        e2 = M.MessageEnvelope(payload_type="HelloPayload",
                               payload=dataclasses.asdict(p))
        assert M.canonical_bytes(e1) == M.canonical_bytes(e2)

    def test_hex_tag_codec_nested(self):
        payload = {"a": b"\x00\xff", "list": [b"\x01", 2, "s"],
                   "nested": {"deep": b"\x02"}}
        raw = M.payload_to_json(payload)
        assert M.payload_from_json(raw) == payload

    def test_deserialize_rejects_truncation(self):
        env = M.MessageEnvelope(payload_type="ErrorPayload",
                                payload={"error_code": 1})
        env.header.payload_len = len(M.payload_to_json(env.payload))
        data = M.serialize_envelope(env)
        with pytest.raises(M.SerializationError):
            M.deserialize_envelope(data[:-3])
        with pytest.raises(M.SerializationError):
            M.deserialize_envelope(data[:30])

    def test_registry_covers_new_recrypt_shape(self):
        rec = M.RecryptSharesPayload()
        assert hasattr(rec, "enc_mask") and hasattr(rec, "masked_share_ints")
        assert rec.enc_mask == b"" and rec.level_target == 0
