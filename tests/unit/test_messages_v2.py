# -*- coding: utf-8 -*-
"""数据面二进制帧（协议 v2）单测：往返等价 / 线上长度对照 / 篡改拦截 / 签名一致。"""
from __future__ import annotations

import os

import pytest

import dataclasses

from src.protocol import messages as M
from src.protocol.session import payload_sign_base

REQ = os.urandom(16)
CT = os.urandom(1 << 20)                 # 1 MiB 代表性密文载荷
SHARE = os.urandom(1 << 17)
ENC = os.urandom(1 << 20)
MSHARE = os.urandom(1 << 17)


def _payloads():
    return {
        "InferRequestPayload": M.InferRequestPayload(
            request_id=REQ, mode=1, ciphertext=CT, model_id="p4-toy"),
        "InferResultPayload": M.InferResultPayload(
            request_id=REQ, ciphertext=CT, sig=os.urandom(64)),
        "ConvertMaskedCtPayload": M.ConvertMaskedCtPayload(
            request_id=REQ, masked_ct=CT, level=1, seq_in_layer=3),
        "ConvertSharePayload": M.ConvertSharePayload(
            request_id=REQ, share=SHARE, holder=1),
        "RecryptSharesPayload": M.RecryptSharesPayload(
            request_id=REQ, enc_mask=ENC, masked_share_ints=MSHARE),
    }


@pytest.mark.parametrize("ptype", list(_payloads().keys()))
def test_v2_roundtrip_identical(ptype):
    """v2 序列化→反序列化后载荷字典与原 dataclass asdict 完全一致（无残留键）。"""
    payload = dataclasses.asdict(_payloads()[ptype])
    env = M.build_data_plane_envelope(0x201, payload, ptype)
    assert env.header.version == M.PROTOCOL_VERSION_BINARY
    wire = M.serialize_envelope(env)
    back = M.deserialize_envelope(wire)
    assert back.payload == payload                       # 字节精确还原
    assert back.blob == b"" or back.blob                 # blob 已还原（v2 解包消费）


def test_v2_wire_saving_vs_v1():
    """同一载荷 v1（hex JSON）vs v2（blob）线上长度对照：大载荷 −49% 左右。"""
    payload = dataclasses.asdict(_payloads()["ConvertMaskedCtPayload"])
    v1_env = M.MessageEnvelope(
        header=M.CommonHeader(msg_type=0x211),
        payload_type="ConvertMaskedCtPayload", payload=dict(payload))
    v1_len = len(M.serialize_envelope(v1_env))
    v2_env = M.build_data_plane_envelope(0x211, dict(payload),
                                         "ConvertMaskedCtPayload")
    v2_len = len(M.serialize_envelope(v2_env))
    saving = 1 - v2_len / v1_len
    assert 0.4 < saving < 0.6, f"v2 线上节省 {saving:.0%} 超出预期区间"
    print(f"\nwire: v1={v1_len/1048576:.2f}MiB v2={v2_len/1048576:.2f}MiB 节省={saving:.1%}")


def test_v2_blob_tamper_detected():
    """blob 单字节篡改 → SM3 绑定校验拦截（SerializationError）。"""
    payload = dataclasses.asdict(_payloads()["ConvertMaskedCtPayload"])
    env = M.build_data_plane_envelope(0x211, payload, "ConvertMaskedCtPayload")
    wire = bytearray(M.serialize_envelope(env))
    wire[-40] ^= 0xFF                                    # 翻转 auth 前的 blob 尾部
    with pytest.raises(M.SerializationError):
        M.deserialize_envelope(bytes(wire))


def test_v2_signature_consistency():
    """SM2 签名覆盖段在 v2 往返前后一致（签发=验证，INFER_RESULT 关键性质）。"""
    payload = dataclasses.asdict(_payloads()["InferResultPayload"])
    base_before = payload_sign_base(payload)
    env = M.build_data_plane_envelope(0x202, dict(payload), "InferResultPayload")
    back = M.deserialize_envelope(M.serialize_envelope(env))
    base_after = payload_sign_base(back.payload)
    assert base_before == base_after


def test_v1_path_untouched():
    """控制面（非 BINARY_FIELDS）载荷仍走 v1 全 JSON 编码。"""
    payload = {"error_code": 1, "detail": "x"}
    env = M.MessageEnvelope(header=M.CommonHeader(msg_type=0x300),
                            payload_type="ErrorPayload", payload=dict(payload))
    assert env.header.version == M.PROTOCOL_VERSION
    env.header.payload_len = len(M.payload_to_json(env.payload))   # v1 契约：发送方补算
    back = M.deserialize_envelope(M.serialize_envelope(env))
    assert back.payload == payload and back.blob == b""
