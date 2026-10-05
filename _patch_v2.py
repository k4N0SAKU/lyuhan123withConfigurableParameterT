# -*- coding: utf-8 -*-
"""P9-1 数据面二进制帧（协议 v2）补丁：messages.py + session.py + orchestrator_m.py。
设计：大载荷字段（bytes）从 JSON hex（2×膨胀）改为信封级 blob（原始二进制，
长度前缀拼接），载荷 JSON 内以 SM3 哈希+长度绑定；header.version=2 门控；
控制面与 v1 路径不动；SM2 签名在重建后的完整载荷上计算（两侧字节一致，零改动）。"""
p = "src/protocol/messages.py"
src = open(p, encoding="utf-8").read()

def rep(old, new, tag):
    global src
    assert old in src, "锚点缺失: " + tag
    src = src.replace(old, new, 1)

# 1) 版本常量 + 二进制字段映射 + SM3 助手
rep('''PROTOCOL_VERSION = 1''',
'''PROTOCOL_VERSION = 1
PROTOCOL_VERSION_BINARY = 2        # v2：数据面大载荷走信封级 blob（原始二进制，免 hex 2× 膨胀）

# 数据面载荷中迁移到 blob 的 bytes 字段（按声明序在 blob 内 4B 长度前缀拼接）
BINARY_FIELDS = {
    "InferRequestPayload": ["ciphertext"],
    "InferResultPayload": ["ciphertext"],
    "ConvertMaskedCtPayload": ["masked_ct"],
    "ConvertSharePayload": ["share"],
    "RecryptSharesPayload": ["enc_mask", "masked_share_ints"],
}


def sm3_hex(data: bytes) -> str:
    """SM3 摘要（hex）：优先 OpenSSL C 实现，回退 gmssl 参考实现。"""
    import hashlib
    try:
        return hashlib.new("sm3", data).hexdigest()
    except Exception:
        from gmssl import sm3 as _sm3
        return _sm3.sm3_hash(list(data))''', "版本与映射")

# 2) MessageEnvelope 增加 blob 字段
rep('''    auth_kind: int = AuthKind.AEAD_TAG.value
    auth_value: bytes = b""                  # tag / 签名''',
'''    auth_kind: int = AuthKind.AEAD_TAG.value
    auth_value: bytes = b""                  # tag / 签名
    blob: bytes = b""                        # v2：数据面大载荷原始二进制（JSON 外）''', "blob字段")

# 3) v2 序列化/反序列化
rep('''def serialize_envelope(envelope: MessageEnvelope) -> bytes:
    """线格式：canonical_bytes ‖ auth_kind u8 ‖ auth_len u32 ‖ auth_value。"""
    if envelope.auth_kind not in (k.value for k in AuthKind):
        raise SerializationError(f"未知 auth_kind {envelope.auth_kind}")
    return (canonical_bytes(envelope)
            + int(envelope.auth_kind).to_bytes(1, "big")
            + len(envelope.auth_value).to_bytes(4, "big")
            + envelope.auth_value)


def deserialize_envelope(data: bytes) -> MessageEnvelope:
    """线格式逆变换；结构不合法抛 SerializationError（通道层转 INTERNAL）。"""
    if len(data) < 39 + 2 + 5:
        raise SerializationError("信封过短")
    header = CommonHeader(
        version=data[0],
        msg_type=int.from_bytes(data[1:3], "big"),
        session_id=data[3:19],
        seq=int.from_bytes(data[19:27], "big"),
        timestamp_ms=int.from_bytes(data[27:35], "big"),
        payload_len=int.from_bytes(data[35:39], "big"),
    )
    pos = 39
    ptype_len = int.from_bytes(data[pos:pos + 2], "big")
    pos += 2
    if pos + ptype_len > len(data):
        raise SerializationError("payload_type 越界")
    payload_type = data[pos:pos + ptype_len].decode("utf-8")
    pos += ptype_len
    payload_end = pos + header.payload_len
    if payload_end > len(data):
        raise SerializationError("payload_len 越界")
    payload = payload_from_json(data[pos:payload_end])
    pos = payload_end
    if pos + 5 > len(data):
        raise SerializationError("auth 段缺失")
    auth_kind = data[pos]
    pos += 1
    av_len = int.from_bytes(data[pos:pos + 4], "big")
    pos += 4
    if pos + av_len != len(data):
        raise SerializationError("auth_value 长度不一致")
    auth_value = data[pos:pos + av_len]
    return MessageEnvelope(header=header, payload_type=payload_type,
                           payload=payload, auth_kind=auth_kind,
                           auth_value=auth_value)''',
'''def serialize_envelope(envelope: MessageEnvelope) -> bytes:
    """线格式：canonical_bytes ‖ auth_kind u8 ‖ auth_len u32 ‖ auth_value。

    v2（header.version=2，信封含 blob）：canonical_bytes 中的 payload 为
    剥离大字段后的小 JSON（含各字段 SM3+长度绑定），blob 原始二进制追加在
    payload_json 之后（8B 长度前缀）——大载荷免 hex 2× 膨胀。"""
    if envelope.auth_kind not in (k.value for k in AuthKind):
        raise SerializationError(f"未知 auth_kind {envelope.auth_kind}")
    wire = canonical_bytes(envelope)
    if envelope.blob:
        wire += len(envelope.blob).to_bytes(8, "big") + envelope.blob
    return (wire
            + int(envelope.auth_kind).to_bytes(1, "big")
            + len(envelope.auth_value).to_bytes(4, "big")
            + envelope.auth_value)


def deserialize_envelope(data: bytes) -> MessageEnvelope:
    """线格式逆变换（v1/v2 按 header.version 门控）；结构不合法抛
    SerializationError（通道层转 INTERNAL）；v2 blob 的 SM3/长度绑定不符
    同样抛 SerializationError。"""
    if len(data) < 39 + 2 + 5:
        raise SerializationError("信封过短")
    version = data[0]
    header = CommonHeader(
        version=version,
        msg_type=int.from_bytes(data[1:3], "big"),
        session_id=data[3:19],
        seq=int.from_bytes(data[19:27], "big"),
        timestamp_ms=int.from_bytes(data[27:35], "big"),
        payload_len=int.from_bytes(data[35:39], "big"),
    )
    pos = 39
    ptype_len = int.from_bytes(data[pos:pos + 2], "big")
    pos += 2
    if pos + ptype_len > len(data):
        raise SerializationError("payload_type 越界")
    payload_type = data[pos:pos + ptype_len].decode("utf-8")
    pos += ptype_len
    payload_end = pos + header.payload_len
    if payload_end > len(data):
        raise SerializationError("payload_len 越界")
    payload = payload_from_json(data[pos:payload_end])
    pos = payload_end
    blob = b""
    if version == 2:
        if pos + 8 > len(data):
            raise SerializationError("v2 blob 长度段缺失")
        blob_len = int.from_bytes(data[pos:pos + 8], "big")
        pos += 8
        if pos + blob_len > len(data):
            raise SerializationError("v2 blob 越界")
        blob = data[pos:pos + blob_len]
        pos += blob_len
        # 按声明序切分并校验 SM3/长度绑定，还原完整载荷字典
        fields = BINARY_FIELDS.get(payload_type, [])
        cur = 0
        for f in fields:
            flen = int.from_bytes(blob[cur:cur + 4], "big")
            raw = blob[cur + 4:cur + 4 + flen]
            if len(raw) != flen:
                raise SerializationError(f"v2 blob 字段 {f} 截断")
            cur += 4 + flen
            if payload.get(f + "_len") != flen:
                raise SerializationError(f"v2 blob 字段 {f} 长度绑定不符")
            if payload.get(f + "_sm3") != sm3_hex(raw):
                raise SerializationError(f"v2 blob 字段 {f} SM3 校验失败")
            payload[f] = raw
        for f in fields:
            payload.pop(f + "_len", None)
            payload.pop(f + "_sm3", None)
        if cur != blob_len:
            raise SerializationError("v2 blob 长度与字段声明不一致")
    if pos + 5 > len(data):
        raise SerializationError("auth 段缺失")
    auth_kind = data[pos]
    pos += 1
    av_len = int.from_bytes(data[pos:pos + 4], "big")
    pos += 4
    if pos + av_len != len(data):
        raise SerializationError("auth_value 长度不一致")
    auth_value = data[pos:pos + av_len]
    return MessageEnvelope(header=header, payload_type=payload_type,
                           payload=payload, auth_kind=auth_kind,
                           auth_value=auth_value, blob=blob)''', "序列化v2")

# 4) 数据面信封工厂（供 make_envelope 与 INFER_RESULT 手工路径共用）
rep('''def payload_to_json(payload: dict) -> bytes:''',
'''def build_data_plane_envelope(msg_type, payload: dict,
                              payload_type: str) -> MessageEnvelope:
    """数据面信封工厂（v2）：大字段迁 blob，小 JSON 含 SM3+长度绑定，
    header.version=2。INFER_RESULT 的 sig 字段保留在小 JSON（签名覆盖段
    = 重建后完整载荷的 JSON，签发/验证两侧字节一致——见模块 docstring）。"""
    fields = BINARY_FIELDS.get(payload_type, [])
    small, parts = dict(payload), b""
    for f in fields:
        raw = small.pop(f)
        small[f + "_len"] = len(raw)
        small[f + "_sm3"] = sm3_hex(raw)
        parts.append(len(raw).to_bytes(4, "big") + raw)
    return MessageEnvelope(
        header=CommonHeader(msg_type=int(msg_type),
                            version=PROTOCOL_VERSION_BINARY),
        payload_type=payload_type, payload=small, blob=b"".join(parts))


def payload_to_json(payload: dict) -> bytes:''', "工厂")

open(p, "w", encoding="utf-8", newline="\n").write(src)
print("messages.py patched")

# ---- session.py make_envelope 路由 ----
p2 = "src/protocol/session.py"
s2 = open(p2, encoding="utf-8").read()
old2 = '''    return M.MessageEnvelope(
        header=M.CommonHeader(msg_type=int(msg_type)),
        payload_type=type(payload_obj).__name__,
        payload=dataclasses.asdict(payload_obj))'''
new2 = '''    ptype = type(payload_obj).__name__
    if ptype in M.BINARY_FIELDS:
        return M.build_data_plane_envelope(msg_type, dataclasses.asdict(payload_obj),
                                           ptype)
    return M.MessageEnvelope(
        header=M.CommonHeader(msg_type=int(msg_type)),
        payload_type=ptype,
        payload=dataclasses.asdict(payload_obj))'''
assert old2 in s2, "session 锚点缺失"
s2 = s2.replace(old2, new2, 1)
open(p2, "w", encoding="utf-8", newline="\n").write(s2)
print("session.py patched")

# ---- orchestrator_m.py INFER_RESULT 手工路径改走 v2 ----
p3 = "src/nodes/orchestrator_m.py"
s3 = open(p3, encoding="utf-8").read()
old3 = '''    payload["sig"] = sm2_sign(composer.identity.static_priv,
                              payload_sign_base(payload), context=CTX_RESULT)
    composer.channels["p0-p2"].send_message(M.MessageEnvelope(
        header=M.CommonHeader(msg_type=int(M.MsgType.INFER_RESULT)),
        payload_type="InferResultPayload", payload=payload))'''
new3 = '''    payload["sig"] = sm2_sign(composer.identity.static_priv,
                              payload_sign_base(payload), context=CTX_RESULT)
    composer.channels["p0-p2"].send_message(M.build_data_plane_envelope(
        int(M.MsgType.INFER_RESULT), payload, "InferResultPayload"))'''
assert old3 in s3, "orchestrator 锚点缺失"
s3 = s3.replace(old3, new3, 1)
open(p3, "w", encoding="utf-8", newline="\n").write(s3)
print("orchestrator_m.py patched")
