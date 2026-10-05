"""消息格式定义（P1 规格的权威代码载体；docs/01 §3 与本模块字段一一对应）。

P1 阶段约束：本模块只含 dataclass / 枚举 / 常量，任何方法实现一律
``raise NotImplementedError``，由 P3（数据面）/P4（控制面）阶段填充。

字段宽度约定（docs/01 §3.1）：
- version: u8；msg_type: u16；seq: u64（会话内每方向单调递增，从 0 起）；
- timestamp_ms: u64（发送方 UTC 毫秒）；session_id: 16 字节随机会话标识。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

PROTOCOL_VERSION = 1
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
        return _sm3.sm3_hash(list(data))

# 端口约定（docs/01 §2；IANA 未注册段 17400-17449）
PORT_KEYNODE = 17401     # P1 密钥服务节点（认证 + 密钥管理 + 门限解密参与）
PORT_INFERnode = 17402   # P2 推理服务节点
PORT_KEYNODE_INTERNAL = 17403  # P1 ↔ P2 内部通道（MPC 交互/审计副本）

SESSION_ID_LEN = 16
NONCE_LEN = 32


class MsgType(enum.IntEnum):
    """消息类型。0x01xx 控制面（认证/密钥），0x02xx 数据面（推理），0x03xx 错误。"""

    # ---- 控制面：认证 ----
    HELLO = 0x0101                  # 节点握手：身份声明 + 随机数
    CERT_EXCHANGE = 0x0102          # 证书交换（结构化签名对象，S4 演示级）
    AUTH_CHALLENGE = 0x0103         # 挑战
    AUTH_RESPONSE = 0x0104          # 应答（SM2 签名 over 会话杂凑）
    AUTH_RESULT = 0x0105            # 双向认证结论
    # ---- 控制面：密钥管理 ----
    KEY_NEGOTIATE = 0x0111          # SM2 密钥协商（临时公钥 + 签名）
    KEY_ACK = 0x0112                # 协商确认
    KEY_ROTATE = 0x0113             # ratchet 轮换通知（epoch + transcript hash）
    KEY_DESTROY = 0x0114            # 销毁指令
    HEARTBEAT = 0x0115              # 存活探测
    # ---- 数据面：模式 A ----
    INFER_REQUEST = 0x0201          # P0→P2：CKKS 密文输入 + 打包元数据
    INFER_RESULT = 0x0202           # P2→P0：密文 logits + P2 签名
    THRESHOLD_PARTIAL = 0x0203      # 门限部分份额（扩展方向保留；主线 D5 修订为
                                    # P0 掩码解密，不使用，见 docs/01 §8.4）
    # ---- 数据面：模式 B 密文↔分享转换（docs/01 §7.4）----
    CONVERT_MASKED_CT = 0x0211      # P2→P0：ct(x)+Enc(r)，P0 单钥解密（D5 修订）
    CONVERT_SHARE = 0x0212          # P0→P1：掩码解密值 y₁=x+r 分发；P2 本地持 −r
    MPC_OPEN = 0x0221               # Beaver 乘法 open（d 或 e 的分享）
    MPC_RESULT_SHARE = 0x0222       # MPC 结果分享
    RECRYPT_SHARES = 0x0231         # 分享→密文：Enc(a1+s) 与 Enc(s)
    # ---- 错误 ----
    ERR = 0x0300                    # 错误通告（载荷 ErrorPayload）


class ErrorCode(enum.IntEnum):
    """错误码（docs/01 §3.4；对应 F3~F6 攻击测试的拒绝原因）。"""

    AUTH_FAILURE = 1            # 挑战应答失败
    CERT_INVALID = 2            # 无证书/伪造证书（F3）
    CERT_EXPIRED = 3            # 过期证书（F3）
    MAC_INVALID = 4             # GCM 校验失败/消息被篡改（F4）
    SEQ_REPLAY = 5              # 序列号回退/重复 → 重放（F6）
    TS_EXPIRED = 6              # 时间戳超出容差窗（F6）
    DECRYPT_FAILURE = 7         # 门限解密失败（份额不匹配/销毁后解密）
    STATE_ILLEGAL = 8           # 状态机非法转换
    TIMEOUT = 9                 # 交互超时
    INTERNAL = 10


class AuthKind(enum.IntEnum):
    """envelope 的认证方式：传输层 AEAD 标签 或 应用层 SM2 签名。"""

    NONE = 0        # 仅允许在 HELLO 之前不存在（实际不允许 NONE 用于业务消息）
    AEAD_TAG = 1    # SM4-GCM tag（通道密钥保护，P4 实现）
    SM2_SIG = 2     # SM2 签名（关键控制面消息：认证/轮换/销毁/结果）


@dataclass
class CommonHeader:
    """所有消息的公共头（docs/01 §3.1 字段表）。"""

    version: int = PROTOCOL_VERSION          # u8
    msg_type: int = 0                        # u16，取 MsgType 值
    session_id: bytes = b"\x00" * SESSION_ID_LEN   # 16B，随机会话标识
    seq: int = 0                             # u64，单调递增
    timestamp_ms: int = 0                    # u64，发送方 UTC 毫秒
    payload_len: int = 0                     # u32，载荷字节数


@dataclass
class MessageEnvelope:
    """传输信封：公共头 + 载荷 + 认证字段。

    完整性/机密性分层（docs/01 §2.3）：数据面消息在 SM4-GCM 通道内传输
    （AEAD_TAG 覆盖 header+payload）；关键控制面消息附加 SM2_SIG。
    """

    header: CommonHeader = field(default_factory=CommonHeader)
    payload_type: str = ""                   # 载荷 dataclass 类名
    payload: dict = field(default_factory=dict)    # 载荷的规范化字典
    auth_kind: int = AuthKind.AEAD_TAG.value
    auth_value: bytes = b""                  # tag / 签名
    blob: bytes = b""                        # v2：数据面大载荷原始二进制（JSON 外）


# ---- 控制面载荷 ----

@dataclass
class HelloPayload:
    node_id: str = ""
    role: int = 0                            # 0=P0 客户端, 1=P1 密钥节点, 2=P2 推理节点
    nonce: bytes = b"\x00" * NONCE_LEN
    supported_versions: tuple = (PROTOCOL_VERSION,)


@dataclass
class CertExchangePayload:
    """演示级结构化签名证书（S4）：{身份, 公钥, 有效期, CA 签名}。"""

    subject_id: str = ""
    subject_pub: bytes = b""                 # SM2 公钥（未压缩 65B）
    not_before_ms: int = 0
    not_after_ms: int = 0
    ca_sig: bytes = b""                      # CA 对前述字段的 SM2 签名


@dataclass
class AuthChallengePayload:
    challenge_nonce: bytes = b"\x00" * NONCE_LEN
    expires_ms: int = 0                      # 挑战有效期（防拖延重放）


@dataclass
class AuthResponsePayload:
    challenge_nonce: bytes = b"\x00" * NONCE_LEN
    responder_nonce: bytes = b"\x00" * NONCE_LEN
    sig: bytes = b""                         # SM2 签名 over SM3(挑战‖应答者随机数‖会话上下文)


@dataclass
class AuthResultPayload:
    ok: bool = False
    error_code: int = 0                      # 失败时取 ErrorCode
    sig: bytes = b""


@dataclass
class KeyNegotiatePayload:
    """SM2 密钥协商（SM2DH）消息；KDF 用 SM3（docs/01 §8.2）。"""

    ephemeral_pub: bytes = b""               # SM2 临时公钥
    epoch: int = 0                           # 目标会话纪元
    sig: bytes = b""                         # 对 ephemeral_pub‖epoch‖transcript_hash 的签名


@dataclass
class KeyAckPayload:
    epoch: int = 0
    confirm_tag: bytes = b"\x00" * 16        # SM3-KDF 派生确认值（密钥一致性检查）


@dataclass
class KeyRotatePayload:
    epoch_new: int = 0
    transcript_hash: bytes = b"\x00" * 32    # SM3 消息头哈希链值 H_i
    sig: bytes = b""


@dataclass
class KeyDestroyPayload:
    epoch: int = 0
    reason: str = "session_end"
    sig: bytes = b""


# ---- 数据面载荷 ----

@dataclass
class InferRequestPayload:
    """P0→P2 密文推理请求（模式 A/B 通用）。"""

    request_id: bytes = b"\x00" * 16
    mode: int = 0                            # 0=模式A 全密文, 1=模式B 混合
    ciphertext: bytes = b""                  # 序列化 CKKS 密文（嵌入层输出）
    cipher_meta: dict = field(default_factory=dict)   # {level, scale_log2, slots, packing}
    model_id: str = ""                       # "gpt2" / "bert-base-chinese-sentiment"
    max_new_tokens: int = 0                  # 仅 GPT-2


@dataclass
class InferResultPayload:
    """P2→P0 密文结果；仅 P0 持钥且最终输出属其角色（t=1, D5 修订）。"""

    request_id: bytes = b"\x00" * 16
    ciphertext: bytes = b""                  # logits（分类）/单步 logits（生成）
    cipher_meta: dict = field(default_factory=dict)
    sig: bytes = b""                         # P2 对 SM3(ciphertext‖meta) 的签名


@dataclass
class ThresholdPartialPayload:
    """门限解密部分份额（扩展方向保留，主线不使用——docs/01 §8.4 D5 修订）。

    主线中 P0 单钥解密掩码值并分发 y₁；本载荷仅在真门限扩展路线启用。"""

    request_id: bytes = b"\x00" * 16
    holder: int = 0                          # 0=P0, 1=P1
    partial: bytes = b""                     # 部分解密值
    commitment: bytes = b"\x00" * 32         # SM3(partial)，防份额替换


# ---- 模式 B：密文↔分享转换（docs/01 §7.4，再随机化为评审重点）----

@dataclass
class ConvertMaskedCtPayload:
    """P2 请求把密文 x 转为加法分享：ct′ = ct(x)+Enc_pk(r)，r 为 P2 fresh 采样。

    发送方向：P2→P0（D5 修订：P0 单钥掩码解密）；P0 只见 x+r（OTP）。"""

    request_id: bytes = b"\x00" * 16
    masked_ct: bytes = b""                   # ct(x) ⊕ Enc_pk(r)（同态加法）
    level: int = 0
    seq_in_layer: int = 0


@dataclass
class ConvertSharePayload:
    """掩码解密值的分发视图：P0 解密 y=x+r 后把 y₁=y 分发 P1；P2 本地持 −r。"""

    request_id: bytes = b"\x00" * 16
    share: bytes = b""                       # 定点编码的 (x+r)
    holder: int = 0


@dataclass
class MpcOpenPayload:
    """Beaver 乘法 open 阶段：一方公开 d = x - a（或 e = y - b）的分享。"""

    request_id: bytes = b"\x00" * 16
    gate_id: int = 0                         # 乘法门编号（匹配离线三元组）
    opened: bytes = b""                      # 定点分享值


@dataclass
class MpcResultSharePayload:
    request_id: bytes = b"\x00" * 16
    gate_id: int = 0
    share: bytes = b""


@dataclass
class RecryptSharesPayload:
    """分享→密文（再随机化出口）：P1 发送 Enc_pk(s) 与 OTP 掩码分享 z=a₁+s，
    s 为 P1 fresh 采样；P2 计算 w=(z+a₂) mod 2⁶⁴=v+s 后在密文域
    Enc(w)−Enc(s)=Enc(v) 合成 fresh 顶层密文。P2 从不获得 s 明文，
    P1 从不获得 a₂。（P4 修订：出口实现为 CrypTFlow 式偏置掩码——P1 只发
    Enc(s) 与整数 z，密文域合成在 P2 完成；原 Enc(a₁+s) 形态在 2⁶⁴ 份额域
    存在 float64 53 位尾数精度墙与 mod 回绕双重问题，见 docs/01 §7.4 修订。）"""

    request_id: bytes = b"\x00" * 16
    enc_mask: bytes = b""                    # Enc_pk(s)，fresh 顶层密文
    masked_share_ints: bytes = b""           # z = (a₁+s) mod 2⁶⁴（8B 大端/槽）
    level_target: int = 0


@dataclass
class ErrorPayload:
    error_code: int = 0                      # ErrorCode
    failed_seq: int = 0                      # 触发错误的消息 seq
    detail: str = ""


PAYLOAD_REGISTRY = {
    "HelloPayload": HelloPayload, "CertExchangePayload": CertExchangePayload,
    "AuthChallengePayload": AuthChallengePayload,
    "AuthResponsePayload": AuthResponsePayload,
    "AuthResultPayload": AuthResultPayload,
    "KeyNegotiatePayload": KeyNegotiatePayload, "KeyAckPayload": KeyAckPayload,
    "KeyRotatePayload": KeyRotatePayload, "KeyDestroyPayload": KeyDestroyPayload,
    "InferRequestPayload": InferRequestPayload,
    "InferResultPayload": InferResultPayload,
    "ThresholdPartialPayload": ThresholdPartialPayload,
    "ConvertMaskedCtPayload": ConvertMaskedCtPayload,
    "ConvertSharePayload": ConvertSharePayload,
    "MpcOpenPayload": MpcOpenPayload, "MpcResultSharePayload": MpcResultSharePayload,
    "RecryptSharesPayload": RecryptSharesPayload, "ErrorPayload": ErrorPayload,
}


class SerializationError(ValueError):
    """信封序列化/反序列化结构错误。"""


_HEX_TAG = "__hex__"


def _encode_json_value(v):
    if isinstance(v, bytes):
        return {_HEX_TAG: v.hex()}
    if isinstance(v, (list, tuple)):
        return [_encode_json_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _encode_json_value(x) for k, x in v.items()}
    return v


def _decode_json_value(v):
    if isinstance(v, dict):
        if len(v) == 1 and _HEX_TAG in v:
            return bytes.fromhex(v[_HEX_TAG])
        return {k: _decode_json_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_decode_json_value(x) for x in v]
    return v


def build_data_plane_envelope(msg_type, payload: dict,
                              payload_type: str) -> MessageEnvelope:
    """数据面信封工厂（v2）：大字段迁 blob，小 JSON 含 SM3+长度绑定，
    header.version=2。INFER_RESULT 的 sig 字段保留在小 JSON——SM2 签名
    覆盖段=重建后完整载荷的 JSON，签发/验证两侧字节一致。"""
    fields = BINARY_FIELDS.get(payload_type, [])
    small, parts = dict(payload), []
    for f in fields:
        raw = small.pop(f)
        small[f + "_len"] = len(raw)
        small[f + "_sm3"] = sm3_hex(raw)
        parts.append(len(raw).to_bytes(4, "big") + raw)
    return MessageEnvelope(
        header=CommonHeader(msg_type=int(msg_type),
                            version=PROTOCOL_VERSION_BINARY,
                            payload_len=len(payload_to_json(small))),
        payload_type=payload_type, payload=small, blob=b"".join(parts))


def payload_to_json(payload: dict) -> bytes:
    """载荷字典 → 规范 JSON 字节（sort_keys + 紧凑分隔符；bytes 用 hex 标记包）。"""
    import json
    return json.dumps(_encode_json_value(payload), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def payload_from_json(raw: bytes) -> dict:
    import json
    return _decode_json_value(json.loads(raw.decode("utf-8")))


def envelope_header_bytes(envelope: MessageEnvelope) -> bytes:
    """公共头定宽编码（39B：version u8 ‖ msg_type u16 ‖ session_id 16B ‖
    seq u64 ‖ timestamp_ms u64 ‖ payload_len u32，全部大端）。"""
    h = envelope.header
    if len(h.session_id) != SESSION_ID_LEN:
        raise SerializationError("session_id 必须 16B")
    return (h.version.to_bytes(1, "big")
            + int(h.msg_type).to_bytes(2, "big")
            + h.session_id
            + int(h.seq).to_bytes(8, "big")
            + int(h.timestamp_ms).to_bytes(8, "big")
            + int(h.payload_len).to_bytes(4, "big"))


def canonical_bytes(envelope: MessageEnvelope) -> bytes:
    """签名/哈希覆盖段（docs/01 §3.3）：header ‖ ptype_len u16 ‖ ptype ‖ payload_json。

    auth_value 不在覆盖段内（GCM tag/签名本身不可自证）。"""
    body = (len(envelope.payload_type).to_bytes(2, "big")
            + envelope.payload_type.encode("utf-8")
            + payload_to_json(envelope.payload))
    return envelope_header_bytes(envelope) + body


def serialize_envelope(envelope: MessageEnvelope) -> bytes:
    """线格式：canonical_bytes ‖ auth_kind u8 ‖ auth_value_len u32 ‖ auth_value。

    v2（header.version=2 且含 blob）：canonical_bytes 中的 payload 为剥离
    大字段后的小 JSON（含 SM3+长度绑定），blob 原始二进制以 8B 长度前缀
    追加其后——数据面大载荷免 hex 2× 膨胀。"""
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
                           auth_value=auth_value, blob=blob)
