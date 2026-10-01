"""会话与传输通道（docs/01 §2/§3/§4/§8 的实现载体，P4 实现）。

三个组成部分：
- **SM2DH 密钥协商**（GB/T 32918.3 核心）：双临时密钥 + w 截断 + 静态私钥
  混合的单标量乘 V = t·(P_peer + x̄_peer·R_own)；临时公钥经静态私钥签名绑定
  （防 MitM），密钥一致性经 SM3 确认标签校验（国标 Option S1/S_B 确认哈希的
  等价替代——确认消息本就在已认证信封内，见 docs/01 §8.2 声明）。
- **Session**：状态机守卫（非法转换拒绝）+ 消息头哈希链 H_i（ratchet 输入）。
- **SecureChannel**：握手期明文 SM2 签名信封 → 会话期 SM4-GCM（每方向独立
  密钥、nonce=方向标签‖seq、seq 严格单调、时间戳容差窗）；ratchet 由发送侧
  触发（消息数/字节数二选一，可配置），KEY_ROTATE 通知对端前向派生；
  上一纪元密钥共存窗口见 keylifecycle 模块 docstring。

帧格式（F9 字节计数口径：整帧计入 meter，含全部头）：
    u32 len ‖ u8 mode ‖ header(39B) ‖ mode=1: nonce(12) ‖ ct ‖ tag(16)
                                    ‖ mode=2: body ‖ sig(64)
    mode=1: GCM；明文段 = u16 ptype_len ‖ ptype ‖ payload_json；
            AAD = mode‖header（头部认证不加密——防重放检查先于解密）
    mode=2: 握手期明文 + SM2 签名（覆盖 mode‖header‖body）

关键控制消息（KEY_ROTATE/KEY_DESTROY/INFER_RESULT）在 GCM 之上**另附 SM2
签名**：签名覆盖 payload_json（sig 字段除外），上下文 CTX_KEY/CTX_RESULT——
GCM 保完整性，SM2 保来源不可否认。
"""
from __future__ import annotations

import dataclasses
import enum
import os
import queue
import time
from typing import Callable, Dict, List, Optional, Tuple

from src.common.netmeter import NetworkMeter
from src.crypto.gm_cipher import (CTX_AUTH, CTX_KEY, _ECC, _SM2_N, _kg_bytes,
                                  _mul_point, _z_a, sm2_sign, sm2_verify,
                                  sm3_hash, sm3_kdf, sm4_gcm_decrypt,
                                  sm4_gcm_encrypt, AuthenticationError)
from src.protocol import messages as M
from src.protocol.auth import AuthError, Authenticator, NodeIdentity
from src.protocol.keylifecycle import (ROTATE_INTERVAL_ROUNDS, KeyManager,
                                       KeyLifecycleError)

# SM2 曲线参数（GB/T 32918.5；与 gm_cipher 同源 ecc_table）
_P = int(_ECC["p"], 16)
_A = int(_ECC["a"], 16)
# GB/T 32918.3：w = ⌈⌈log2(n)⌉ / 2⌉（SM2 阶为 256-bit ⇒ w = 128）
_W = (_SM2_N.bit_length() + 1) // 2
_MASK_W = (1 << _W) - 1

NONCE_PREFIX_LEN = 4
TS_WINDOW_DEFAULT_MS = 30_000
CONFIRM_SALT = b"A1-22-CONFIRM"
_KEY_MSG_TYPES = {int(M.MsgType.KEY_NEGOTIATE), int(M.MsgType.KEY_ACK),
                  int(M.MsgType.KEY_ROTATE), int(M.MsgType.KEY_DESTROY)}


class ChannelError(Exception):
    """通道层错误。error_code 取 ErrorCode，供 ERR 消息与审计引用。"""

    def __init__(self, error_code: int, detail: str) -> None:
        super().__init__(f"[ErrorCode {error_code}] {detail}")
        self.error_code = error_code
        self.detail = detail


class StateError(ChannelError):
    """状态机非法转换（docs/01 §8.1：拒绝并审计）。"""

    def __init__(self, detail: str) -> None:
        super().__init__(8, detail)   # ErrorCode.STATE_ILLEGAL


# ---- SM2 点运算（协商所需；gmssl 底层仅提供标量乘，点加自实现） ----

def _point_add(p1: bytes, p2: bytes) -> bytes:
    """仿射坐标点加/倍点（p1, p2 为 x‖y 64B；无穷远点用 64B 零表示）。"""
    if len(p1) != 64 or len(p2) != 64:
        raise ValueError("点坐标必须 x‖y 64B")
    x1, y1 = int.from_bytes(p1[:32], "big"), int.from_bytes(p1[32:], "big")
    x2, y2 = int.from_bytes(p2[:32], "big"), int.from_bytes(p2[32:], "big")
    if x1 == 0 and y1 == 0:
        return p2
    if x2 == 0 and y2 == 0:
        return p1
    if x1 == x2:
        if (y1 + y2) % _P == 0:
            return b"\x00" * 64                      # 无穷远点
        lam = (3 * x1 * x1 + _A) * pow(2 * y1, -1, _P) % _P
    else:
        lam = (y2 - y1) * pow((x2 - x1) % _P, -1, _P) % _P
    x3 = (lam * lam - x1 - x2) % _P
    y3 = (lam * (x1 - x3) - y1) % _P
    return x3.to_bytes(32, "big") + y3.to_bytes(32, "big")


def _gen_ephemeral() -> Tuple[int, bytes]:
    """临时密钥 r ∈ [1, n-1]（拒绝采样，S3），返回 (r, R=x‖y)。"""
    while True:
        r = int.from_bytes(os.urandom(32), "big")
        if 1 <= r <= _SM2_N - 1:
            return r, _kg_bytes(r)


def _trunc(x_coord: bytes) -> int:
    """x̄ = 2^w + (x mod 2^w)（GB/T 32918.3 第 5 部分）。"""
    return (1 << _W) + (int.from_bytes(x_coord, "big") & _MASK_W)


def sm2_key_agreement(own_static_priv: bytes, peer_static_pub: bytes,
                      own_eph_priv: int, own_eph_pub: bytes,
                      peer_eph_pub: bytes, initiator_pub: bytes,
                      responder_pub: bytes) -> bytes:
    """GB/T 32918.3 核心协商：返回 32B 共享秘密（双方一致）。

    V = t·(P_peer + x̄_peer·R_own)，t = (d_own + x̄_own·r_own) mod n；
    KDF(x_V ‖ y_V ‖ Z_init ‖ Z_resp, 32)。Z 按 GB/T 32918.2（gm_cipher._z_a）。
    V 为无穷远点（t≡0 或退化）抛 ChannelError——协商失败不产生弱密钥。"""
    x_bar_own = _trunc(own_eph_pub[:32])
    x_bar_peer = _trunc(peer_eph_pub[:32])
    t = (int.from_bytes(own_static_priv, "big") + x_bar_own * own_eph_priv) % _SM2_N
    if t == 0:
        raise ChannelError(7, "协商退化：t ≡ 0")
    base = _point_add(peer_static_pub[-64:],
                      _mul_point(x_bar_peer, peer_eph_pub[-64:]))
    if base == b"\x00" * 64:
        raise ChannelError(7, "协商退化：基点为无穷远")
    v = _mul_point(t, base)
    if v == b"\x00" * 64:
        raise ChannelError(7, "协商退化：V 为无穷远")
    z_init = _z_a(initiator_pub[-64:])
    z_resp = _z_a(responder_pub[-64:])
    return sm3_kdf(v[:32] + v[32:] + z_init + z_resp, 32)


def confirm_tag(shared: bytes) -> bytes:
    """密钥一致性确认标签（SM3(shared ‖ salt) 截断 16B）。"""
    return sm3_hash(shared + CONFIRM_SALT)[:16]


class SessionState(enum.IntEnum):
    """会话状态机（docs/01 §8.1 状态转换表；非法转换必须拒绝并审计）。"""

    IDLE = 0            # 已建连，未认证
    NEGOTIATED = 1      # SM2 双向认证 + 密钥协商完成，通道密钥已派生
    ACTIVE = 2          # 正常业务收发
    ROTATING = 3        # ratchet 轮换进行中（双方向独立计数）
    DESTROYED = 4       # 密钥已覆写销毁（终态，禁止复用）
    ERROR = 5           # 可恢复错误计数超限 / 不可恢复错误


LEGAL_TRANSITIONS: Dict[SessionState, set] = {
    SessionState.IDLE: {SessionState.NEGOTIATED, SessionState.ERROR,
                        SessionState.DESTROYED},
    SessionState.NEGOTIATED: {SessionState.ACTIVE, SessionState.ERROR,
                              SessionState.DESTROYED},
    SessionState.ACTIVE: {SessionState.ROTATING, SessionState.DESTROYED,
                          SessionState.ERROR},
    SessionState.ROTATING: {SessionState.ACTIVE, SessionState.DESTROYED,
                            SessionState.ERROR},
    SessionState.ERROR: {SessionState.DESTROYED},
    SessionState.DESTROYED: set(),
}


class Session:
    """一次端到端推理会话的生命周期容器（状态机守卫，F5）。"""

    def __init__(self, session_id: bytes, role: int) -> None:
        self.session_id = session_id
        self.role = role
        self.state = SessionState.IDLE
        self.history: List[Tuple[int, str, str]] = []   # (ts_ms, from, to) 实测轨迹
        self._transcript = b"\x00" * 32          # H_0 = 0

    def transition(self, target: SessionState) -> None:
        """状态转换守卫：合法转换表见 docs/01 §8.1；非法转换抛 StateError 并审计。"""
        if target not in LEGAL_TRANSITIONS[self.state]:
            raise StateError(
                f"非法状态转换 {self.state.name} → {target.name}"
                f"（session {self.session_id.hex()[:8]}…）")
        self.history.append((int(time.time() * 1000), self.state.name, target.name))
        self.state = target

    def update_transcript(self, header_bytes: bytes) -> bytes:
        """H_i = SM3(H_{i-1} ‖ msg_header_i)——每收发一帧由通道调用一次。"""
        self._transcript = sm3_hash(self._transcript + header_bytes)
        return self._transcript

    def transcript_hash(self) -> bytes:
        """当前消息头哈希链值 H_i（ratchet 输入之一）。"""
        return self._transcript


def _msg_sign_ctx(msg_type: int) -> bytes:
    """信封签名上下文：密钥管理类消息用 CTX_KEY，其余握手用 CTX_AUTH。"""
    return CTX_KEY if int(msg_type) in _KEY_MSG_TYPES else CTX_AUTH


_MODE_GCM = 1
_MODE_CLEAR = 2


def make_envelope(msg_type: M.MsgType, payload_obj) -> M.MessageEnvelope:
    """载荷 dataclass → 信封（payload_type 取类名，payload 为 asdict）。"""
    return M.MessageEnvelope(
        header=M.CommonHeader(msg_type=int(msg_type)),
        payload_type=type(payload_obj).__name__,
        payload=dataclasses.asdict(payload_obj))


def payload_sign_base(payload: dict) -> bytes:
    """关键消息 SM2 签名覆盖段 = payload_json（sig 字段除外）。"""
    p = {k: v for k, v in payload.items() if k != "sig"}
    return M.payload_to_json(p)


class SecureChannel:
    """SM4-GCM 加密通道（每方向独立密钥 + 序列号 + 时间戳防重放）。

    实现要点（P4）：
    - 发送封装层必须经 NetworkMeter 统一字节计数（陷阱 5，含协议头）；
    - 收到 seq 回退/重复 → ERR_SEQ_REPLAY；|Δt| > 容差窗 → ERR_TS_EXPIRED；
    - GCM 校验失败 → ERR_MAC_INVALID 并写审计告警（F4）；
    - ratchet 触发（消息数/字节数，二选一可配置）→ 先发 KEY_ROTATE 再切换。
    """

    def __init__(self, session: Session, own: NodeIdentity, km: KeyManager,
                 meter: NetworkMeter | None = None,
                 clock_ms: Optional[Callable[[], int]] = None,
                 ts_window_ms: int = TS_WINDOW_DEFAULT_MS,
                 ratchet_mode: str = "messages",
                 ratchet_threshold: int = ROTATE_INTERVAL_ROUNDS,
                 audit: Optional[Callable[[str, dict], None]] = None) -> None:
        self.session = session
        self.own = own
        self.km = km
        self.meter = meter if meter is not None else NetworkMeter()
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.ts_window_ms = ts_window_ms
        self.ratchet_mode = ratchet_mode
        self.ratchet_threshold = int(ratchet_threshold)
        self._audit = audit
        self.link = None                          # establish() 时注入
        self._handshake_timeout_s = 30.0
        self.peer: Optional[NodeIdentity] = None
        self.auth: Optional[Authenticator] = None
        self._send_seq = 0
        self._recv_seq = -1
        self._sent_in_epoch = 0
        self._bytes_in_epoch = 0

    # ---- 审计辅助 ----
    def _event(self, event: str, detail: dict) -> None:
        if self._audit is not None:
            self._audit(event, detail)

    # ---- 帧编解码 ----
    def _direction_tag(self) -> bytes:
        """nonce 前 4B：SM3(session_id ‖ 发送方 node_id)——接收方可独立校验
        nonce 与 (seq, 对端) 绑定（防跨通道 nonce 重用）。"""
        return sm3_hash(self.session.session_id
                        + self.own.node_id.encode("utf-8"))[:NONCE_PREFIX_LEN]

    def _expected_peer_tag(self) -> bytes:
        peer_id = self.peer.node_id if self.peer else "?"
        return sm3_hash(self.session.session_id
                        + peer_id.encode("utf-8"))[:NONCE_PREFIX_LEN]

    def _body_bytes(self, env: M.MessageEnvelope) -> bytes:
        return (len(env.payload_type).to_bytes(2, "big")
                + env.payload_type.encode("utf-8")
                + M.payload_to_json(env.payload))

    def _frame_clear(self, env: M.MessageEnvelope) -> bytes:
        cover = bytes([_MODE_CLEAR]) + M.envelope_header_bytes(env) \
            + self._body_bytes(env)
        sig = sm2_sign(self.own.static_priv, cover,
                       context=_msg_sign_ctx(env.header.msg_type))
        frame = cover + sig
        return len(frame).to_bytes(4, "big") + frame

    def _frame_gcm(self, env: M.MessageEnvelope) -> bytes:
        header = M.envelope_header_bytes(env)
        body = self._body_bytes(env)
        nonce = self._direction_tag() + env.header.seq.to_bytes(8, "big")
        ct, tag = sm4_gcm_encrypt(self.km.channel_key("send"), body,
                                  aad=bytes([_MODE_GCM]) + header, nonce=nonce)
        frame = bytes([_MODE_GCM]) + header + nonce + ct + tag
        return len(frame).to_bytes(4, "big") + frame

    # ---- 握手（docs/01 §4.1 时序；证书先行，信任锚 = CA 根公钥） ----
    def establish(self, link, ca_pub: bytes, initiator: bool = False,
                  timeout_s: float = 30.0) -> None:
        """SM2 双向认证 + SM2DH 协商 → 派生双方向 SM4-GCM 密钥（§8.2）。

        握手帧一律明文 + SM2 签名（mode=2）；派生完成后自动进入 ACTIVE。
        timeout_s：握手帧接收超时（多进程 spawn 场景需放宽——对端进程
        导入/启动可能耗时数十秒）。"""
        self.link = link
        self._handshake_timeout_s = timeout_s
        self.auth = Authenticator(ca_pub, self.own, clock_ms=self._clock_ms)
        if initiator:
            self._handshake_initiator()
        else:
            self._handshake_responder()

    def _send_clear(self, msg_type: M.MsgType, payload_obj,
                    payload_sig: bytes = b"") -> None:
        env = make_envelope(msg_type, payload_obj)
        env.header.session_id = self.session.session_id
        env.header.seq = self._send_seq
        env.header.timestamp_ms = self._clock_ms()
        env.header.payload_len = len(M.payload_to_json(env.payload))
        env.auth_kind = int(M.AuthKind.SM2_SIG.value)
        if payload_sig:
            env.payload = {**env.payload, "sig": payload_sig}
            env.header.payload_len = len(M.payload_to_json(env.payload))
        self._send_seq += 1
        frame = self._frame_clear(env)
        self.meter.on_send(len(frame))
        self.link.send(frame)
        self.session.update_transcript(M.envelope_header_bytes(env))

    def _recv_clear(self) -> M.MessageEnvelope:
        try:
            frame = self.link.recv(timeout_s=self._handshake_timeout_s)
        except (queue.Empty, TimeoutError) as exc:
            raise ChannelError(9, f"握手接收超时: {exc!r}") from exc
        self.meter.on_recv(len(frame))
        if len(frame) < 4 + 1 + 39 + 2 + 64:
            raise ChannelError(10, "握手帧过短")
        body = frame[4:]
        mode = body[0]
        if mode != _MODE_CLEAR:
            raise ChannelError(10, "握手期收到非明文帧")
        header = body[1:40]
        rest = body[40:]
        ptype_len = int.from_bytes(rest[:2], "big")
        ptype = rest[2:2 + ptype_len].decode("utf-8")
        payload = M.payload_from_json(rest[2 + ptype_len:-64])
        sig = rest[-64:]
        env = M.MessageEnvelope(
            header=M.CommonHeader(
                version=header[0], msg_type=int.from_bytes(header[1:3], "big"),
                session_id=header[3:19], seq=int.from_bytes(header[19:27], "big"),
                timestamp_ms=int.from_bytes(header[27:35], "big"),
                payload_len=int.from_bytes(header[35:39], "big")),
            payload_type=ptype, payload=payload,
            auth_kind=int(M.AuthKind.SM2_SIG.value), auth_value=sig)
        # transcript 链收发双侧等价更新（链一致性是 ratchet 的前提）
        self.session.update_transcript(M.envelope_header_bytes(env))
        return env

    def _accept_cert(self, env: M.MessageEnvelope) -> None:
        if env.payload_type != "CertExchangePayload":
            self._event("SECURITY_ALERT", {"code": int(M.ErrorCode.CERT_INVALID),
                                           "why": "握手期未出示证书"})
            raise ChannelError(2, "握手顺序错误：期待证书交换")
        cert = M.CertExchangePayload(
            **{k: M._decode_json_value(v) for k, v in env.payload.items()})
        try:
            self.auth.verify_cert(cert, self._clock_ms())
        except AuthError as exc:
            # F7-2：伪造/过期证书的接入尝试必须留审计告警
            self._event("SECURITY_ALERT", {
                "code": int(M.ErrorCode.CERT_INVALID
                            if exc.error_code == 2 else M.ErrorCode.CERT_EXPIRED),
                "why": f"握手证书验证失败: {exc.detail}",
                "claimed_subject": cert.subject_id})
            raise
        self.peer = NodeIdentity(node_id=cert.subject_id,
                                 static_pub=cert.subject_pub, cert=cert)

    def _verify_peer_env(self, env: M.MessageEnvelope) -> None:
        """验对端信封签名（对端证书必须已先行验证）。"""
        if self.peer is None:
            raise ChannelError(2, "对端证书缺失")
        cover = bytes([_MODE_CLEAR]) + M.envelope_header_bytes(env) \
            + self._body_bytes(env)
        self.auth.verify_envelope(self.peer.node_id, cover, env.auth_value,
                                  context=_msg_sign_ctx(env.header.msg_type))

    # -- 发起方 --
    def _handshake_initiator(self) -> None:
        self._send_clear(M.MsgType.CERT_EXCHANGE, self.own.cert)
        self._accept_cert(self._recv_clear())
        own_nonce = os.urandom(M.NONCE_LEN)
        self._send_clear(M.MsgType.HELLO,
                         M.HelloPayload(node_id=self.own.node_id,
                                        role=self.own.role, nonce=own_nonce))
        env = self._recv_clear()
        self._verify_peer_env(env)
        if env.payload_type != "HelloPayload":
            raise ChannelError(1, "握手顺序错误：期待 HELLO")
        env = self._recv_clear()
        self._verify_peer_env(env)
        if env.payload_type != "AuthChallengePayload":
            raise ChannelError(1, "握手顺序错误：期待挑战")
        p = env.payload
        challenge = M.AuthChallengePayload(
            challenge_nonce=bytes(M._decode_json_value(p["challenge_nonce"])),
            expires_ms=int(p["expires_ms"]))
        resp = self.auth.respond(challenge, responder_id=self.peer.node_id)
        self._send_clear(M.MsgType.AUTH_RESPONSE, resp, payload_sig=resp.sig)
        env = self._recv_clear()
        self._verify_peer_env(env)
        if env.payload_type != "AuthResultPayload":
            raise ChannelError(1, "握手顺序错误：期待 AUTH_RESULT")
        if not env.payload.get("ok"):
            raise ChannelError(int(env.payload.get("error_code", 1) or 1),
                               "对端拒绝认证")
        self.auth.verify_result(self.peer.cert,
                                bytes(M._decode_json_value(env.payload["sig"])),
                                challenge.challenge_nonce, resp.responder_nonce)
        self._key_agreement(initiator=True)

    # -- 应答方 --
    def _handshake_responder(self) -> None:
        self._accept_cert(self._recv_clear())
        self._send_clear(M.MsgType.CERT_EXCHANGE, self.own.cert)  # 先回证书，防死锁
        env = self._recv_clear()
        self._verify_peer_env(env)
        if env.payload_type != "HelloPayload":
            raise ChannelError(1, "握手顺序错误：期待 HELLO")
        own_nonce = os.urandom(M.NONCE_LEN)
        self._send_clear(M.MsgType.HELLO,
                         M.HelloPayload(node_id=self.own.node_id,
                                        role=self.own.role, nonce=own_nonce))
        challenge = self.auth.issue_challenge()
        self._send_clear(M.MsgType.AUTH_CHALLENGE, challenge)
        env = self._recv_clear()
        self._verify_peer_env(env)
        if env.payload_type != "AuthResponsePayload":
            raise ChannelError(1, "握手顺序错误：期待应答")
        p = env.payload
        resp = M.AuthResponsePayload(
            challenge_nonce=bytes(M._decode_json_value(p["challenge_nonce"])),
            responder_nonce=bytes(M._decode_json_value(p["responder_nonce"])),
            sig=bytes(M._decode_json_value(p["sig"])))
        self.auth.verify_response(resp, challenge.challenge_nonce, self.peer.cert)
        result_sig = self.auth.make_result(self.peer.node_id,
                                           challenge.challenge_nonce,
                                           resp.responder_nonce)
        self._send_clear(M.MsgType.AUTH_RESULT,
                         M.AuthResultPayload(ok=True, error_code=0),
                         payload_sig=result_sig)
        self._key_agreement(initiator=False)

    # -- SM2DH 协商（双角色） --
    def _key_agreement(self, initiator: bool) -> None:
        own_eph_priv, own_eph_pub = _gen_ephemeral()
        eph_sig = sm2_sign(self.own.static_priv,
                           b"\x04" + own_eph_pub + (0).to_bytes(8, "big")
                           + self.session.session_id, context=CTX_KEY)
        if initiator:
            self._send_clear(M.MsgType.KEY_NEGOTIATE,
                             M.KeyNegotiatePayload(ephemeral_pub=b"\x04" + own_eph_pub,
                                                   epoch=0, sig=eph_sig))
            peer_eph = self._parse_negotiate(self._recv_clear())
            shared = self._compute_shared(own_eph_priv, own_eph_pub, peer_eph)
            ack = self._recv_clear()
            self._verify_peer_env(ack)
            if ack.payload_type != "KeyAckPayload":
                raise ChannelError(1, "期待 KEY_ACK")
            if bytes(M._decode_json_value(ack.payload["confirm_tag"])) \
                    != confirm_tag(shared):
                raise ChannelError(7, "KEY_ACK 确认标签不符（密钥不一致/中间人）")
        else:
            peer_eph = self._parse_negotiate(self._recv_clear())
            shared = self._compute_shared(own_eph_priv, own_eph_pub, peer_eph)
            self._send_clear(M.MsgType.KEY_NEGOTIATE,
                             M.KeyNegotiatePayload(ephemeral_pub=b"\x04" + own_eph_pub,
                                                   epoch=0, sig=eph_sig))
            self._send_clear(M.MsgType.KEY_ACK,
                             M.KeyAckPayload(epoch=0, confirm_tag=confirm_tag(shared)))
        # 双方派生通道密钥（中间值登记入销毁清单）。方向标签按 node_id 全序
        # 规范化：同一条方向两端派生同一密钥（A 的 send == B 的 recv）。
        if self.own.node_id < self.peer.node_id:
            send_label, recv_label = "d0", "d1"
        else:
            send_label, recv_label = "d1", "d0"
        self.km.register_intermediate("sm2dh_shared", shared)
        self.km.negotiate_root(self.session.transcript_hash())
        self.km.derive_channel_keys(self.session.session_id,
                                    send_label=send_label,
                                    recv_label=recv_label)
        self.session.transition(SessionState.NEGOTIATED)
        self.session.transition(SessionState.ACTIVE)
        self._event("SESSION_ACTIVE", {"peer": self.peer.node_id,
                                       "km_state": self.km.audit_state()})

    def _parse_negotiate(self, env: M.MessageEnvelope) -> bytes:
        self._verify_peer_env(env)
        if env.payload_type != "KeyNegotiatePayload":
            raise ChannelError(1, "期待 KEY_NEGOTIATE")
        eph_pub = bytes(M._decode_json_value(env.payload["ephemeral_pub"]))
        epoch = int(env.payload["epoch"])
        sig = bytes(M._decode_json_value(env.payload["sig"]))
        if not sm2_verify(self.peer.static_pub,
                          eph_pub + epoch.to_bytes(8, "big")
                          + self.session.session_id, sig, context=CTX_KEY):
            raise ChannelError(1, "临时公钥签名验证失败（疑似 MitM）")
        return eph_pub[-64:]

    def _compute_shared(self, own_eph_priv: int, own_eph_pub: bytes,
                        peer_eph_pub: bytes) -> bytes:
        """角色序以 node_id 全序固定（双方独立计算得到同一 Z 序）。"""
        i_am_initiator = self.own.node_id < self.peer.node_id
        initiator_pub = self.own.static_pub if i_am_initiator else self.peer.static_pub
        responder_pub = self.peer.static_pub if i_am_initiator else self.own.static_pub
        return sm2_key_agreement(self.own.static_priv, self.peer.static_pub,
                                 own_eph_priv, own_eph_pub, peer_eph_pub,
                                 initiator_pub, responder_pub)

    # ---- 会话期收发 ----
    def send_message(self, envelope: M.MessageEnvelope) -> int:
        """发送信封：会话期 GCM；ratchet 触发时先发 KEY_ROTATE。返回帧长。"""
        if self.session.state == SessionState.DESTROYED:
            raise ChannelError(8, "会话已销毁，禁止发送")
        if self.session.state != SessionState.ACTIVE:
            raise ChannelError(8,
                               f"会话期发送需 ACTIVE（当前 {self.session.state.name}）")
        # ratchet 触发（发送侧计数，双方向独立）；先复位计数再发 KEY_ROTATE，
        # 防止 KEY_ROTATE 自身再次触发（重入）。
        if ((self.ratchet_mode == "messages"
             and self._sent_in_epoch >= self.ratchet_threshold)
                or (self.ratchet_mode == "bytes"
                    and self._bytes_in_epoch >= self.ratchet_threshold)):
            self._rotate_send_key()
        envelope.header.version = M.PROTOCOL_VERSION
        envelope.header.session_id = self.session.session_id
        envelope.header.seq = self._send_seq
        envelope.header.timestamp_ms = self._clock_ms()
        envelope.header.payload_len = len(M.payload_to_json(envelope.payload))
        envelope.auth_kind = int(M.AuthKind.AEAD_TAG.value)
        self._send_seq += 1
        frame = self._frame_gcm(envelope)
        self.meter.on_send(len(frame))
        self.link.send(frame)
        self._sent_in_epoch += 1
        self._bytes_in_epoch += len(frame)
        self.session.update_transcript(M.envelope_header_bytes(envelope))
        return len(frame)

    def send_signed(self, msg_type: M.MsgType, payload_obj,
                    sig_ctx: bytes = CTX_KEY) -> int:
        """关键控制消息：GCM 之上另附 SM2 签名（payload.sig，覆盖段见
        :func:`payload_sign_base`）。"""
        payload = dataclasses.asdict(payload_obj)
        payload["sig"] = sm2_sign(self.own.static_priv,
                                  payload_sign_base(payload), context=sig_ctx)
        env = M.MessageEnvelope(header=M.CommonHeader(msg_type=int(msg_type)),
                                payload_type=type(payload_obj).__name__,
                                payload=payload)
        return self.send_message(env)

    def _rotate_send_key(self) -> None:
        self._sent_in_epoch = 0
        self._bytes_in_epoch = 0
        h = self.session.transcript_hash()
        new_epoch = self.km.current_epoch("send") + 1
        self.send_signed(M.MsgType.KEY_ROTATE,
                         M.KeyRotatePayload(epoch_new=new_epoch,
                                            transcript_hash=h))
        self.km.ratchet("send", h)
        self._event("KEY_ROTATE_SENT", {"new_epoch": new_epoch})

    def recv_message(self, timeout_s: float = 30.0) -> M.MessageEnvelope:
        """接收信封：seq 重放检查 → 时间戳窗 → GCM 解密（旧纪元窗口内回退）。

        KEY_ROTATE 在本方法内部完成接收侧前向派生后原样返回（调用方可按
        payload_type 过滤控制消息）。"""
        if self.session.state == SessionState.DESTROYED:
            raise ChannelError(8, "会话已销毁，禁止接收")
        try:
            frame = self.link.recv(timeout_s=timeout_s)
        except (queue.Empty, TimeoutError) as exc:
            raise ChannelError(9, "接收超时") from exc
        self.meter.on_recv(len(frame))
        if len(frame) < 4 + 1 + 39 + 12 + 16:
            raise ChannelError(10, "帧过短")
        body = frame[4:]
        mode = body[0]
        if mode != _MODE_GCM:
            raise ChannelError(10, "会话期收到非 GCM 帧")
        header = body[1:40]
        seq = int.from_bytes(header[19:27], "big")
        ts = int.from_bytes(header[27:35], "big")
        # 重放检查（先于解密：防旧帧进入密码层）
        if seq <= self._recv_seq:
            self._event("SECURITY_ALERT", {"code": int(M.ErrorCode.SEQ_REPLAY),
                                           "seq": seq, "last": self._recv_seq})
            raise ChannelError(5, f"seq {seq} ≤ 已收 {self._recv_seq}（重放）")
        # 时间戳容差窗（F6）
        if abs(self._clock_ms() - ts) > self.ts_window_ms:
            self._event("SECURITY_ALERT", {"code": int(M.ErrorCode.TS_EXPIRED),
                                           "seq": seq, "ts": ts})
            raise ChannelError(6, f"时间戳超窗（Δ={abs(self._clock_ms() - ts)}ms）")
        nonce = body[40:52]
        if nonce != self._expected_peer_tag() + seq.to_bytes(8, "big"):
            self._event("SECURITY_ALERT", {"code": int(M.ErrorCode.MAC_INVALID),
                                           "seq": seq,
                                           "why": "nonce 与 seq/对端不绑定"})
            raise ChannelError(4, "nonce 与 (seq, 对端) 绑定校验失败")
        ct, tag = body[52:-16], body[-16:]
        aad = bytes([_MODE_GCM]) + header
        plaintext = self._decrypt_with_epoch_fallback(ct, tag, aad, nonce)
        ptype_len = int.from_bytes(plaintext[:2], "big")
        env = M.MessageEnvelope(
            header=M.CommonHeader(
                version=header[0], msg_type=int.from_bytes(header[1:3], "big"),
                session_id=header[3:19], seq=seq, timestamp_ms=ts,
                payload_len=int.from_bytes(header[35:39], "big")),
            payload_type=plaintext[2:2 + ptype_len].decode("utf-8"),
            payload=M.payload_from_json(plaintext[2 + ptype_len:]),
            auth_kind=int(M.AuthKind.AEAD_TAG.value), auth_value=tag)
        self._recv_seq = seq
        # KEY_ROTATE 的 transcript 基准 = 本帧之前的链值（发送侧在发出轮换帧
        # 前捕获；FIFO 下接收侧此刻的链与发送侧一致），先取基准再更新链。
        pre_hash = self.session.transcript_hash()
        self.session.update_transcript(header)
        self.km.on_epoch_message("recv", self.km.current_epoch("recv"))
        if env.payload_type == "KeyRotatePayload":
            self._handle_key_rotate(env, pre_hash)
        return env

    def _decrypt_with_epoch_fallback(self, ct: bytes, tag: bytes, aad: bytes,
                                     nonce: bytes) -> bytes:
        cur = self.km.current_epoch("recv")
        try:
            return sm4_gcm_decrypt(self.km.channel_key("recv"), ct, tag,
                                   aad=aad, nonce=nonce)
        except AuthenticationError:
            pass
        # 上一纪元共存窗口（keylifecycle docstring 语义）：窗口内可解
        try:
            prev = self.km.channel_key("recv", cur - 1)
        except KeyLifecycleError as exc:
            self._event("SECURITY_ALERT", {"code": int(M.ErrorCode.MAC_INVALID),
                                           "why": "旧纪元密钥已销毁（窗口外）"})
            raise ChannelError(4, f"旧纪元密钥已销毁（窗口外乱序/攻击）: {exc}") from exc
        try:
            self._event("OLD_EPOCH_ACCEPTED", {"epoch": cur - 1,
                                               "window": self.km.accept_window})
            return sm4_gcm_decrypt(prev, ct, tag, aad=aad, nonce=nonce)
        except AuthenticationError as exc:
            self._event("SECURITY_ALERT", {"code": int(M.ErrorCode.MAC_INVALID),
                                           "why": "当前与上一纪元均校验失败"})
            raise ChannelError(4, "GCM 标签校验失败（消息被篡改或密钥不匹配）") from exc

    def _handle_key_rotate(self, env: M.MessageEnvelope,
                           pre_hash: bytes) -> None:
        """KEY_ROTATE：验 payload 内 SM2 签名 → 接收侧前向派生。

        transcript 校验基准为**本帧之前**的链值 pre_hash（与发送侧捕获点
        严格一致，FIFO 顺序保证）。"""
        sig = bytes(M._decode_json_value(env.payload.get("sig") or b""))
        if not sig or not sm2_verify(self.peer.static_pub,
                                     payload_sign_base(env.payload), sig,
                                     context=CTX_KEY):
            self._event("SECURITY_ALERT", {"code": int(M.ErrorCode.AUTH_FAILURE),
                                           "why": "KEY_ROTATE 签名不符"})
            raise ChannelError(1, "KEY_ROTATE 签名验证失败")
        h = bytes(M._decode_json_value(env.payload["transcript_hash"]))
        if h != pre_hash:
            self._event("SECURITY_ALERT", {"why": "KEY_ROTATE transcript 不符"})
            raise ChannelError(1, "KEY_ROTATE transcript hash 与本地链不符")
        self.km.ratchet("recv", h)
        self._event("KEY_ROTATE_RECV", {"new_epoch": self.km.current_epoch("recv")})

    def current_seq(self, direction: str) -> int:
        """direction: 'send' | 'recv'；用于重放窗口检查的测试观测点。"""
        return self._send_seq if direction == "send" else self._recv_seq

    def destroy(self, reason: str = "session_end") -> dict:
        """销毁会话密钥（终态）；通道此后拒绝收发。"""
        detail = self.km.destroy(reason)
        self.session.transition(SessionState.DESTROYED)
        self._event("SESSION_DESTROYED", {"reason": reason,
                                          "km_state": self.km.audit_state()})
        return detail
