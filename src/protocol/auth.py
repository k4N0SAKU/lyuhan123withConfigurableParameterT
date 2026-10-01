"""SM2 双向身份认证（docs/01 §4.1 / §8.2；F3 的实现载体，P4 实现）。

证书模型（S4 演示级）：结构化签名证书 {身份, 公钥, 有效期, CA 签名}——CA 根
密钥离线生成（``python -m src.nodes.provision``，docs/01 §8.7），节点仅持
CA 公钥作信任锚。商用需替换为 GM/T 0015 X.509 证书链（docs/01 §4.1 声明）。

挑战-应答双向认证（防中间人 + 防重放 + 防反射）：
    ① A→B CERT_EXCHANGE(cert_A, sig_A@CTX_AUTH)   B 先 CA 验签证书再验信封签名
    ② B→A CERT_EXCHANGE(cert_B, sig_B) ＋ HELLO(nonce_b, sig_B) ＋ AUTH_CHALLENGE
    ③ A→B AUTH_RESPONSE：sig_A@CTX_AUTH over digest(challenge ‖ responder_nonce)
    ④ B→A AUTH_RESULT：sig_B@CTX_AUTH_RESULT over digest(nonce_a ‖ nonce_b)
拒绝矩阵（F3 测试用例）：无证书/CA 签名不匹配 → CERT_INVALID；
not_after < now（或 not_before 未到）→ CERT_EXPIRED；签名验证失败/挑战 nonce
不回填/挑战过期/挑战复用 → AUTH_FAILURE。
反射防护：digest 绑定双方 node_id 与角色序（发起方在前）+ 每类签名独立上下文
字符串（CTX_AUTH vs CTX_AUTH_RESULT vs CTX_CA），跨消息类型/跨协议重放失败。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from src.crypto.gm_cipher import (CTX_AUTH, sm2_sign, sm2_verify, sm3_hash)
from src.protocol.messages import (AuthChallengePayload, AuthResponsePayload,
                                   CertExchangePayload, ErrorCode)

CTX_CA = b"A1-22-CA-CERT-V1"
CTX_AUTH_RESULT = b"A1-22-AUTH-RESULT-V1"
DEFAULT_VALIDITY_MS = 365 * 24 * 3600 * 1000
CHALLENGE_TTL_MS = 30_000


class AuthError(Exception):
    """认证失败统一异常。error_code 取 ErrorCode（F3 拒绝原因）。"""

    def __init__(self, error_code: int, detail: str) -> None:
        super().__init__(f"[ErrorCode {error_code}] {detail}")
        self.error_code = error_code
        self.detail = detail


def _default_clock_ms() -> int:
    return int(time.time() * 1000)


def cert_canonical_bytes(cert: CertExchangePayload) -> bytes:
    """CA 签名覆盖段：u16 len ‖ subject_id ‖ subject_pub(65B) ‖ not_before ‖ not_after。"""
    if len(cert.subject_pub) != 65 or cert.subject_pub[0] != 0x04:
        raise AuthError(int(ErrorCode.CERT_INVALID), "公钥必须 65B 未压缩 04‖x‖y")
    sid = cert.subject_id.encode("utf-8")
    return (len(sid).to_bytes(2, "big") + sid + cert.subject_pub
            + int(cert.not_before_ms).to_bytes(8, "big")
            + int(cert.not_after_ms).to_bytes(8, "big"))


def ca_issue_cert(ca_priv: bytes, subject_id: str, subject_pub: bytes,
                  now_ms: Optional[int] = None,
                  validity_ms: int = DEFAULT_VALIDITY_MS) -> CertExchangePayload:
    """CA 签发节点证书（离线工具入口；S4 演示级）。"""
    now = now_ms if now_ms is not None else _default_clock_ms()
    cert = CertExchangePayload(subject_id=subject_id, subject_pub=subject_pub,
                               not_before_ms=now - 60_000,   # 时钟偏移容差
                               not_after_ms=now + validity_ms)
    cert.ca_sig = sm2_sign(ca_priv, cert_canonical_bytes(cert), context=CTX_CA)
    return cert


def verify_cert(ca_pub: bytes, cert: CertExchangePayload, now_ms: int) -> None:
    """校验 CA 签名与有效期；失败抛 AuthError(ErrorCode.CERT_*)。"""
    if cert is None or not cert.ca_sig:
        raise AuthError(int(ErrorCode.CERT_INVALID), "无证书或 CA 签名缺失")
    if len(cert.subject_pub) != 65 or cert.subject_pub[0] != 0x04:
        raise AuthError(int(ErrorCode.CERT_INVALID), "公钥格式非法")
    if not sm2_verify(ca_pub, cert_canonical_bytes(cert), cert.ca_sig,
                      context=CTX_CA):
        raise AuthError(int(ErrorCode.CERT_INVALID),
                        f"CA 签名验证失败（subject={cert.subject_id}）")
    if now_ms > cert.not_after_ms or now_ms < cert.not_before_ms:
        raise AuthError(int(ErrorCode.CERT_EXPIRED),
                        f"证书窗口外（subject={cert.subject_id}, now={now_ms}）")


@dataclass
class NodeIdentity:
    """演示级结构化签名证书对应的节点身份（S4：商用需替换为 GM X.509 证书链）。"""

    node_id: str = ""
    role: int = 0
    static_pub: bytes = b""                  # SM2 静态公钥（65B 未压缩）
    static_priv: bytes = b""                 # 仅本节点持有；导出/审计快照前必须剔除
    cert: Optional[CertExchangePayload] = None


def auth_digest(initiator_id: str, responder_id: str,
                nonce_a: bytes, nonce_b: bytes) -> bytes:
    """认证摘要：域分隔框架 ‖ 发起方 id ‖ 应答方 id ‖ 双 nonce（反射防护）。"""
    a, b = initiator_id.encode("utf-8"), responder_id.encode("utf-8")
    return sm3_hash(b"A1-22-AUTH-DIGEST-V1" + len(a).to_bytes(2, "big") + a
                    + len(b).to_bytes(2, "big") + b + nonce_a + nonce_b)


class Authenticator:
    """挑战-应答双向认证（防中间人：证书绑定 + 随机挑战 + transcript 签名）。"""

    def __init__(self, ca_pub: bytes, local_identity: NodeIdentity,
                 clock_ms: Optional[Callable[[], int]] = None,
                 challenge_ttl_ms: int = CHALLENGE_TTL_MS) -> None:
        self.ca_pub = ca_pub
        self.local = local_identity
        self._clock_ms = clock_ms or _default_clock_ms
        self._ttl = challenge_ttl_ms
        self._issued: Dict[bytes, int] = {}      # 已签发挑战 → 过期时刻（单次使用）
        self._peer_pub_cache: Dict[str, bytes] = {}

    # ---- 证书 ----
    def verify_cert(self, cert: CertExchangePayload, now_ms: int) -> None:
        verify_cert(self.ca_pub, cert, now_ms)
        self._peer_pub_cache[cert.subject_id] = cert.subject_pub

    def sign_envelope(self, message: bytes, context: bytes = CTX_AUTH) -> bytes:
        """本节点静态私钥签名（信封级 auth_value；context 按消息用途区分）。"""
        return sm2_sign(self.local.static_priv, message, context=context)

    def verify_envelope(self, peer_id: str, message: bytes, sig: bytes,
                        context: bytes = CTX_AUTH) -> None:
        """对端静态公钥验签（公钥必须来自已验证证书——防无证书签名伪造）。"""
        pub = self._peer_pub_cache.get(peer_id)
        if pub is None:
            raise AuthError(int(ErrorCode.CERT_INVALID),
                            f"对端 {peer_id} 证书未验证，拒绝验签")
        if not sm2_verify(pub, message, sig, context=context):
            raise AuthError(int(ErrorCode.AUTH_FAILURE),
                            f"对端 {peer_id} 信封签名验证失败")

    # ---- 挑战应答 ----
    def issue_challenge(self) -> AuthChallengePayload:
        """os.urandom 生成挑战（S3：随机数纪律）；单次使用，TTL 内有效。"""
        nonce = os.urandom(32)
        self._issued[nonce] = self._clock_ms() + self._ttl
        return AuthChallengePayload(challenge_nonce=nonce,
                                    expires_ms=self._issued[nonce])

    def respond(self, challenge: AuthChallengePayload,
                responder_id: str = "") -> AuthResponsePayload:
        """发起方应答：签 digest(本地=发起方, 应答方, challenge, fresh nonce)。

        responder_id 为挑战签发方（握手应答方）的 node_id——digest 角色序为
        (发起方, 应答方)，与 verify_response / make_result / verify_result
        全链一致（防反射：身份对调后 digest 不同）。"""
        now = self._clock_ms()
        if now > challenge.expires_ms:
            raise AuthError(int(ErrorCode.AUTH_FAILURE), "挑战已过期（F6 拖延重放）")
        responder_nonce = os.urandom(32)
        digest = auth_digest(self.local.node_id, responder_id,
                             challenge.challenge_nonce, responder_nonce)
        sig = sm2_sign(self.local.static_priv, digest, context=CTX_AUTH)
        return AuthResponsePayload(challenge_nonce=challenge.challenge_nonce,
                                   responder_nonce=responder_nonce, sig=sig)

    def verify_response(self, response: AuthResponsePayload,
                        expected_nonce: bytes, peer_cert: CertExchangePayload) -> None:
        """应答方视角：挑战 nonce 回填 + 单次使用 + digest(对端=发起方, 本地) 验签。"""
        if response.challenge_nonce != expected_nonce:
            raise AuthError(int(ErrorCode.AUTH_FAILURE), "挑战 nonce 未回填")
        if expected_nonce not in self._issued:
            raise AuthError(int(ErrorCode.AUTH_FAILURE), "挑战不存在/已使用（重放）")
        del self._issued[expected_nonce]   # 单次使用：同挑战二次应答一律拒绝
        verify_cert(self.ca_pub, peer_cert, self._clock_ms())
        self._peer_pub_cache[peer_cert.subject_id] = peer_cert.subject_pub
        digest = auth_digest(peer_cert.subject_id, self.local.node_id,
                             expected_nonce, response.responder_nonce)
        if not sm2_verify(peer_cert.subject_pub, digest, response.sig,
                          context=CTX_AUTH):
            raise AuthError(int(ErrorCode.AUTH_FAILURE), "应答签名验证失败")

    def make_result(self, peer_id: str, nonce_a: bytes, nonce_b: bytes) -> bytes:
        """应答方签发 AUTH_RESULT 签名（对 nonce_a‖nonce_b 的确认，独立上下文）。"""
        digest = auth_digest(peer_id, self.local.node_id, nonce_a, nonce_b)
        return sm2_sign(self.local.static_priv, digest, context=CTX_AUTH_RESULT)

    def verify_result(self, peer_cert: CertExchangePayload, sig: bytes,
                      own_nonce: bytes, peer_nonce: bytes) -> None:
        """发起方验证 AUTH_RESULT（对端持有静态私钥的最终证明）。"""
        verify_cert(self.ca_pub, peer_cert, self._clock_ms())
        digest = auth_digest(self.local.node_id, peer_cert.subject_id,
                             own_nonce, peer_nonce)
        if not sm2_verify(peer_cert.subject_pub, digest, sig,
                          context=CTX_AUTH_RESULT):
            raise AuthError(int(ErrorCode.AUTH_FAILURE), "AUTH_RESULT 签名验证失败")
