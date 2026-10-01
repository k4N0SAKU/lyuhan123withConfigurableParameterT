"""SM2 双向认证测试（P4；F3 锚点：证书校验拒绝矩阵 + 挑战应答防重放）。"""
from __future__ import annotations

import os

import pytest

from src.crypto.gm_cipher import sm2_generate_keypair, sm2_sign
from src.protocol.auth import (CTX_CA, CTX_AUTH, AuthError, Authenticator,
                               NodeIdentity, auth_digest, ca_issue_cert,
                               cert_canonical_bytes, verify_cert)
from src.protocol.messages import CertExchangePayload

NOW = 1_700_000_000_000


@pytest.fixture(scope="module")
def ca():
    pub, priv = sm2_generate_keypair()
    return {"pub": pub, "priv": priv}


def _identity(ca_priv, node_id, role=0):
    pub, priv = sm2_generate_keypair()
    cert = ca_issue_cert(ca_priv, node_id, pub, now_ms=NOW)
    return NodeIdentity(node_id=node_id, role=role, static_pub=pub,
                        static_priv=priv, cert=cert)


class TestCerts:
    def test_issue_and_verify_ok(self, ca):
        ident = _identity(ca["priv"], "p1_keynode")
        verify_cert(ca["pub"], ident.cert, NOW)

    def test_expired_cert_rejected(self, ca):
        ident = _identity(ca["priv"], "p1")
        with pytest.raises(AuthError) as ei:
            verify_cert(ca["pub"], ident.cert, NOW + 400 * 24 * 3600 * 1000)
        assert ei.value.error_code == 3          # CERT_EXPIRED

    def test_not_yet_valid_rejected(self, ca):
        pub, _ = sm2_generate_keypair()
        cert = ca_issue_cert(ca["priv"], "p2", pub, now_ms=NOW + 1000)
        # not_before 有 60s 时钟偏移容差：签发前 1 小时验证 → CERT_EXPIRED
        with pytest.raises(AuthError) as ei:
            verify_cert(ca["pub"], cert, NOW - 3600_000)
        assert ei.value.error_code == 3

    def test_forged_cert_rejected(self, ca):
        """F3：伪造证书（自签/篡改字段）必须 CERT_INVALID。"""
        pub, evil_priv = sm2_generate_keypair()
        cert = CertExchangePayload(subject_id="p2_infernode", subject_pub=pub,
                                   not_before_ms=NOW - 60_000,
                                   not_after_ms=NOW + 365 * 24 * 3600 * 1000)
        cert.ca_sig = sm2_sign(evil_priv, cert_canonical_bytes(cert),
                               context=CTX_CA)   # 自签而非 CA 签
        with pytest.raises(AuthError) as ei:
            verify_cert(ca["pub"], cert, NOW)
        assert ei.value.error_code == 2          # CERT_INVALID

    def test_field_tamper_rejected(self, ca):
        ident = _identity(ca["priv"], "p1")
        cert = ident.cert
        cert.subject_id = "p2_infernode"         # 冒用身份
        with pytest.raises(AuthError):
            verify_cert(ca["pub"], cert, NOW)

    def test_wrong_ca_rejected(self, ca):
        other_pub, other_priv = sm2_generate_keypair()
        ident = _identity(other_priv, "p1")
        with pytest.raises(AuthError):
            verify_cert(ca["pub"], ident.cert, NOW)


class TestChallengeResponse:
    def test_happy_path(self, ca):
        a = _identity(ca["priv"], "p0_client")
        b = _identity(ca["priv"], "p1_keynode")
        auth_a = Authenticator(ca["pub"], a, clock_ms=lambda: NOW)
        auth_b = Authenticator(ca["pub"], b, clock_ms=lambda: NOW)
        auth_b.verify_cert(a.cert, NOW)
        auth_a.verify_cert(b.cert, NOW)
        challenge = auth_b.issue_challenge()
        resp = auth_a.respond(challenge, responder_id="p1_keynode")
        auth_b.verify_response(resp, challenge.challenge_nonce, a.cert)
        result_sig = auth_b.make_result("p0_client", challenge.challenge_nonce,
                                        resp.responder_nonce)
        auth_a.verify_result(b.cert, result_sig, challenge.challenge_nonce,
                             resp.responder_nonce)

    def test_replayed_challenge_rejected(self, ca):
        """F6：同一挑战二次应答（重放）拒绝。"""
        a = _identity(ca["priv"], "p0")
        b = _identity(ca["priv"], "p1")
        auth_b = Authenticator(ca["pub"], b, clock_ms=lambda: NOW)
        auth_b.verify_cert(a.cert, NOW)
        challenge = auth_b.issue_challenge()
        resp = Authenticator(ca["pub"], a, clock_ms=lambda: NOW).respond(challenge, responder_id="p1")
        auth_b.verify_response(resp, challenge.challenge_nonce, a.cert)
        with pytest.raises(AuthError):
            auth_b.verify_response(resp, challenge.challenge_nonce, a.cert)

    def test_wrong_signer_rejected(self, ca):
        a = _identity(ca["priv"], "p0")
        impostor = _identity(ca["priv"], "evil")
        b = _identity(ca["priv"], "p1")
        auth_b = Authenticator(ca["pub"], b, clock_ms=lambda: NOW)
        auth_b.verify_cert(a.cert, NOW)
        challenge = auth_b.issue_challenge()
        resp = Authenticator(ca["pub"], impostor, clock_ms=lambda: NOW).respond(challenge,
                                                          responder_id="p1")
        with pytest.raises(AuthError):
            auth_b.verify_response(resp, challenge.challenge_nonce, a.cert)

    def test_expired_challenge_rejected(self, ca):
        a = _identity(ca["priv"], "p0")
        b = _identity(ca["priv"], "p1")
        clock = [NOW]
        auth_a = Authenticator(ca["pub"], a, clock_ms=lambda: clock[0])
        auth_b = Authenticator(ca["pub"], b, clock_ms=lambda: clock[0])
        challenge = auth_b.issue_challenge()
        clock[0] += 60_000                        # 越过 30s TTL
        with pytest.raises(AuthError):
            auth_a.respond(challenge, responder_id="p1")

    def test_reflection_blocked(self, ca):
        """反射防护：把对 A 的应答签名冒充 B 的 AUTH_RESULT——上下文不同而失败。"""
        a = _identity(ca["priv"], "p0")
        b = _identity(ca["priv"], "p1")
        auth_a = Authenticator(ca["pub"], a, clock_ms=lambda: NOW)
        auth_a.verify_cert(b.cert, NOW)
        n1, n2 = os.urandom(32), os.urandom(32)
        resp_sig = sm2_sign(a.static_priv, auth_digest("p0", "p1", n1, n2),
                            context=CTX_AUTH)
        with pytest.raises(Exception):
            auth_a.verify_result(b.cert, resp_sig, n1, n2)   # CTX_AUTH_RESULT 验签失败
