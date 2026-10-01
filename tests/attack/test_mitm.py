"""F7-2 中间人攻击：冒充 P0/P1/P2 三种伪造接入 + 双向中继。

断言：三种伪造接入均被认证拒绝（证书/签名验证失败），且在各诚实节点的
审计链留下 SECURITY_ALERT 告警。

运行：pytest tests/attack/test_mitm.py -v
"""
from __future__ import annotations

import threading

import pytest

from src.crypto.gm_cipher import sm2_generate_keypair, sm2_sign
from src.protocol.auth import CTX_CA
from src.protocol.auth import (AuthError, NodeIdentity, ca_issue_cert,
                               cert_canonical_bytes)
from src.protocol.messages import CertExchangePayload
from src.protocol.session import (ChannelError, SecureChannel, Session,
                                  SessionState)
from tests.attack.conftest import NOW, make_identity
from tests.attack.framework import VERDICTS


class _Clock:
    def __init__(self):
        self.ms = NOW

    def __call__(self):
        return self.ms


def _pair_channels(ca_pub, honest, attacker):
    la, lb = LoopbackPair()
    cha = SecureChannel(Session(__import__("os").urandom(16), 0), honest,
                        __import__("src.protocol.keylifecycle",
                                   fromlist=["KeyManager"]).KeyManager(),
                        clock_ms=_Clock())
    chb = SecureChannel(Session(cha.session.session_id, 1), attacker,
                        __import__("src.protocol.keylifecycle",
                                   fromlist=["KeyManager"]).KeyManager(),
                        clock_ms=_Clock())
    return cha, chb, la, lb


def LoopbackPair():
    from src.nodes.base import LoopbackLink
    return LoopbackLink.create_pair()


def _establish_and_capture(cha, chb, la, lb, ca_pub, alerts_a, alerts_b):
    """并发握手，收集两端异常与审计（不抛出——供断言检查）。"""
    cha._audit = lambda e, d: alerts_a.append((e, d))
    chb._audit = lambda e, d: alerts_b.append((e, d))
    box = {}

    def run(name, chan, link, initiator):
        try:
            chan.establish(link, ca_pub, initiator=initiator)
            box[name] = "ACTIVE"
        except (ChannelError, AuthError) as exc:
            box[name] = f"rejected: {type(exc).__name__}"

    t1 = threading.Thread(target=run, args=("A", cha, la, True))
    t2 = threading.Thread(target=run, args=("B", chb, lb, False))
    t1.start(); t2.start(); t1.join(60); t2.join(60)
    return box


def _forge_cert(node_id):
    """攻击者自签冒充 CA 签发。"""
    evil_pub, evil_priv = sm2_generate_keypair()
    cert = CertExchangePayload(subject_id=node_id, subject_pub=evil_pub,
                               not_before_ms=NOW - 60_000,
                               not_after_ms=NOW + 365 * 24 * 3600 * 1000)
    cert.ca_sig = sm2_sign(evil_priv, cert_canonical_bytes(cert), context=CTX_CA)
    return NodeIdentity(node_id=node_id, role=0, static_pub=evil_pub,
                        static_priv=evil_priv, cert=cert)


class TestMitmImpersonation:
    """三种伪造接入：冒充 P0 接入 P1 / 冒充 P1 接入 P0 / 冒充 P2 接入 P0。"""

    def test_impersonate_p0_to_p1(self, attack_ca):
        honest_p1 = make_identity(attack_ca, "p1_keynode")
        fake_p0 = _forge_cert("p0_client")
        alerts = []
        cha, chb, la, lb = _pair_channels(attack_ca["pub"], fake_p0, honest_p1)
        box = _establish_and_capture(cha, chb, la, lb, attack_ca["pub"],
                                     [], alerts)
        assert box["B"] != "ACTIVE"                    # P1 拒绝
        assert "rejected:" in box["B"]
        assert any(e[0] == "SECURITY_ALERT" or "CERT" in str(e[1])
                   for e in alerts), "P1 审计无告警"
        VERDICTS.collect("F7-2", "冒充 P0→P1", "防御成功",
                         {"result": box["B"], "alerts": len(alerts)})

    def test_impersonate_p1_to_p0(self, attack_ca):
        honest_p0 = make_identity(attack_ca, "p0_client")
        fake_p1 = _forge_cert("p1_keynode")
        alerts = []
        cha, chb, la, lb = _pair_channels(attack_ca["pub"], honest_p0, fake_p1)
        box = _establish_and_capture(cha, chb, la, lb, attack_ca["pub"],
                                     alerts, [])
        assert box["A"] != "ACTIVE"                    # P0 拒绝
        assert "rejected:" in box["A"]
        assert any(e[0] == "SECURITY_ALERT" for e in alerts), "P0 审计无告警"
        VERDICTS.collect("F7-2", "冒充 P1→P0", "防御成功",
                         {"result": box["A"], "alerts": len(alerts)})

    def test_impersonate_p2_to_p0(self, attack_ca):
        honest_p0 = make_identity(attack_ca, "p0_client")
        fake_p2 = _forge_cert("p2_infernode")
        alerts = []
        cha, chb, la, lb = _pair_channels(attack_ca["pub"], honest_p0, fake_p2)
        box = _establish_and_capture(cha, chb, la, lb, attack_ca["pub"],
                                     alerts, [])
        assert box["A"] != "ACTIVE"
        assert any(e[0] == "SECURITY_ALERT" for e in alerts)
        VERDICTS.collect("F7-2", "冒充 P2→P0", "防御成功",
                         {"result": box["A"], "alerts": len(alerts)})

    def test_rogue_ca_cert_rejected(self, attack_ca):
        """合法格式但由另一根签发的证书（假 CA 链）同样拒绝。"""
        rogue_pub, rogue_priv = sm2_generate_keypair()
        rogue_cert = ca_issue_cert(rogue_priv, "p2_infernode", rogue_pub,
                                   now_ms=NOW)
        fake_p2 = NodeIdentity(node_id="p2_infernode", role=2,
                               static_pub=rogue_pub, static_priv=rogue_priv,
                               cert=rogue_cert)
        honest_p0 = make_identity(attack_ca, "p0_client")
        alerts = []
        cha, chb, la, lb = _pair_channels(attack_ca["pub"], honest_p0, fake_p2)
        box = _establish_and_capture(cha, chb, la, lb, attack_ca["pub"],
                                     alerts, [])
        assert box["A"] != "ACTIVE"
        VERDICTS.collect("F7-2", "异 CA 根证书", "防御成功",
                         {"result": box["A"]})


class TestMitmRelay:
    def test_active_relay_between_honest_parties(self, attack_ca):
        """双向中继：攻击者同时冒充两端与两端握手——两端各自拒绝并留告警。"""
        honest_p0 = make_identity(attack_ca, "p0_client")
        honest_p1 = make_identity(attack_ca, "p1_keynode")
        fake_p0, fake_p1 = _forge_cert("p0_client"), _forge_cert("p1_keynode")
        alerts_a, alerts_b = [], []
        # P0 ↔ 冒充 P1
        cha1, chb1, la1, lb1 = _pair_channels(attack_ca["pub"], honest_p0,
                                              fake_p1)
        box1 = _establish_and_capture(cha1, chb1, la1, lb1, attack_ca["pub"],
                                      alerts_a, [])
        # 冒充 P0 ↔ P1
        cha2, chb2, la2, lb2 = _pair_channels(attack_ca["pub"], fake_p0,
                                              honest_p1)
        box2 = _establish_and_capture(cha2, chb2, la2, lb2, attack_ca["pub"],
                                      [], alerts_b)
        assert box1["A"] != "ACTIVE" and box2["B"] != "ACTIVE"
        assert alerts_a and alerts_b                   # 两端均有告警
        VERDICTS.collect("F7-2", "双向中继（两端伪造）", "防御成功",
                         {"p0_view": box1["A"], "p1_view": box2["B"],
                          "alerts_total": len(alerts_a) + len(alerts_b)})

    def test_honest_handshake_still_works(self, attack_ca):
        """对照：同 CA 诚实双方握手成功（拒绝不是误伤）。"""
        a, b = make_identity(attack_ca, "p0_client"), make_identity(
            attack_ca, "p1_keynode")
        cha, chb, la, lb = _pair_channels(attack_ca["pub"], a, b)
        box = _establish_and_capture(cha, chb, la, lb, attack_ca["pub"], [], [])
        assert box["A"] == "ACTIVE" and box["B"] == "ACTIVE"
        VERDICTS.collect("F7-2", "诚实握手对照", "防御成功",
                         {"result": "ACTIVE/ACTIVE"})
