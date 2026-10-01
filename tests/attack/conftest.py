"""攻击套件共享夹具与会话收尾（判定落盘 benchmarks/results/attack_verdicts.json）。"""
from __future__ import annotations

import pytest

from tests.attack.framework import VERDICTS

NOW = 1_700_000_000_000


@pytest.fixture(scope="session")
def attack_ca():
    """会话级 CA（离线根：签发测试身份）。"""
    from src.crypto.gm_cipher import sm2_generate_keypair
    ca_pub, ca_priv = sm2_generate_keypair()
    return {"pub": ca_pub, "priv": ca_priv}


def make_identity(ca, node_id):
    from src.crypto.gm_cipher import sm2_generate_keypair
    from src.protocol.auth import NodeIdentity, ca_issue_cert
    pub, priv = sm2_generate_keypair()
    return NodeIdentity(node_id=node_id, role=0, static_pub=pub,
                        static_priv=priv,
                        cert=ca_issue_cert(ca["priv"], node_id, pub, now_ms=NOW))


def pytest_sessionfinish(session, exitstatus):
    if VERDICTS.entries:
        summary = VERDICTS.dump()
        print(f"\n=== 攻击判定汇总 → {summary} ===")
