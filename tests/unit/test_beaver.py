"""Beaver 三元组与在线乘法协议测试（P2）：正确性 / 消耗对账 / 库完整性。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.crypto.beaver import (DEFAULT_MODULUS, BeaverTriple, OfflineDepletedError,
                               TripleStore, beaver_multiply, split_triple)
from src.crypto.secret_sharing import additive_share

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestOnlineMultiply:
    def test_correctness_positive(self):
        x, y = 1234, 567
        x0, x1 = additive_share(x)
        y0, y1 = additive_share(y)
        tr = BeaverTriple(gate_id=0, a=7, b=5, c=35)
        s1, s2 = split_triple(tr)
        z1, z2 = beaver_multiply(x0, y0, s1, x1, y1, s2)
        assert (z1 + z2) % DEFAULT_MODULUS == x * y

    def test_correctness_random_many(self):
        import random
        random.seed(11)
        for _ in range(50):
            x = random.randrange(0, 2**40)
            y = random.randrange(0, 2**40)
            tr = BeaverTriple(gate_id=0,
                              a=random.randrange(0, DEFAULT_MODULUS),
                              b=random.randrange(0, DEFAULT_MODULUS),
                              c=0)
            tr.c = (tr.a * tr.b) % DEFAULT_MODULUS
            s1, s2 = split_triple(tr)
            x0, x1 = additive_share(x)
            y0, y1 = additive_share(y)
            z1, z2 = beaver_multiply(x0, y0, s1, x1, y1, s2)
            assert (z1 + z2) % DEFAULT_MODULUS == (x * y) % DEFAULT_MODULUS

    def test_triple_shares_reconstruct(self):
        tr = BeaverTriple(gate_id=3, a=11, b=13, c=143)
        s1, s2 = split_triple(tr)
        assert (s1.a + s2.a) % DEFAULT_MODULUS == tr.a
        assert (s1.b + s2.b) % DEFAULT_MODULUS == tr.b
        assert (s1.c + s2.c) % DEFAULT_MODULUS == tr.c


class TestTripleStore:
    def test_fetch_and_depletion(self, tmp_path):
        store = tmp_path / "store.json"
        rows = [{"gate_id": i, "a": i, "b": i + 1,
                 "c": (i * (i + 1)) % DEFAULT_MODULUS, "used": False}
                for i in range(5)]
        data = {"triples": rows, "checksum": _checksum(rows)}
        store.write_text(json.dumps(data), encoding="utf-8")
        ts = TripleStore(str(store))
        assert ts.remaining() == 5
        got = ts.fetch(3)
        assert len(got) == 3 and ts.remaining() == 2
        assert all(t.b == t.a + 1 for t in got)          # 内容与库存一致
        ts.fetch(2)
        with pytest.raises(OfflineDepletedError):
            ts.fetch(1)                                   # D7：不得在线补生成

    def test_tamper_detected(self, tmp_path):
        store = tmp_path / "store.json"
        rows = [{"gate_id": 0, "a": 7, "b": 5, "c": 35, "used": False}]
        store.write_text(json.dumps({"triples": rows,
                                     "checksum": _checksum(rows)}), encoding="utf-8")
        rows[0]["c"] = 99                                  # 篡改 c（校验和不变）
        store.write_text(json.dumps({"triples": rows,
                                     "checksum": _checksum(
                                         [{"gate_id": 0, "a": 7, "b": 5,
                                           "c": 35, "used": False}])}),
                         encoding="utf-8")
        with pytest.raises(ValueError):
            TripleStore(str(store))


def _checksum(rows) -> str:
    from gmssl import sm3
    return sm3.sm3_hash(
        list("".join(json.dumps(t, sort_keys=True) for t in rows).encode("utf-8")))


class TestOfflineGenProcess:
    def test_cli_generates_and_store_loads(self, tmp_path):
        store = tmp_path / "gen.json"
        proc = subprocess.run(
            [sys.executable, "-m", "src.nodes.offline_triple_gen",
             "--store", str(store), "--batch", "10"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        ts = TripleStore(str(store))
        assert ts.remaining() == 10
        tr = ts.fetch(1)[0]
        assert (tr.a * tr.b) % DEFAULT_MODULUS == tr.c     # 生成器正确性
