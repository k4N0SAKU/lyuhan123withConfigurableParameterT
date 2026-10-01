"""Beaver m 方拆分与在线乘法单测（P8；beaver_m.py）。"""
from __future__ import annotations

import pytest

from src.crypto.beaver import BeaverTriple
from src.crypto.beaver_m import beaver_multiply_n, split_triple_n
from src.crypto.secret_sharing import DEFAULT_MODULUS


def _rand64() -> int:
    import os
    return int.from_bytes(os.urandom(8), "big") % DEFAULT_MODULUS


def _rand_triple() -> BeaverTriple:
    a, b = _rand64(), _rand64()
    return BeaverTriple(gate_id=0, a=a, b=b, c=(a * b) % DEFAULT_MODULUS)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_split_triple_n_sums_to_full(n):
    t = _rand_triple()
    shares = split_triple_n(t, n)
    assert len(shares) == n
    assert sum(s.a for s in shares) % DEFAULT_MODULUS == t.a
    assert sum(s.b for s in shares) % DEFAULT_MODULUS == t.b
    assert sum(s.c for s in shares) % DEFAULT_MODULUS == t.c


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_beaver_multiply_n_identity(n):
    for _ in range(200):
        t = _rand_triple()
        shares = split_triple_n(t, n)
        xs = [_rand64() for _ in range(n)]
        ys = [_rand64() for _ in range(n)]
        zs = beaver_multiply_n(xs, ys, shares)
        x = sum(xs) % DEFAULT_MODULUS
        y = sum(ys) % DEFAULT_MODULUS
        assert sum(zs) % DEFAULT_MODULUS == (x * y) % DEFAULT_MODULUS


def test_beaver_multiply_n_rejects_mismatch():
    shares = split_triple_n(_rand_triple(), 3)
    with pytest.raises(ValueError):
        beaver_multiply_n([1, 2], [1, 2, 3], shares)
    with pytest.raises(ValueError):
        split_triple_n(_rand_triple(), 0)
