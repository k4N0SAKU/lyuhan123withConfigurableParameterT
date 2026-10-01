"""m 方加性秘密分享单测（P8；secret_sharing_m.py）。"""
from __future__ import annotations

import pytest

from src.crypto.secret_sharing import DEFAULT_MODULUS
from src.crypto.secret_sharing_m import reconstruct_n, share_vector_n


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_share_vector_n_exact_reconstruction(n):
    vals = [12345678901234567 + i * (1 << 40) for i in range(7)]
    shares = share_vector_n(vals, n)
    assert len(shares) == n
    assert reconstruct_n(shares) == vals


def test_share_vector_n_mod_wrap():
    vals = [DEFAULT_MODULUS - 3, 5, (1 << 63) + 11]
    shares = share_vector_n(vals, 3)
    assert reconstruct_n(shares) == [v % DEFAULT_MODULUS for v in vals]


def test_share_vector_n_single_party_is_plaintext():
    vals = [1, 2, 3]
    assert share_vector_n(vals, 1) == [vals]


def test_share_vector_n_randomness():
    vals = [42, 43]
    a = share_vector_n(vals, 3)
    b = share_vector_n(vals, 3)
    assert a[0] != b[0]          # 独立随机（共享随机数会破坏安全性）


def test_share_vector_n_rejects_bad_n():
    with pytest.raises(ValueError):
        share_vector_n([1], 0)


def test_reconstruct_n_length_mismatch():
    with pytest.raises(ValueError):
        reconstruct_n([[1, 2], [3]])
    with pytest.raises(ValueError):
        reconstruct_n([])
