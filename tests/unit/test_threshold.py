"""门限解密协议逻辑（模拟层）测试（P2）：正确性 / 失败路径 / 销毁拒绝。

诚实声明：本模块为门限协议逻辑的标量 LWE 模拟（非真 CKKS 门限），见
src/crypto/threshold.py 与 docs/01 §8.4 的 D5 修订说明。
"""
from __future__ import annotations

import pytest

from src.crypto.threshold import (ThresholdError, combine, destroy_share,
                                  partial_decrypt, threshold_encrypt,
                                  threshold_keygen)


@pytest.fixture(scope="module")
def keys():
    return threshold_keygen()


class TestCombineCorrectness:
    @pytest.mark.parametrize("value", [0, 1, 42, -777, 12345, 10**9])
    def test_recover_value(self, keys, value):
        ct = threshold_encrypt(keys, value)
        p0 = partial_decrypt(keys[0], ct)
        p1 = partial_decrypt(keys[1], ct)
        assert combine([(0, *p0), (1, *p1)], ct) == value


class TestFailurePaths:
    """任务书要求：缺份额不可解 + 伪造份额拒绝。"""

    def test_missing_share_cannot_decrypt(self, keys):
        ct = threshold_encrypt(keys, 7)
        p0 = partial_decrypt(keys[0], ct)
        with pytest.raises(ThresholdError):
            combine([(0, *p0)], ct)                       # 只有 1 份
        with pytest.raises(ThresholdError):
            combine([], ct)                               # 零份

    def test_duplicate_party_rejected(self, keys):
        ct = threshold_encrypt(keys, 7)
        p0 = partial_decrypt(keys[0], ct)
        with pytest.raises(ThresholdError):
            combine([(0, *p0), (0, *p0)], ct)             # 同一方重复 ≠ 两方

    def test_forged_share_commitment_rejected(self, keys):
        ct = threshold_encrypt(keys, 7)
        p0 = partial_decrypt(keys[0], ct)
        p1 = partial_decrypt(keys[1], ct)
        forged = (p1[0][0] ^ 1).to_bytes(16, "big")       # 翻转 1 字节
        with pytest.raises(ThresholdError):
            combine([(0, *p0), (1, forged, p1[1])], ct)

    def test_destroyed_key_rejected(self, keys):
        ct = threshold_encrypt(keys, 7)
        s0, s1 = keys
        destroy_share(s1)
        with pytest.raises(ThresholdError):
            partial_decrypt(s1, ct)                       # F5：销毁后拒绝
        destroy_share(s0)
        with pytest.raises(ThresholdError):
            partial_decrypt(s0, ct)
