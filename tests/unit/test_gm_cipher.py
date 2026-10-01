"""国密封装测试（P2）：国标向量 / 签名上下文绑定 / GCM 自洽与篡改 / DH 一致性。"""
from __future__ import annotations

import pytest

from src.crypto.gm_cipher import (CTX_AUTH, CTX_RESULT, AuthenticationError,
                                  _sm4_block_encrypt, gf128_mult, sm2_decrypt,
                                  sm2_ecdh, sm2_encrypt, sm2_generate_keypair,
                                  sm2_sign, sm2_verify, sm3_hash, sm3_kdf,
                                  sm4_gcm_decrypt, sm4_gcm_encrypt)


class TestSM3:
    def test_gbt0004_vector(self):
        """国标 GM/T 0004-2012 测试向量。"""
        assert sm3_hash(b"abc").hex() == \
            "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"
        assert sm3_hash(b"abcd" * 16).hex() == \
            "debe9ff92275b8a138604889c18e5a4d6fdb70e5387e5765293dcba39c0c5732"

    def test_kdf_length_and_determinism(self):
        k1 = sm3_kdf(b"secret", 64)
        k2 = sm3_kdf(b"secret", 64)
        k3 = sm3_kdf(b"secret", 64, salt=b"other")
        assert k1 == k2 and len(k1) == 64 and k1 != k3
        assert sm3_kdf(b"x", 0) == b""


class TestSM2:
    def test_keypair_shape(self):
        pub, priv = sm2_generate_keypair()
        assert len(pub) == 65 and pub[0] == 0x04
        assert len(priv) == 32

    def test_sign_verify_roundtrip(self):
        pub, priv = sm2_generate_keypair()
        sig = sm2_sign(priv, b"authorize session 42", context=CTX_AUTH)
        assert sm2_verify(pub, b"authorize session 42", sig, context=CTX_AUTH)

    def test_cross_context_reuse_rejected(self):
        """陷阱清单：签名必须带用途上下文字符串，防跨协议重用。"""
        pub, priv = sm2_generate_keypair()
        sig = sm2_sign(priv, b"same message", context=CTX_AUTH)
        assert not sm2_verify(pub, b"same message", sig, context=CTX_RESULT)
        assert not sm2_verify(pub, b"other message", sig, context=CTX_AUTH)

    def test_tampered_message_rejected(self):
        pub, priv = sm2_generate_keypair()
        sig = sm2_sign(priv, b"msg", context=CTX_AUTH)
        assert not sm2_verify(pub, b"msG", sig, context=CTX_AUTH)

    def test_wrong_key_rejected(self):
        pub, priv = sm2_generate_keypair()
        pub2, priv2 = sm2_generate_keypair()
        sig = sm2_sign(priv, b"m", context=CTX_AUTH)
        assert not sm2_verify(pub2, b"m", sig, context=CTX_AUTH)

    def test_encrypt_decrypt_roundtrip(self):
        pub, priv = sm2_generate_keypair()
        msg = b"channel key material" * 3
        assert sm2_decrypt(priv, sm2_encrypt(pub, msg)) == msg

    def test_ecdh_both_sides_agree(self):
        """评审 J 项可行性：gmssl 底层点运算可承载协商原语。"""
        pub_a, priv_a = sm2_generate_keypair()
        pub_b, priv_b = sm2_generate_keypair()
        k_a = sm2_ecdh(priv_a, pub_b)
        k_b = sm2_ecdh(priv_b, pub_a)
        assert k_a == k_b and len(k_a) == 32
        pub_c, _ = sm2_generate_keypair()
        assert sm2_ecdh(priv_a, pub_c) != k_a             # 不同对端不同密钥


class TestSM4GCM:
    KEY = bytes.fromhex("0123456789abcdeffedcba9876543210")
    NONCE = bytes.fromhex("000000000000000000000003")

    def test_ecb_gbt0002_vector(self):
        """国标 GM/T 0002-2012 示例 1（锚定块加密原语正确性）。"""
        out = _sm4_block_encrypt(self.KEY, self.KEY)
        assert out.hex() == "681edf34d206965e86b3e94f536e4246"

    def test_gcm_roundtrip_sizes(self):
        for n in (0, 1, 16, 17, 100, 1024):
            pt = bytes(range(256)) * 4
            pt = pt[:n]
            ct, tag = sm4_gcm_encrypt(self.KEY, pt, aad=b"hdr", nonce=self.NONCE)
            assert len(tag) == 16 and len(ct) == n
            assert sm4_gcm_decrypt(self.KEY, ct, tag, aad=b"hdr",
                                   nonce=self.NONCE) == pt

    def test_tamper_ciphertext_detected(self):
        pt = b"payload" * 20
        ct, tag = sm4_gcm_encrypt(self.KEY, pt, aad=b"h", nonce=self.NONCE)
        for i in (0, 15, 31, len(ct) - 1):                # 首块/跨块/末字节
            bad = bytearray(ct)
            bad[i] ^= 1
            with pytest.raises(AuthenticationError):
                sm4_gcm_decrypt(self.KEY, bytes(bad), tag, aad=b"h",
                                nonce=self.NONCE)

    def test_tamper_aad_and_tag_detected(self):
        pt = b"data"
        ct, tag = sm4_gcm_encrypt(self.KEY, pt, aad=b"header", nonce=self.NONCE)
        with pytest.raises(AuthenticationError):
            sm4_gcm_decrypt(self.KEY, ct, tag, aad=b"heades", nonce=self.NONCE)
        bad_tag = bytearray(tag)
        bad_tag[0] ^= 1
        with pytest.raises(AuthenticationError):
            sm4_gcm_decrypt(self.KEY, ct, bytes(bad_tag), aad=b"header",
                            nonce=self.NONCE)

    def test_nonce_mismatch_detected(self):
        pt = b"data"
        ct, tag = sm4_gcm_encrypt(self.KEY, pt, nonce=self.NONCE)
        with pytest.raises(AuthenticationError):
            sm4_gcm_decrypt(self.KEY, ct, tag,
                            nonce=bytes.fromhex("000000000000000000000004"))

    def test_nonce_required(self):
        with pytest.raises(ValueError):
            sm4_gcm_encrypt(self.KEY, b"x")               # nonce 必须显式（F6 口径）

    def test_counter_convention_single_zero_block(self):
        """P2-R1 评审 A 项锚定：单零块密文必须 = E(K, J0+1)（NIST GCM 计数器
        约定）。初版实现密钥流从 E(J0+2) 起（偏移一格，往返自洽但非标准），
        本用例钉死约定防回归。"""
        import os
        key, nonce = os.urandom(16), os.urandom(12)
        j0 = nonce + b"\x00\x00\x00\x01"
        ct, _tag = sm4_gcm_encrypt(key, b"\x00" * 16, nonce=nonce)
        expect = _sm4_block_encrypt(key, (int.from_bytes(j0, "big") + 1).to_bytes(16, "big"))
        assert ct == expect

    def test_rfc8998_sm4_gcm_vector(self):
        """RFC 8998 附录 A.1 SM4-GCM 测试向量（外部标准锚，P2-R1 评审 A 项）。"""
        key = bytes.fromhex("0123456789ABCDEFFEDCBA9876543210")
        iv = bytes.fromhex("00001234567800000000ABCD")
        pt = bytes.fromhex(
            "AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCCDDDDDDDDDDDDDDDD"
            "EEEEEEEEEEEEEEEEFFFFFFFFFFFFFFFFEEEEEEEEEEEEEEEEAAAAAAAAAAAAAAAA")
        aad = bytes.fromhex("FEEDFACEDEADBEEFFEEDFACEDEADBEEFABADDAD2")
        ct_exp = bytes.fromhex(
            "17F399F08C67D5EE19D0DC9969C4BB7D5FD46FD3756489069157B282BB200735"
            "D82710CA5C22F0CCFA7CBF93D496AC15A56834CBCF98C397B4024A2691233B8D")
        tag_exp = bytes.fromhex("83DE3541E4C2B58177E065A9BF7B62EC")
        ct, tag = sm4_gcm_encrypt(key, pt, aad=aad, nonce=iv)
        assert ct == ct_exp and tag == tag_exp
        assert sm4_gcm_decrypt(key, ct, tag, aad=aad, nonce=iv) == pt


class TestGF128:
    # NIST MSB-first 约定下的乘法幺元：X₀=1 即最高位为 1
    ONE = bytes.fromhex("80000000000000000000000000000000")

    def test_identity_and_commutativity(self):
        x = bytes.fromhex("25629347589242761d31f826ba4b757b")
        assert gf128_mult(self.ONE, x) == x and gf128_mult(x, self.ONE) == x
        y = bytes.fromhex("27bfe9d0bcd01f82170c0dbd369e4bea")
        assert gf128_mult(x, y) == gf128_mult(y, x)

    def test_associativity_random(self):
        import os
        for _ in range(20):
            a, b, c = (os.urandom(16) for _ in range(3))
            lhs = gf128_mult(gf128_mult(a, b), c)
            rhs = gf128_mult(a, gf128_mult(b, c))
            assert lhs == rhs

    def test_reduction_polynomial(self):
        # x^127 · 幺元 = x^127（约简多项式路径自洽）
        x127 = bytes.fromhex("01000000000000000000000000000000")
        prod = gf128_mult(x127, bytes.fromhex("80000000000000000000000000000000"))
        assert prod == x127
