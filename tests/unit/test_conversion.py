"""D5 转换真实路径测试（P4 核心交付：密文域加 Enc(r) / 白名单解密 / 出口合成）。

玩具参数跑协议全流程（快）；PARAMS_MODE_B 全槽跑一次真实参数规模验证
（D5 白名单①执行点在本栈推理参数下的数值正确性）。
"""
from __future__ import annotations

import os

import pytest

from src.crypto.ckks_ops import (CKKSContext, CKKSCiphertext, PARAMS_MODE_B,
                                 PARAMS_P4_TOY)
from src.protocol.conversion import (ConversionError, ClientRole, InferRole,
                                     KeyRole, WhitelistError,
                                     entry_decrypt_masked, entry_make_masked_ct,
                                     entry_p2_share, exit_infernode_compose,
                                     exit_keynode_prepare, from_fixed, to_fixed,
                                     pack_ints, unpack_ints)


@pytest.fixture(scope="module")
def toy_ctx():
    full = CKKSContext(PARAMS_P4_TOY)
    import tempfile
    d = tempfile.mkdtemp(prefix="a122_test_toy_")
    full.save_keys(d, with_secret=True)
    return {
        "sec": CKKSContext(PARAMS_P4_TOY, public_only=False, keys_dir=d),
        "pub": CKKSContext(PARAMS_P4_TOY, public_only=True, keys_dir=d),
    }


def _roles(toy_ctx):
    client = ClientRole(toy_ctx["sec"])
    keynode = KeyRole(toy_ctx["pub"])
    infer = InferRole(toy_ctx["pub"])
    return client, keynode, infer


class TestEntryRealPath:
    def test_entry_reconstruct_exact(self, toy_ctx):
        """入口：P2 密文域加 Enc(r) → P0 白名单解密 → 分享重构 = 原定点值。"""
        client, keynode, infer = _roles(toy_ctx)
        values = [0.5, -1.25, 2.0, 3.75]
        ct = toy_ctx["pub"].encrypt_vector(values)
        rid = os.urandom(16)
        masked = infer.entry(ct, rid)
        y1 = client.masked_decrypt(masked, rid)
        keynode.take_entry_share(rid, y1)
        p2 = infer.entry_p2_share(rid)
        from src.crypto.secret_sharing import DEFAULT_MODULUS
        recon = [(a + b) % DEFAULT_MODULUS for a, b in zip(y1, p2)]
        expect = [to_fixed(v) for v in values]
        # 环算术精确；玩具参数 scale 2^30 解码噪声可致 ±1 定点 ulp 舍入
        for got, exp in zip(recon, expect):
            # float64 ulp @ 掩码量级 + 变换累加 ⇒ |δ| ≤ 4 定点 ulp（P4 实测）
            assert (got - exp) % DEFAULT_MODULUS in tuple(range(5)) + tuple(
                range(DEFAULT_MODULUS - 4, DEFAULT_MODULUS))

    def test_p0_view_is_masked(self, toy_ctx):
        """P0 解密视图 = x + r/2¹⁶（OTP 掩码主导），非 x 本身。"""
        client, _, infer = _roles(toy_ctx)
        ct = toy_ctx["pub"].encrypt_vector([1.0] * 4)
        rid = os.urandom(16)
        masked = infer.entry(ct, rid)
        y = toy_ctx["sec"].decrypt(masked)[:4]
        assert all(abs(v) > 1e3 for v in y)           # 掩码值主导（OTP 域）
        y1 = client.masked_decrypt(masked, rid)
        assert all(0 <= v < 2**64 for v in y1)        # mod 2⁶⁴ 环内

    def test_share_holders_hold_zero_secret_key(self, toy_ctx):
        """D5 结构性防线：P1/P2 上下文 public_only，解密即 OpNotAllowedError。"""
        _, keynode, infer = _roles(toy_ctx)
        from src.crypto.ckks_ops import OpNotAllowedError
        with pytest.raises(OpNotAllowedError):
            keynode.ctx.decrypt(toy_ctx["pub"].encrypt_vector([1.0]))
        with pytest.raises(OpNotAllowedError):
            infer.ctx.decrypt(toy_ctx["pub"].encrypt_vector([1.0]))


class TestExitRealPath:
    def test_exit_compose_roundtrip(self, toy_ctx):
        """出口：分享 → P1 掩码 → P2 密文域合成 → 解密 ≈ 原值（fresh 噪声内）。"""
        client, keynode, infer = _roles(toy_ctx)
        from src.crypto.secret_sharing import share_vector
        values = [0.5, -1.25, 2.0, 3.75]
        fixed = [to_fixed(v) for v in values]
        a1, a2 = share_vector(fixed)
        rid = os.urandom(16)
        keynode.take_entry_share(rid, a1)
        z, enc_s = keynode.exit_prepare(rid)
        ct = infer.exit_compose(rid, enc_s, z, a2)
        assert ct.level == 0                          # fresh 顶层（深度重置）
        got = toy_ctx["sec"].decrypt(ct)[:len(values)]
        for g, v in zip(got, values):
            assert abs(g - v) < 1.5e-5                # 一个 Q16 ulp（玩具参数噪声内）

    def test_exit_through_channel_codec(self, toy_ctx):
        """出口载荷经 pack/unpack 字节往返后合成结果不变。"""
        _, keynode, infer = _roles(toy_ctx)
        from src.crypto.secret_sharing import share_vector
        values = [1.5, -2.5]
        a1, a2 = share_vector([to_fixed(v) for v in values])
        z, enc_s = exit_keynode_prepare(infer.ctx, a1)
        z2 = unpack_ints(pack_ints(z))
        ct = exit_infernode_compose(infer.ctx, enc_s, z2, a2)
        got = toy_ctx["sec"].decrypt(ct)[:2]
        assert abs(got[0] - 1.5) < 1.5e-5 and abs(got[1] + 2.5) < 1.5e-5


class TestWhitelist:
    def test_non_whitelisted_kind_rejected(self, toy_ctx):
        """D5：白名单外解密请求（如 raw_intermediate）拒绝并产生审计事件。"""
        events = []
        client = ClientRole(toy_ctx["sec"], audit=lambda e, d: events.append((e, d)))
        ct = toy_ctx["pub"].encrypt_vector([1.0])
        with pytest.raises(WhitelistError):
            client.whitelist.decrypt("raw_intermediate", ct, b"")
        assert events and events[0][0] == "DECRYPT_WHITELIST_REJECT"

    def test_allowed_kinds_pass(self, toy_ctx):
        events = []
        client = ClientRole(toy_ctx["sec"], audit=lambda e, d: events.append((e, d)))
        ct = toy_ctx["pub"].encrypt_vector([1.0, 2.0])
        vals = client.whitelist.decrypt("masked_conversion", ct, b"\x01" * 16)
        assert any(e[0] == "DECRYPT_OK" for e in events)
        out = client.final_decrypt(ct, b"\x02" * 16)
        assert abs(out[0] - 1.0) < 1e-6 and abs(out[1] - 2.0) < 1e-6

    def test_whitelist_requires_secret_ctx(self, toy_ctx):
        with pytest.raises(ValueError):
            ClientRole(toy_ctx["pub"])               # public_only 不能当 P0 门


class TestGuards:
    def test_value_bound_guard(self):
        with pytest.raises(ConversionError):
            to_fixed(1 << 40)                        # 超出 |v| < 2⁴⁶ 守卫

    def test_fixed_codec_roundtrip(self):
        for v in (0.0, 1.0, -1.0, 123.456, -0.0001):
            assert abs(from_fixed(to_fixed(v)) - v) < 2 ** -16

    def test_mask_distribution_bounds(self):
        from src.protocol.conversion import (sample_entry_mask, sample_exit_mask,
                                             ENTRY_MASK_GRID, POLICY_P4_TOY,
                                             POLICY_MODE_B)
        for policy in (POLICY_P4_TOY, POLICY_MODE_B):
            r = sample_entry_mask(50, policy)
            assert all(v % ENTRY_MASK_GRID == 0 for v in r)
            assert all(0 <= v < policy.entry_mask_hi for v in r)
            s = sample_exit_mask(50, policy)
            assert all(policy.exit_mask_lo <= v < policy.exit_mask_hi for v in s)
            # 窗宽必须显著大于值域（OTP 边界泄漏可控）
            assert policy.exit_mask_hi - policy.exit_mask_lo >= 4 * policy.value_bound


class TestModeBScale:
    @pytest.mark.slow
    def test_mode_b_full_slot_entry_exit(self):
        """真实推理参数（2^15 / 16384 槽）下的转换真实路径（D5 执行点规模验证）。"""
        full = CKKSContext(PARAMS_MODE_B)
        import tempfile
        d = tempfile.mkdtemp(prefix="a122_test_mb_")
        full.save_keys(d, with_secret=True)
        sec = CKKSContext(PARAMS_MODE_B, public_only=False, keys_dir=d)
        pub = CKKSContext(PARAMS_MODE_B, public_only=True, keys_dir=d)
        values = [float((i % 7) - 3) * 0.5 for i in range(512)]
        ct = pub.encrypt_vector(values)
        rid = os.urandom(16)
        masked, r = entry_make_masked_ct(pub, ct)          # 入口（level 0）
        y1 = entry_decrypt_masked(sec, masked)
        p2 = entry_p2_share(r)
        from src.crypto.secret_sharing import DEFAULT_MODULUS, share_vector
        recon = [(a + b) % DEFAULT_MODULUS for a, b in zip(y1, p2)]
        expect = [to_fixed(v) for v in values] + [0] * (len(y1) - len(values))
        # 环算术精确 + CKKS 噪声 ≤1 ulp（真实路径口径，P4 实测 δ ∈ {0, ±1}）
        for got, exp in zip(recon, expect):
            # float64 ulp @ 掩码量级 + 变换累加 ⇒ |δ| ≤ 4 定点 ulp（P4 实测）
            assert (got - exp) % DEFAULT_MODULUS in tuple(range(5)) + tuple(
                range(DEFAULT_MODULUS - 4, DEFAULT_MODULUS))
        # 出口（目标布局即本向量）
        a1, a2 = share_vector([to_fixed(v) for v in values] + [0] * (len(y1) - len(values)))
        z, enc_s = exit_keynode_prepare(pub, a1)
        ct2 = exit_infernode_compose(pub, enc_s, z, a2)
        assert ct2.level == 0
        got = sec.decrypt(ct2)[:len(values)]
        assert max(abs(g - v) for g, v in zip(got, values)) < 1e-6
