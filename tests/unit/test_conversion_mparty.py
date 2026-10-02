"""D5′ m 方转换协议族单测（P8；conversion_m.py）。

覆盖：入口链式掩码 m=2/3/4 精确重构、出口 m 片合成精确回读、
m=2 与旧单片协议数值等价、m 上限（float64 2⁵³ 墙）、合成守卫、
角色校验。CKKS 用 p4-toy 玩具参数（2¹²，链 [30,30,40]）。
说明：掩码按全槽采样（slot_count=2048），测试向量取前 n 槽比对
（尾部槽为 0 值 + 掩码，重构恒为 0——同为精确性的一部分）。
"""
from __future__ import annotations

import os

import pytest

from src.crypto.ckks_ops import CKKSContext, PARAMS_MODE_B, PARAMS_P4_TOY
from src.crypto.secret_sharing import DEFAULT_MODULUS
from src.crypto.secret_sharing_m import reconstruct_n, share_vector_n
from src.protocol.conversion import (FIXED_ONE, ENTRY_MASK_GRID,
                                     POLICY_MODE_B, POLICY_P4_TOY,
                                     entry_decrypt_masked,
                                     entry_make_masked_ct, entry_p2_share,
                                     exit_infernode_compose,
                                     exit_keynode_prepare, from_fixed,
                                     sample_entry_mask, sample_exit_mask,
                                     to_fixed)
from src.protocol.conversion_m import (ComputePartyRole, check_m, entry_add_piece,
                                       exit_compose_m, m_cap)

N = 64


@pytest.fixture(scope="module")
def modeb_ctx(tmp_path_factory):
    """(pub, sec)：mode-b 生产参数（2^15，链 200bit，scale 2^40）。

    精确性测试必须用生产参数化——toy（scale 2^30）下 m 方合成噪声
    ×√(m+1) 存在 ~1e-4/slot 的 ±1 尾巴（实测，P8 工作记录 §精度）；
    mode-b 余量 ×2^10，精确性确定成立。"""
    keys = str(tmp_path_factory.mktemp("p8_modeb_keys"))
    full = CKKSContext(PARAMS_MODE_B)
    full.save_keys(keys, with_secret=True)
    pub = CKKSContext(PARAMS_MODE_B, public_only=True, keys_dir=keys)
    sec = CKKSContext(PARAMS_MODE_B, public_only=False, keys_dir=keys)
    return pub, sec


@pytest.fixture(scope="module")
def toy_ctx(tmp_path_factory):
    """(pub, sec)：p4-toy 玩具参数——仅结构类测试（守卫/角色/网格）。"""
    keys = str(tmp_path_factory.mktemp("p8_toy_keys"))
    full = CKKSContext(PARAMS_P4_TOY)
    full.save_keys(keys, with_secret=True)
    pub = CKKSContext(PARAMS_P4_TOY, public_only=True, keys_dir=keys)
    sec = CKKSContext(PARAMS_P4_TOY, public_only=False, keys_dir=keys)
    return pub, sec


def _ulp_close(rec, fixed, redline=4):
    """±redline ulp 口径（P4 δ≤5 记录内；环算术相对 y₁ 精确）。"""
    Mod = 1 << 64
    return all(abs((r - f + Mod // 2) % Mod - Mod // 2) <= redline
               for r, f in zip(rec, fixed))


def _parties(pub, m):
    return [ComputePartyRole(pub, index=i + 1, m=m, is_anchor=(i == 0))
            for i in range(m)]


def _vals():
    return [(os.urandom(4)[0] / 32 - 4.0) for _ in range(N)]


def _run_entry(pub, sec, parties, vals):
    """入口协议：链式加片（P2..Pm→P1）→ P0 白名单①解密 → 份额重构（前 N 槽）。"""
    fixed = [to_fixed(v) for v in vals]
    req = os.urandom(16)
    masked = pub.encrypt_vector(vals)
    order = parties[1:] + parties[:1]
    for p in order:
        masked = p.entry_apply_piece(masked, req)
    y = entry_decrypt_masked(sec, masked)
    shares = [p.entry_own_share(req, y=y if p.is_anchor else None)
              for p in parties]
    return fixed, [s[:N] for s in shares]


def _run_exit(pub, sec, parties, fixed):
    """出口协议：m 方结果分享 → 各片 (zᵢ, Enc(sᵢ)) → 合成方密文域合成。"""
    shares = share_vector_n(fixed, len(parties))
    req = os.urandom(16)
    pieces = []
    for p, a in zip(parties, shares):
        p.set_result_share(req, a)
        pieces.append(p.exit_piece(req))
    return exit_compose_m(pub, pieces)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_entry_chain_exact_reconstruction(modeb_ctx, m):
    pub, sec = modeb_ctx
    parties = _parties(pub, m)
    for _ in range(3):
        vals = _vals()
        fixed, shares = _run_entry(pub, sec, parties, vals)
        assert _ulp_close(reconstruct_n(shares), fixed)   # ±ulp 口径；环算术相对 y₁ 精确


@pytest.mark.parametrize("m", [2, 3, 4])
def test_exit_compose_exact_roundtrip(modeb_ctx, m):
    pub, sec = modeb_ctx
    parties = _parties(pub, m)
    for _ in range(3):
        vals = _vals()
        fixed = [to_fixed(v) for v in vals]
        ct = _run_exit(pub, sec, parties, fixed)
        dec = sec.decrypt(ct)
        rec = [int(round(v * FIXED_ONE)) % DEFAULT_MODULUS for v in dec[:N]]
        assert _ulp_close(rec, fixed)                # ±ulp 口径回读
        for v, f in zip(vals, rec):
            assert from_fixed(f) == pytest.approx(v, abs=1e-9)


def test_m2_equivalent_to_legacy_single_piece(modeb_ctx):
    """m=2 与旧 2 方单片协议数值等价（重构精确一致——行为等价，非字节等价）。"""
    pub, sec = modeb_ctx
    vals = [0.5, -1.25, 2.0, 3.75, -0.125, 1.0]
    k = len(vals)
    fixed = [to_fixed(v) for v in vals]

    # 旧协议：P2 整片 r → P0 解密 → (y₁, −r)
    ct_leg = pub.encrypt_vector(vals)
    masked, r = entry_make_masked_ct(pub, ct_leg)
    y1 = entry_decrypt_masked(sec, masked)
    leg_entry = [(a + b) % DEFAULT_MODULUS
                 for a, b in zip(y1[:k], entry_p2_share(r)[:k])]
    # 新协议 m=2：P1/P2 各持一片
    parties = _parties(pub, 2)
    _, new_entry = _run_entry(pub, sec, parties, vals)
    assert _ulp_close(reconstruct_n(new_entry)[:k], fixed)
    assert _ulp_close(leg_entry, fixed)              # 双路径同一口径重构

    # 出口对照
    a1, a2 = share_vector_n(fixed, 2)
    z, enc_s = exit_keynode_prepare(pub, a1[:k])
    ct_leg_out = exit_infernode_compose(pub, enc_s, z, a2[:k])
    dec_leg = [int(round(v * FIXED_ONE)) % DEFAULT_MODULUS
               for v in sec.decrypt(ct_leg_out)[:k]]
    ct_m_out = _run_exit(pub, sec, parties, fixed)
    dec_m = [int(round(v * FIXED_ONE)) % DEFAULT_MODULUS
             for v in sec.decrypt(ct_m_out)[:k]]
    assert _ulp_close(dec_leg, fixed)
    assert _ulp_close(dec_m, fixed)


def test_entry_piece_grid_uniform():
    """入口片 2¹¹ 网格 + 出口片偏置窗（单片窗不随 m 变——泄漏界不变）。"""
    r = sample_entry_mask(32, POLICY_P4_TOY)
    assert all(ri % ENTRY_MASK_GRID == 0 for ri in r)
    s = sample_exit_mask(32, POLICY_P4_TOY)
    assert all(POLICY_P4_TOY.exit_mask_lo <= si < POLICY_P4_TOY.exit_mask_hi
               for si in s)


def test_m_cap_float64_wall():
    """m 上限由 float64 2⁵³ 精确域反推：mode-b 入口窗 2⁴⁹ ⇒ 15。"""
    assert m_cap(POLICY_MODE_B) == 15
    assert m_cap(POLICY_P4_TOY) == 8191
    check_m(POLICY_MODE_B, 15)
    check_m(POLICY_P4_TOY, 8191)
    with pytest.raises(Exception):
        check_m(POLICY_MODE_B, 16)
    with pytest.raises(Exception):
        check_m(POLICY_P4_TOY, 8192)
    with pytest.raises(Exception):
        check_m(POLICY_MODE_B, 1)


def test_exit_compose_guards(toy_ctx):
    """合成守卫：片数 <2、长度不一致、越界 z 一律拒绝。"""
    pub, _sec = toy_ctx
    with pytest.raises(Exception):
        exit_compose_m(pub, [([1, 2, 3], b"\x00")])
    parties = _parties(pub, 2)
    fixed = [to_fixed(0.5)] * 8
    shares = share_vector_n(fixed, 2)
    req = os.urandom(16)
    pieces = []
    for p, a in zip(parties, shares):
        p.set_result_share(req, a)
        pieces.append(p.exit_piece(req))
    bad = [(pieces[0][0] + [123], pieces[0][1]), pieces[1]]
    with pytest.raises(Exception):
        exit_compose_m(pub, bad)
    huge = [([(1 << 63)] * 8, pieces[0][1]), pieces[1]]
    with pytest.raises(Exception):
        exit_compose_m(pub, huge)


def test_role_validation(toy_ctx):
    pub, sec = toy_ctx
    with pytest.raises(ValueError):
        ComputePartyRole(sec, index=1, m=2)      # 计算方禁持私钥上下文
    with pytest.raises(ValueError):
        ComputePartyRole(pub, index=3, m=2)      # index 越界
    anchor = ComputePartyRole(pub, index=1, m=2, is_anchor=True)
    ct = pub.encrypt_vector([0.5] * 16)
    masked = anchor.entry_apply_piece(ct, b"\x01" * 16)
    with pytest.raises(Exception):
        anchor.entry_own_share(b"\x01" * 16)     # 锚点缺 y 显式报错
    assert masked is not None
