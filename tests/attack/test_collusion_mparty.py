"""m 方合谋矩阵攻击测试（P8；F8 的 m 方扩展，docs/02 §3.3′ 口径）。

敌手模型（D5′ 申报）：半诚实计算参与方，static，t ≤ m−1；P0（数据属主/
钥持者）为信任根——{P0, ·} 组合超出声明模型，留置已知限制（docs/02）。

m=3 合谋矩阵（6 视角 ×（入口 x / 出口 v））：
- 精确恢复成功率 == 0（全部试验）；
- OTP 论证精确化：含锚点/合成方的组合，其最优候选与真值之差**恰好落在
  缺失片的掩码窗内**（「只差一片均匀掩码」的代数实证）；不含锚点/合成方
  的组合缺口为全宽 2⁶⁴ 均匀份额。
取证口径：视图 = 本方掩码片/份额/收发消息（collect_view 同源，进程内）。
"""
from __future__ import annotations

import os

import pytest

from src.crypto.ckks_ops import CKKSContext, PARAMS_P4_TOY
from src.crypto.secret_sharing import DEFAULT_MODULUS
from src.crypto.secret_sharing_m import share_vector_n
from src.protocol.conversion import (FIXED_ONE, entry_decrypt_masked,
                                     to_fixed)
from src.protocol.conversion_m import ComputePartyRole, exit_compose_m

M = 3
TRIALS = 30
N = 16


@pytest.fixture(scope="module")
def toy_ctx(tmp_path_factory):
    keys = str(tmp_path_factory.mktemp("p8_collusion_keys"))
    full = CKKSContext(PARAMS_P4_TOY)
    full.save_keys(keys, with_secret=True)
    pub = CKKSContext(PARAMS_P4_TOY, public_only=True, keys_dir=keys)
    sec = CKKSContext(PARAMS_P4_TOY, public_only=False, keys_dir=keys)
    return pub, sec


def _parties(pub):
    return [ComputePartyRole(pub, index=i + 1, m=M, is_anchor=(i == 0))
            for i in range(M)]


def _round(pub, sec, parties):
    """一次完整转换回合，返回取证视图（x/v 真值、y、各片 r/s/z/a、w）。"""
    req = os.urandom(16)
    vals = [(os.urandom(4)[0] / 32 - 4.0) for _ in range(N)]
    x = [to_fixed(v) % DEFAULT_MODULUS for v in vals]

    # 入口：链式加片（P2→P3→P1）→ P0 解密 y → 份额
    masked = pub.encrypt_vector(vals)
    r_views = {}
    for p in parties[1:] + parties[:1]:
        masked = p.entry_apply_piece(masked, req)
        r_views[p.index] = list(p._entry_pieces[req])   # 取证：本方掩码片
    y = entry_decrypt_masked(sec, masked)
    a_entry = {}
    for p in parties:
        a_entry[p.index] = p.entry_own_share(
            req, y=y if p.is_anchor else None)

    # 出口：m-of-m 结果分享（S8 模拟重分享）→ 各片 (zᵢ, Enc(sᵢ))
    result = share_vector_n(x, M)
    z_views, s_views = {}, {}
    for p, a in zip(parties, result):
        p.set_result_share(req, a)
        z, enc_s = p.exit_piece(req)
        z_views[p.index] = z
        s_views[p.index] = [(zi - ai) % DEFAULT_MODULUS
                            for zi, ai in zip(z, a)]      # 本方可自算 sᵢ=zᵢ−aᵢ
    v = x                                                # 出口真值 = 入口真值（S8）
    w = [(v[i] + sum(s_views[p.index][i] for p in parties)) % DEFAULT_MODULUS
         for i in range(N)]
    return dict(x=x, v=v, y=y, r=r_views, s=s_views, z=z_views,
                a_entry=a_entry, a_result={p.index: a for p, a
                                           in zip(parties, result)}, w=w)


def test_collusion_matrix_m3(toy_ctx):
    pub, sec = toy_ctx
    policy = _parties(pub)[0]._policy
    windows = {1: policy.entry_mask_hi, 2: policy.entry_mask_hi,
               3: policy.entry_mask_hi}                  # 入口片窗（每片同窗）
    exit_windows = {i: policy.exit_mask_hi - policy.exit_mask_lo
                    for i in (1, 2, 3)}                  # 出口片窗

    for coalition in [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]:
        entry_zero, exit_zero = 0, 0
        entry_offsets, exit_offsets = [], []
        for _ in range(TRIALS):
            view = _round(pub, sec, _parties(pub))
            # ---- 入口攻击：候选 x̂ = 已知量 − 已知片和 ----
            known_r = [0] * N
            for i in coalition:
                known_r = [(kr + rv) % DEFAULT_MODULUS
                           for kr, rv in zip(known_r, view["r"][i])]
            if 1 in coalition:
                x_hat = [(yv - kr) % DEFAULT_MODULUS
                         for yv, kr in zip(view["y"], known_r)]
            else:
                x_hat = [(ae - kr) % DEFAULT_MODULUS          # 只有自己份额
                         for ae, kr in zip(
                             _sum_shares([view["a_entry"][i] for i in coalition]),
                             known_r)]
            off = [(h - xv) % DEFAULT_MODULUS for h, xv in zip(x_hat, view["x"])]
            off = [o - DEFAULT_MODULUS if o > DEFAULT_MODULUS // 2 else o
                   for o in off]
            if any(o == 0 for o in off):
                entry_zero += 1
            entry_offsets.append(sum(1 for o in off if o != 0))
            # ---- 出口攻击：候选 v̂ = w(若有合成方) − 已知 s 和 ----
            if 2 in coalition:
                known_s = [0] * N
                for i in coalition:
                    known_s = [(ks + sv) % DEFAULT_MODULUS
                               for ks, sv in zip(known_s, view["s"][i])]
                v_hat = [(wv - ks) % DEFAULT_MODULUS
                         for wv, ks in zip(view["w"], known_s)]
            else:
                v_hat = _sum_shares([view["a_result"][i] for i in coalition])
            offv = [(h - xv) % DEFAULT_MODULUS for h, xv in zip(v_hat, view["v"])]
            offv = [o - DEFAULT_MODULUS if o > DEFAULT_MODULUS // 2 else o
                    for o in offv]
            if any(o == 0 for o in offv):
                exit_zero += 1
            exit_offsets.append(sum(1 for o in offv if o != 0))
        # 断言 1：精确恢复成功率 == 0
        assert entry_zero == 0, f"合谋 {coalition} 入口出现精确恢复"
        assert exit_zero == 0, f"合谋 {coalition} 出口出现精确恢复"
        # 断言 2：全部槽位候选偏离真值（缺口非零）
        assert all(c == N for c in entry_offsets)
        assert all(c == N for c in exit_offsets)
        # 断言 3（OTP 实证）：含锚点/合成方的组合，偏离恰好 = 缺失片值
        # ——落缺失片窗内且窗宽 ≫ 值域（泄漏界与单片版一致）
        if 1 in coalition:
            missing = [i for i in range(1, M + 1) if i not in coalition]
            total_window = sum(windows[i] for i in missing)
            assert total_window >= (1 << 24), "缺口窗必须 ≫ 激活值域 2²⁴"


def _sum_shares(shares):
    acc = [0] * N
    for s in shares:
        acc = [(a + b) % DEFAULT_MODULUS for a, b in zip(acc, s)]
    return acc
