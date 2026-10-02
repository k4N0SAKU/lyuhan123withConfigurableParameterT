"""m 方合谋矩阵全谱补测（审计补全：t=4~14 执行级覆盖）。

背景：tests/attack/test_collusion_mparty.py（P8）实测到 t=3（m=4 的三个
size-3 最坏组合），m=5~15（t=4~14）原由代数论证覆盖（P8 工作记录§覆盖
矩阵）。本文件把边界合谋（size = m−1，恰缺一片）的三个最坏组合在每个
m=5~15 档实测——闭掉「t=4~14 仅论证」的缺口：

- 缺锚点 P1：coalition = {2..m}（持 y 路径缺失——只能拼份额）；
- 缺合成方 P2：coalition = {1,3..m}（入口全知但出口只见份额）；
- 缺最远片 Pm：coalition = {1..m−1}（含锚点+合成方的最坏 full 视角）。

驱动逻辑与断言语义与 test_collusion_mparty.py 完全同源（直接 import 其
_round/_parties/_sum_shares，避免复制漂移）；每组合 × 30 试验 ×（入口/
出口）：
- 断言 1：精确恢复成功率 == 0；
- 断言 2：缺失片窗之和 ≥ 2²⁴（OTP 缺片代数实证）。

边界情形与 m 无关的论证由本测试在每档 m 上锚定：size-(m−1) 合谋恰缺
一片均匀掩码，方程结构 x̂ = y − Σ_{i∈C} rᵢ 与 m 无关（docs/02 A.3）。
"""
from __future__ import annotations

import os

import pytest

from src.crypto.secret_sharing import DEFAULT_MODULUS
from tests.attack.test_collusion_mparty import (
    N, TRIALS, _parties, _round, _sum_shares, toy_ctx,
)

M_RANGE = list(range(int(os.environ.get("P8_FULL_M_MIN", 5)),       # m=5..15
                     int(os.environ.get("P8_FULL_M_MAX", 16))))      #（mode-b 全域；toy cap 8191 内）
WORST = {
    m: (
        tuple(range(2, m + 1)),       # 缺锚点 P1
        tuple([1] + list(range(3, m + 1))),   # 缺合成方 P2
        tuple(range(1, m)),           # 缺最远片 Pm（含锚点+合成方）
    )
    for m in M_RANGE
}


@pytest.mark.parametrize("m", M_RANGE)
def test_collusion_worst_full_spectrum(toy_ctx, m):
    """每档 m 的三个最坏 size-(m−1) 组合：入口/出口精确恢复率必须为 0。"""
    pub, sec = toy_ctx
    policy = _parties(pub, m)[0]._policy
    entry_window = policy.entry_mask_hi          # 入口片窗（每片同窗）
    exit_window = policy.exit_mask_hi - policy.exit_mask_lo

    for coalition in WORST[m]:
        assert len(coalition) == m - 1, "最坏组合必须恰为 size m−1"
        entry_zero, exit_zero = 0, 0
        for _ in range(TRIALS):
            view = _round(pub, sec, _parties(pub, m), m)
            # ---- 入口攻击：候选 x̂ = 已知量 − 已知片和 ----
            known_r = [0] * N
            for i in coalition:
                known_r = [(kr + rv) % DEFAULT_MODULUS
                           for kr, rv in zip(known_r, view["r"][i])]
            if 1 in coalition:
                x_hat = [(yv - kr) % DEFAULT_MODULUS
                         for yv, kr in zip(view["y"], known_r)]
            else:                                 # 缺锚点：只见己方份额
                x_hat = _sum_shares([view["a_entry"][i] for i in coalition])
            off = [(h - xv) % DEFAULT_MODULUS for h, xv in zip(x_hat, view["x"])]
            off = [o - DEFAULT_MODULUS if o > DEFAULT_MODULUS // 2 else o
                   for o in off]
            if any(o == 0 for o in off):
                entry_zero += 1
            # ---- 出口攻击：候选 v̂ = w(若有合成方) − 已知 s 和 ----
            if 2 in coalition:
                known_s = [0] * N
                for i in coalition:
                    known_s = [(ks + sv) % DEFAULT_MODULUS
                               for ks, sv in zip(known_s, view["s"][i])]
                v_hat = [(wv - ks) % DEFAULT_MODULUS
                         for wv, ks in zip(view["w"], known_s)]
            else:                                 # 缺合成方：只见结果份额
                v_hat = _sum_shares([view["a_result"][i] for i in coalition])
            offv = [(h - xv) % DEFAULT_MODULUS for h, xv in zip(v_hat, view["v"])]
            offv = [o - DEFAULT_MODULUS if o > DEFAULT_MODULUS // 2 else o
                    for o in offv]
            if any(o == 0 for o in offv):
                exit_zero += 1
        # 断言 1：精确恢复成功率 == 0（全部试验）
        assert entry_zero == 0, f"m={m} 合谋 {coalition} 入口出现精确恢复"
        assert exit_zero == 0, f"m={m} 合谋 {coalition} 出口出现精确恢复"
        # 断言 2（OTP 实证）：缺口窗 ≫ 激活值域 2²⁴（缺失片均匀窗之和）
        missing = [i for i in range(1, m + 1) if i not in coalition]
        assert len(missing) == 1, "size-(m−1) 组合必须恰缺一片"
        assert sum(entry_window for _ in missing) >= (1 << 24)
        assert exit_window >= (1 << 24)
