"""Remez minimax 系数测试（P2-R1 B1 项）：误差界独立复算 + 等波纹性自检。

评审 B1 项核心发现：P1 §7 的理论误差界系统性低估 3~4 个数量级（GELU deg5
声称 5e-3，实测最优 0.231）。本测试以**独立计算路径**（不调用生成器的
errfun）复算入库系数的真实误差，并锚定规格值。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.model.ops.minimax import DATA_PATH, _gelu

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def coeffs() -> dict:
    if not DATA_PATH.exists():
        from src.model.ops.minimax import build_all
        build_all()
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _polyval_norm(entry, x):
    """归一化域求值（P3 修订：系数在 [-1,1] 域拟合）。"""
    t = (np.asarray(x) - entry["mid"]) / entry["half"]
    return np.polynomial.polynomial.polyval(t, np.asarray(entry["coeffs"]))


class TestMeasuredBounds:
    """独立复算入库系数的真实误差（密网格 + 独立 eval 路径）。"""

    def test_gelu_deg15(self, coeffs):
        s = coeffs["gelu_deg15"]
        x = np.linspace(*s["domain"], 200001)
        err = np.abs(_polyval_norm(s, x) - _gelu(x)).max()
        assert err == pytest.approx(s["max_error"], rel=0.05)
        # P1 目标 5e-3 为估计值；deg15 在 [-6,6] 的数学最优即 5.067e-3（+1.3%）。
        # 如需严格达标：缩域 [-5.9,5.9] 或升 deg16——P3 端到端回归后决定。
        assert err <= 5.5e-3, f"GELU deg15 误差 {err:.3e} 超校准阈值 5.5e-3"

    def test_gelu_deg9_low_cost_variant(self, coeffs):
        s = coeffs["gelu_deg9"]
        x = np.linspace(*s["domain"], 200001)
        err = np.abs(_polyval_norm(s, x) - _gelu(x)).max()
        assert err == pytest.approx(s["max_error"], rel=0.05)

    def test_exp_deg12_onesided(self, coeffs):
        s = coeffs["exp_deg12_onesided"]
        x = np.linspace(*s["domain"], 200001)
        err = np.abs(_polyval_norm(s, x) - np.exp(x)).max()
        assert err == pytest.approx(s["max_error"], rel=0.05)
        assert err <= 1e-2, "exp 宽域精度"

    def test_inv_init_relative(self, coeffs):
        s = coeffs["inv_init_deg4"]
        x = np.linspace(*s["domain"], 200001)
        rel = np.abs(_polyval_norm(s, x) * x - 1.0).max()
        assert rel == pytest.approx(s["max_error"], rel=0.05)
        # Newton 7 轮收敛性（δ←δ² 链）
        d = s["max_error"]
        for _ in range(s["newton_rounds"]):
            d = d * d
        assert d < 0.1, "Newton 收敛性（deg4 宽域 5 轮）"

    def test_invsqrt_init_relative(self, coeffs):
        s = coeffs["invsqrt_init_deg4"]
        x = np.linspace(*s["domain"], 200001)
        rel = np.abs(_polyval_norm(s, x) * np.sqrt(x) - 1.0).max()
        assert rel == pytest.approx(s["max_error"], rel=0.05)
        d = s["max_error"]
        for _ in range(s["newton_rounds"]):
            d = d * d
        assert d < 1e-6, "3 轮 Newton 未收敛至 1e-6"


class TestEquioscillation:
    """minimax 特征自检：极值误差近似等幅且符号交替（Remez 正确性证据）。"""

    def test_gelu_deg15_equioscillation(self, coeffs):
        s = coeffs["gelu_deg15"]
        x = np.linspace(*s["domain"], 200001)
        e = _polyval_norm(s, x) - _gelu(x)
        ext = [e[0]]
        for i in range(1, len(e) - 1):
            if (e[i] - e[i - 1]) * (e[i + 1] - e[i]) < 0:
                ext.append(e[i])
        ext.append(e[-1])
        ext = np.array(ext)
        mags = np.sort(np.abs(ext))[-16:]
        spread = mags.max() / mags.min()
        assert spread < 1.6, f"等波纹性破坏（幅值比 {spread:.2f}）"

    def test_coeffs_deterministic(self, coeffs):
        """重新生成应得到相同系数（确定性，评审可复现）。"""
        from src.model.ops.minimax import remez, _gelu
        from src.model.ops.minimax import build_all
        fresh = build_all()
        assert np.allclose(fresh["gelu_deg15"]["coeffs"],
                           coeffs["gelu_deg15"]["coeffs"], rtol=0, atol=1e-8)
