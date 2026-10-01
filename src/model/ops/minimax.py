"""Remez（minimax）多项式逼近系数生成器（P2-R1 补做评审 B1 项）。

P1 §7 承诺"P2 用 Remez 数值生成非线性近似系数入库"，初版以理论估计占位
（评审 B1 指出被静默丢弃）。本模块用 numpy 实现经典 Remez 交换算法：

- 加权 minimax：min_p max_x |p(x) − t(x)|·w(x)；
  绝对误差 w=1（GELU/exp）；相对误差 w=1/t（inv/inv-sqrt 初值，即
  |p·s−1| / |p·√v−1| 口径）。
- 迭代：切比雪夫极值点初始化 → 解 (n+2) 点线性系统（**切比雪夫基**，数值
  稳定）→ 稠密采样找误差极值 → 交换节点（**符号交替**保持），直到 E 收敛。

产出 `data/minimax/poly_approx.json`（系数升幂、定义域、实测 max_error、
误差类型、Newton 轮数），由 `tests/unit/test_minimax.py` 独立复算锚定。

P2-R1 核心发现：P1 §7 的理论误差界系统性低估 3~4 个数量级——GELU deg5
@[-6,6] 真实最优 0.231（P1 声称 5e-3）；exp [-8,8] deg7 真实 ~20（P1 声称
3e-3；切比雪夫系数 2I_k(8) 衰减远慢于 P1 假设）。最终配置以实测为准。

用法：python -m src.model.ops.minimax   # 重新生成全部系数
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "minimax" / "poly_approx.json"


def _gelu(x: np.ndarray) -> np.ndarray:
    """精确 GELU：x·Φ(x) = x·(1+erf(x/√2))/2。"""
    return x * 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def remez(t: Callable[[np.ndarray], np.ndarray], lo: float, hi: float, degree: int,
          weight: Callable[[np.ndarray], np.ndarray] | None = None,
          iterations: int = 100, samples: int = 20001) -> Tuple[np.ndarray, float]:
    """单区间 Remez 交换算法（切比雪夫基求解，数值稳定）。

    返回 (升幂**单项式**系数, 实测 max |(p−t)·w|)。weight=None 即绝对误差。

    实现要点：
    - 求解在 Chebyshev 基上进行（[-1,1] 缩放），避免高次单项式 Vandermonde
      病态；终值经 cheb2poly + Polynomial 组合转回原域单项式系数；
    - 交换步保持**符号交替**（同号连续段取 |e| 最大者）——初版不保交替致
      系统退化（P2-R1 修复记录）。"""
    u = lambda x: (2.0 * x - (lo + hi)) / (hi - lo)        # noqa: E731

    def cheb_basis(x: np.ndarray) -> np.ndarray:
        xu = u(np.atleast_1d(x))
        T = [np.ones_like(xu), xu]
        for j in range(2, degree + 1):
            T.append(2.0 * xu * T[-1] - T[-2])
        return np.stack(T[:degree + 1], axis=1)

    def errfun(cs, x):
        e = np.asarray(cheb_basis(x) @ cs) - t(np.atleast_1d(x))
        if weight is not None:
            e = e * weight(np.atleast_1d(x))
        return np.atleast_1d(e)

    k = np.arange(degree + 2)
    nodes = (lo + hi) / 2 + (hi - lo) / 2 * np.cos(np.pi * k / (degree + 1))
    grid = np.linspace(lo, hi, samples)

    cs = np.zeros(degree + 1)
    E = np.inf
    for _ in range(iterations):
        w_nodes = weight(nodes) if weight is not None else np.ones_like(nodes)
        A = np.hstack([cheb_basis(nodes), ((-1.0) ** k / w_nodes).reshape(-1, 1)])
        sol, *_ = np.linalg.lstsq(A, t(nodes), rcond=None)
        cs, E_new = sol[:-1], abs(float(sol[-1]))

        e = errfun(cs, grid)
        cand = [(grid[0], e[0])]
        for i in range(1, len(grid) - 1):
            if (e[i] - e[i - 1]) * (e[i + 1] - e[i]) < 0:
                cand.append((grid[i], e[i]))
        cand.append((grid[-1], e[-1]))
        # 符号段分组 → 每段取 |e| 最大 → 符号交替的节点序列
        runs: List[List[Tuple[float, float]]] = []
        for x_c, e_c in cand:
            if runs and (e_c >= 0) == (runs[-1][-1][1] >= 0):
                runs[-1].append((x_c, e_c))
            else:
                runs.append([(x_c, e_c)])
        picks = [max(r, key=lambda p: abs(p[1])) for r in runs]
        while len(picks) > degree + 2:
            picks.pop(min(range(len(picks)), key=lambda i: abs(picks[i][1])))
        while len(picks) < degree + 2:
            rest = sorted((p for r in runs for p in r if p not in picks),
                          key=lambda p: -abs(p[1]))
            added = False
            for p in rest:
                for i in range(len(picks) + 1):
                    trial = sorted(picks[:i] + [p] + picks[i:])
                    signs = [q[1] >= 0 for q in trial]
                    if all(signs[j] != signs[j + 1] for j in range(len(signs) - 1)):
                        picks = trial
                        added = True
                        break
                if added:
                    break
            if not added:
                break
        nodes = np.array([p[0] for p in picks])

        if abs(E_new - E) <= 1e-12 * max(E, 1e-300) + 1e-18:
            E = E_new
            break
        E = E_new

    from numpy.polynomial.chebyshev import cheb2poly
    from numpy.polynomial import polynomial as P
    # 切比雪夫基 → [-1,1] 域单项式 → 复合 u(x)=(2x-(lo+hi))/(hi-lo) 得原域单项式。
    # 注意：必须用 Polynomial.__call__ 的组合语义（polyval 对数组实参是逐元素
    # 幂，不是多项式复合——初版即栽在这里，输出被截断成 deg1）。
    q = cheb2poly(cs)
    inner = P.Polynomial([-(lo + hi) / (hi - lo), 2.0 / (hi - lo)])
    composed = (P.Polynomial(q)(inner)).coef
    measured = float(np.max(np.abs(errfun(cs, grid))))
    return composed, measured


def build_all() -> dict:
    """生成全部近似系数（归一化域 [-1,1] 拟合）并写入 data/minimax/poly_approx.json。

    P3 修订：归一化域拟合消除 Horner 截断误差的 |x|^k 放大（原域 deg15 在
    |x|>1 处发散）。exp 域 [-16,0]（C-S 界 + 公开移位 −8）；inv 域 [0.05,320]
    （覆盖 softmax 行和范围），deg4 + 12 轮 Newton。
    """
    specs = {}

    def _fit(name, target, lo, hi, degree, weight=None, extra=None):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        tgt = lambda tt: target(mid + half * tt)  # noqa: E731
        c, e = remez(tgt, -1.0, 1.0, degree,
                     weight=(lambda tt: weight(mid + half * tt)) if weight else None)
        entry = {"coeffs": c.tolist(), "domain": [lo, hi], "normalized": True,
                 "mid": mid, "half": half, "max_error": e}
        if extra:
            entry.update(extra)
        return name, entry

    specs.update([_fit("gelu_deg15", _gelu, -6, 6, 15)])
    specs.update([_fit("gelu_deg9", _gelu, -6, 6, 9)])
    specs.update([_fit("exp_deg12_onesided", np.exp, -16, 0, 12)])
    specs.update([_fit("inv_init_deg4", lambda s: 1.0/s, 0.05, 320, 4,
                       weight=lambda s: s, extra={"newton_rounds": 12})])
    specs.update([_fit("invsqrt_init_deg4", lambda v: 1.0/np.sqrt(v), 0.5, 16, 4,
                       weight=lambda v: np.sqrt(v), extra={"newton_rounds": 3})])

    for name, sp in specs.items():
        lo, hi = sp["domain"]
        x = np.linspace(lo, hi, 100001)
        t = (x - sp["mid"]) / sp["half"]
        pv = np.polynomial.polynomial.polyval(t, sp["coeffs"])
        if name.startswith("gelu"):
            sp["max_error"] = float(np.abs(pv - _gelu(x)).max())
        elif name.startswith("exp"):
            sp["max_error"] = float(np.abs(pv - np.exp(x)).max())
        elif "inv_init" in name:
            sp["max_error"] = float(np.abs(pv * x - 1).max())
        else:
            sp["max_error"] = float(np.abs(pv * np.sqrt(x) - 1).max())

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    return specs


def load_coeffs(name: str) -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))[name]


def main() -> int:
    specs = build_all()
    for name, s in specs.items():
        print(f"{name}: deg={len(s['coeffs'])-1} domain={s['domain']} "
              f"kind={s['kind']} max_error={s['max_error']:.4e}")
    print(f"-> {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
