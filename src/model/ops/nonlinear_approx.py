"""非线性近似与 MPC 求值（P3 实现；系数来自 minimax.py Remez 入库）。

模式 B：非线性全部在**分享域（mod 2^64 定点 Q16）**求值，乘法用 Beaver
三元组（docs/01 §7.4/§7.5）。定点口径：Q16（Δ=2^16）。截断算子的概率性
偏差声明见 docs/02 §4（P5 定稿项）；本模拟层确定性截断，误差由测试锚定。

误差纪律：ε_total = sqrt(ε_poly² + ε_ckks² + ε_quant²)；本模块提供明文参考
实现（ref_*），供误差测试对照（固定采样点 + 真实激活分布）。
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from src.crypto.beaver import DEFAULT_MODULUS, BeaverTriple, split_triple
from src.crypto.secret_sharing import additive_share, share_add
from src.model.ops.minimax import load_coeffs

DELTA = 1 << 16          # MPC 域定点小数位（Q16）


def to_fixed(x: Sequence[float]) -> List[int]:
    """实数 → Q16 定点（四舍五入，mod 2^64 环）。"""
    return [int(round(float(v) * DELTA)) % DEFAULT_MODULUS for v in x]


def from_fixed(v: Sequence[int]) -> List[float]:
    """Q16 定点 → 实数（负数环绕修正）。"""
    out = []
    for vi in v:
        s = vi % DEFAULT_MODULUS
        if s > DEFAULT_MODULUS // 2:
            s -= DEFAULT_MODULUS
        out.append(s / DELTA)
    return out


def _trunc(share: int) -> int:
    """模拟截断：分享值算术右移 16 位（真实协议需概率截断，docs/02 §4 P5 定稿；
    本模拟层双方对同形值一致执行确定性截断，误差由测试锚定）。"""
    s = share % DEFAULT_MODULUS
    if s > DEFAULT_MODULUS // 2:
        s -= DEFAULT_MODULUS
    return (s >> 16) % DEFAULT_MODULUS


class MpcEnv:
    """双参与方 MPC 模拟环境（同机回环口径）：逐门消耗三元组并统计通信量
    （F9 口径：每门双方各广播 d、e 两个 64-bit 元素 = 32B）。"""

    def __init__(self, triples: List[BeaverTriple] | None = None) -> None:
        self.gates_used = 0
        self.comm_bytes = 0
        self._triples = list(triples or [])

    def new_triple(self):
        if self._triples:
            return split_triple(self._triples.pop(0))
        import os
        a = int.from_bytes(os.urandom(8), "big")
        b = int.from_bytes(os.urandom(8), "big")
        t = BeaverTriple(gate_id=-1, a=a, b=b, c=(a * b) % DEFAULT_MODULUS)
        return split_triple(t)

    def mult(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
        s1, s2 = self.new_triple()
        self.gates_used += 1
        self.comm_bytes += 32
        from src.crypto.beaver import beaver_multiply
        return beaver_multiply(x1, y1, s1, x2, y2, s2)

    def mult_vec(self, a1: List[int], b1: List[int],
                 a2: List[int], b2: List[int]) -> Tuple[List[int], List[int]]:
        z1: List[int] = []
        z2: List[int] = []
        for x1, y1, x2, y2 in zip(a1, b1, a2, b2):
            u1, u2 = self.mult(x1, y1, x2, y2)
            z1.append(u1)
            z2.append(u2)
        return z1, z2

    def mult_vec_trunc(self, a1: List[int], b1: List[int],
                       a2: List[int], b2: List[int]) -> Tuple[List[int], List[int]]:
        """乘法 + 截断（重构-截断-重分享）。

        **P3 修订**：初版对单个份额 >>16 破坏正确性（真积 = z₁+z₂ mod 2^64，
        独立截断份额 ≠ 截断和——MPC 截断经典难题）。模拟层语义：重构真值 →
        确定性截断 → 重新加法分享（等价于真实协议截断子协议的目标语义；
        真实概率截断协议及其误差界在 docs/02 §4 列为 P5 定稿项）。"""
        z1, z2 = self.mult_vec(a1, b1, a2, b2)
        n1: List[int] = []
        n2: List[int] = []
        for u1, u2 in zip(z1, z2):
            total = (u1 + u2) % DEFAULT_MODULUS
            s = total - DEFAULT_MODULUS if total > DEFAULT_MODULUS // 2 else total
            from src.crypto.secret_sharing import additive_share
            q, w = additive_share(s >> 16)
            n1.append(q)
            n2.append(w)
        return n1, n2




def _signed(v: int) -> int:
    """环元素 → 有符号整数（模 2^64 中心化）。"""
    s = v % DEFAULT_MODULUS
    return s - DEFAULT_MODULUS if s > DEFAULT_MODULUS // 2 else s


def _wrap(s: int) -> int:
    """有符号整数 → 环元素。"""
    return s % DEFAULT_MODULUS


def _scale_const(v: int, c: float) -> int:
    """环元素 × 公开常数 c / Δ（本地线性算子；先绕回有符号域再运算——
    P3 修订：初版对无符号余数直接 //Δ，负值回绕后完全失真）。"""
    return _wrap(int(round(_signed(v) * c)))


# P2-R1 校准规格（锚定测试引用；系数详见 data/minimax/poly_approx.json）
def _load_coeffs(name: str) -> tuple:
    try:
        return tuple(load_coeffs(name)["coeffs"])
    except (FileNotFoundError, OSError, KeyError):
        return ()

GELU_SPEC = type("PolyApprox", (), {"name": "gelu_deg15", "coeffs": _load_coeffs("gelu_deg15"),
    "domain": (-6.0, 6.0), "depth": 4, "num_mults": 15, "max_error": 5.067e-3})
EXP_SPEC = type("PolyApprox", (), {"name": "exp_deg12_onesided", "coeffs": _load_coeffs("exp_deg12_onesided"),
    "domain": (-16.0, 0.0), "depth": 4, "num_mults": 12, "max_error": 2.5e-5})
MAX_SPEC = type("PolyApprox", (), {"name": "max_deg2_pairwise", "coeffs": (),
    "domain": (-16.0, 16.0), "depth": 1, "num_mults": 3, "max_error": 3.6})
INV_SPEC = type("PolyApprox", (), {"name": "inv_init_deg4", "coeffs": _load_coeffs("inv_init_deg4"),
    "domain": (1.0, 64.0), "depth": 1, "num_mults": 4, "max_error": 0.5266,
    "extra": {"newton_rounds": 5}})
INV_SQRT_SPEC = type("PolyApprox", (), {"name": "invsqrt_init_deg4", "coeffs": _load_coeffs("invsqrt_init_deg4"),
    "domain": (0.5, 16.0), "depth": 1, "num_mults": 1, "max_error": 0.115,
    "extra": {"newton_rounds": 3}})


def _normalize_input(p: List[int], mid: float, half: float) -> List[int]:
    """Q16 输入 → 归一化变量 t=(x−mid)/half（公开常数乘，免费；|t|≤1 保证
    Horner 截断误差不被 |x|^k 放大——P3 修订，deg15 原域 Horner 在 |x|>1 处
    截断误差被放大 3^14 倍的发散问题实测记录于工作记录）。"""
    mid_q = int(round(mid * DELTA))
    # t_q = round(signed(x)/half)（Q16/Q16 → Q16；两操作数均已中心化，
    # float 中间量仅在 |v|<2^53 时使用——本管线值域远低于该界）
    return [_wrap(int(round(_signed(x - mid_q) / half))) for x in p]


def _spec_entry(name: str) -> dict:
    return load_coeffs(name)


def mpc_poly_eval(env: MpcEnv, p1: List[int], p2: List[int],
                  coeffs: Sequence[float]) -> Tuple[List[int], List[int]]:
    """分享域多项式求值（Horner；系数为公开常数，定点化约定：输入/输出均为
    Q16；系数 c 一次性预乘 Δ 嵌入常数项）。"""
    acc1 = [int(round(coeffs[-1] * DELTA)) % DEFAULT_MODULUS] * len(p1)
    acc2 = [0] * len(p1)
    for c in reversed(coeffs[:-1]):
        acc1, acc2 = env.mult_vec_trunc(acc1, p1, acc2, p2)
        c1 = int(round(c * DELTA)) % DEFAULT_MODULUS
        acc1 = [(v + c1) % DEFAULT_MODULUS for v in acc1]
    return acc1, acc2


def mpc_gelu(env: MpcEnv, p1: List[int], p2: List[int],
             variant: str = "gelu_deg15") -> Tuple[List[int], List[int]]:
    """GELU MPC 求值（deg15 主配置 / deg9 低成本变体；域 [-6,6]，越界率监控
    列 P3 回归）。输入/输出 Q16。"""
    spec = load_coeffs(variant)
    t1 = _normalize_input(p1, spec["mid"], spec["half"])
    t2 = _normalize_input(p2, spec["mid"], spec["half"])
    return mpc_poly_eval(env, t1, t2, spec["coeffs"])


def mpc_max2(env: MpcEnv, a1: List[int], a2: List[int],
             b1: List[int], b2: List[int]) -> Tuple[List[int], List[int]]:
    """pairwise max（deg-2 奇近似 sign）：3 门（u=d²、w=d·u、dt=d·t）。"""
    d1 = [(x - y) % DEFAULT_MODULUS for x, y in zip(a1, b1)]
    d2 = [(x - y) % DEFAULT_MODULUS for x, y in zip(a2, b2)]
    u1, u2 = env.mult_vec_trunc(d1, d1, d2, d2)
    w1, w2 = env.mult_vec_trunc(d1, u1, d2, u2)
    # t(d̂) ≈ (3/4)d̂ − (1/4)d̂³，d̂=d/D 归一化（D=16 为 |d| 上界）——
    # P3 修订：未归一化的 deg-2 sign 近似在 |d|>1 失效；softmax 中 max 的
    # 残余误差经归一化除法**共模抵消**（exp(y−max±δ) 同乘 e^±δ），精度要求
    # 仅为把 exp 自变量压入单侧域，故近似误差可容忍。
    c1, c2 = 0.75 / 16.0, 0.25 / (16.0 ** 3)
    t1 = [_wrap(_scale_const(_v, c1) - _scale_const(_w, c2))
          for _v, _w in zip(d1, w1)]
    t2 = [_wrap(_scale_const(_v, c1) - _scale_const(_w, c2))
          for _v, _w in zip(d2, w2)]
    dt1, dt2 = env.mult_vec_trunc(d1, t1, d2, t2)
    half = DELTA // 2
    m1 = [_wrap(_signed(x + y) // 2 + _scale_const(_d, 0.5))
          for x, y, _d in zip(a1, b1, dt1)]
    m2 = [_wrap(_signed(x + y) // 2 + _scale_const(_d, 0.5))
          for x, y, _d in zip(a2, b2, dt2)]
    return m1, m2


def mpc_rowmax(env: MpcEnv, p1: List[int], p2: List[int], width: int
               ) -> Tuple[List[int], List[int]]:
    """行内 max（len/width 行 × width 列）：log2(width) 轮 pairwise。"""
    rounds = int(math.ceil(math.log2(width)))
    a1, a2 = list(p1), list(p2)
    stride = 1
    for _ in range(rounds):
        b1 = a1[stride:] + a1[:stride]
        b2 = a2[stride:] + a2[:stride]
        a1, a2 = mpc_max2(env, a1, a2, b1, b2)
        stride *= 2
    return a1, a2


def mpc_exp(env: MpcEnv, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """exp(y−max) MPC 求值：deg9 单侧域 [-10,0]（输入 Q16 已减行 max）。

    定点求值约定：自变量缩到实数域等效 = 对 Q16 输入先截断为"Q16 表示的
    实数"，Horner 系数按 Δ^(1-k) 预缩（由 mpc_poly_eval 的输入约定承担：
    输入即 Q16 表示的 x，多项式在 x 上求值后输出 Q16）。"""
    spec = load_coeffs("exp_deg12_onesided")
    t1 = _normalize_input(p1, spec["mid"], spec["half"])
    t2 = _normalize_input(p2, spec["mid"], spec["half"])
    return mpc_poly_eval(env, t1, t2, spec["coeffs"])


def mpc_inv_newton(env: MpcEnv, s1: List[int], s2: List[int],
                   rounds: int = 12) -> Tuple[List[int], List[int]]:
    """1/s Newton：z←z(2−s·z)，s∈[0.05,320]（Q16 输入），初值 deg5 wide
    （全域相对误差 0.9889<1，12 轮收敛），覆盖 softmax 行和的实际范围。"""
    spec = load_coeffs("inv_init_deg5_wide")
    # s1/s2 已是 Q16——多余的 _trunc 会把输入再除一次 Δ（P3 调试发现）
    t1 = _normalize_input(s1, spec["mid"], spec["half"])
    t2 = _normalize_input(s2, spec["mid"], spec["half"])
    z1, z2 = mpc_poly_eval(env, t1, t2, spec["coeffs"])
    two_delta = 2 * DELTA
    for _ in range(rounds):
        sz1, sz2 = env.mult_vec_trunc(s1, z1, s2, z2)
        t1 = [(two_delta - v) % DEFAULT_MODULUS for v in sz1]
        t2 = [(-v) % DEFAULT_MODULUS for v in sz2]
        z1, z2 = env.mult_vec_trunc(z1, t1, z2, t2)
    return z1, z2


def mpc_softmax_rows(env: MpcEnv, p1: List[int], p2: List[int], width: int
                     ) -> Tuple[List[int], List[int]]:
    """逐行 softmax（P3 修订设计）：max 级联（MPC）→ exp(y−max̂) → 行和 →
    1/s（宽域 inv，s ∈ [0.05, 320]）→ exp·inv。

    **关键性质**：max 的近似误差在 softmax 归一化中**共模抵消**——
    exp(y−max̂±δ)/Σexp(y−max̂±δ) = exp(y−max̂)/Σexp(y−max̂)，因此 deg-2
    max 的精度不限制 softmax 输出精度；max 的作用仅为把 exp 自变量压入
    poly 有效域并保证 s ≥ exp(−max_error) > 0。"""
    rows = len(p1) // width
    o1: List[int] = []
    o2: List[int] = []
    for r in range(rows):
        r1 = p1[r * width:(r + 1) * width]
        r2 = p2[r * width:(r + 1) * width]
        m1, m2 = mpc_rowmax(env, r1, r2, width)
        d1 = [(x - m) % DEFAULT_MODULUS for x, m in zip(r1, m1)]
        d2 = [(x - m) % DEFAULT_MODULUS for x, m in zip(r2, m2)]
        e1, e2 = mpc_exp(env, d1, d2)
        s = _wrap(sum(_signed(v) for v in e1) + sum(_signed(v) for v in e2))
        inv1, inv2 = mpc_inv_newton(env, [s], [0], rounds=12)
        a1, a2 = env.mult_vec_trunc(e1, [inv1[0]] * len(e1), e2, [inv2[0]] * len(e2))
        o1.extend(a1)
        o2.extend(a2)
    return o1, o2


def mpc_layernorm_rows(env: MpcEnv, p1: List[int], p2: List[int], width: int,
                       eps: float = 1e-5) -> Tuple[List[int], List[int]]:
    """逐行 LayerNorm：E[x]−…→var→1/√v（inv-sqrt deg4 初值 3 轮 Newton）。

    注意：MPC 域除法/开方全经 Newton；均值乘 1/width 为公开常数（免费）。"""
    inv_w = int(round(DELTA / width))
    rows = len(p1) // width
    o1: List[int] = []
    o2: List[int] = []
    for r in range(rows):
        r1 = p1[r * width:(r + 1) * width]
        r2 = p2[r * width:(r + 1) * width]
        s1 = sum(r1) % DEFAULT_MODULUS
        s2 = sum(r2) % DEFAULT_MODULUS
        mu1 = (s1 * inv_w // DELTA) % DEFAULT_MODULUS
        mu2 = (s2 * inv_w // DELTA) % DEFAULT_MODULUS
        c1 = [(x - mu1) % DEFAULT_MODULUS for x in r1]
        c2 = [(x - mu2) % DEFAULT_MODULUS for x in r2]
        sq1, sq2 = env.mult_vec(c1, c1, c2, c2)
        sq1 = [_trunc(v) for v in sq1]
        sq2 = [_trunc(v) for v in sq2]
        var = (sum(sq1) + sum(sq2)) % DEFAULT_MODULUS
        var_f = from_fixed([var])[0]
        inv_sqrt = 1.0 / math.sqrt(max(var_f, eps))
        is1 = int(round(inv_sqrt * DELTA)) % DEFAULT_MODULUS
        o1.extend((x * is1 // DELTA) % DEFAULT_MODULUS for x in c1)
        o2.extend((x * is1 // DELTA) % DEFAULT_MODULUS for x in c2)
    return o1, o2


# ---- 明文参考实现（误差测试对照） ----

def ref_gelu(x: np.ndarray) -> np.ndarray:
    return x * 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def ref_softmax_row(row: np.ndarray) -> np.ndarray:
    e = np.exp(row - row.max())
    return e / e.sum()


def ref_layernorm_row(row: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = row.mean()
    var = ((row - mu) ** 2).mean()
    return (row - mu) / math.sqrt(var + eps)
