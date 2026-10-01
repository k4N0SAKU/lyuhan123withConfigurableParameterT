"""lattice-estimator 安全参数复核（P2-R1 评审 B2/C 项——P3 决策数据包）。

参数对应 docs/01 §5.3 表 7（CKKS/RLWE 按 LWE 模型估计，n=poly 度，q=链总乘积；
secret 取均匀三元组（较 SEAL 稀疏秘密更保守），error 取 σ=3.2 离散高斯）：
    A) [60,40,40,60]        @ 2^15  （mode-b 主线，200bit）
    B) [60]+[40]*19+[60]    @ 2^15  （deep19 回归链，880bit）
    C) [60]+[40]*40+[60]    @ 2^16  （评审 C 项：NONE 级可用，决策参数，1720bit）
    D) [60]+[40]*84+[60]    @ 2^17  （mode-a 理论，3480bit）

依赖：malb/lattice-estimator（需 SageMath 环境，P3 首日 Docker/WSL 运行——
Windows 裸环境 sage.all 不可用，见 P2 工作记录 B2 项受阻声明）。
用法：python -m benchmarks.security_estimator [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "benchmarks" / "results" / "security_estimator.json"
# P6-R2 双输出路径（事故修复）：--smoke 与 A/C 决策运行曾共用单 OUT——
# smoke 重跑把决策 JSON（A/C 组+provenance+终裁依据）整体覆盖（P6-R2 第 1 项）。
# 此后 smoke 一律写 estimator_logs/，决策文件只由 A/C 运行产出。


def _load_estimator():
    import glob
    candidates = []
    for base in ("/opt/le",   # P6：持久位置（WSL /tmp 跨 VM 重启易失，P6 事故）
                 "C:/Users/27471/AppData/Local/Temp/le",
                 "/tmp/le", str(Path.home() / "AppData/Local/Temp/le")):
        candidates += glob.glob(base)
    for c in candidates:
        if (Path(c) / "estimator").is_dir():
            sys.path.insert(0, c)
            break
    from estimator import LWE  # noqa
    from estimator import nd as _nd  # noqa（P4/P5 实测 API：Ternary 为模块级
    # 单例实例（Uniform(-1,1)，不可调用）；DiscreteGaussian 为类（可调用））
    LWE.Ternary = _nd.Ternary
    LWE.DiscreteGaussian = _nd.DiscreteGaussian
    return LWE


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="只跑决策关键参数 C（2^16/1720bit）")
    parser.add_argument("--smoke", action="store_true",
                        help="小参数冒烟（n=2048/q=2^60，~2min——提取修复验证用，"
                             "P6 事故重试：验证 UserDict/inf 过滤后可产出可验证 JSON）")
    args = parser.parse_args()

    LWE = _load_estimator()
    from estimator.io import Logging
    import logging
    Logging.set_level(logging.WARNING)

    cases = {
        "A_modeb_2p15_200bit": dict(n=1 << 15, bits=200),
        "B_deep19_2p15_880bit": dict(n=1 << 15, bits=880),
        "C_decision_2p16_1720bit": dict(n=1 << 16, bits=1720),
        "D_modea_2p17_3480bit": dict(n=1 << 17, bits=3480),
    }
    if args.smoke:
        cases = {"SMOKE_n2048_q60": dict(n=2048, bits=60)}
        smoke_note = ("管线验证/玩具参数——仅证明提取修复后可产出可验证 JSON；"
                      "113-bit 为该玩具参数组的安全水平，非 A1-22 方案参数的"
                      "安全主张（方案参数 A/C 见 --quick）")
    elif args.quick:
        cases = {k: v for k, v in cases.items()
                 if k.startswith("C") or k.startswith("A")}   # 决策组：A+C

    results = {}
    if args.smoke:
        # F①：note 显式入档——裸 113-bit 有被误读风险（玩具参数，非方案安全主张）
        results["note"] = ("管线验证/玩具参数——仅证明提取修复后可产出可验证 JSON；"
                           "113-bit 为该玩具参数组（n=2048/q=2^60）的安全水平，"
                           "非 A1-22 方案参数的安全主张（方案参数 A/C 见 --quick）")
    for name, c in cases.items():
        q = 2 ** c["bits"]
        params = LWE.Parameters(
            n=c["n"], q=q,
            Xs=LWE.Ternary,                        # 保守：均匀三元组（nd.Ternary 为单例）
            Xe=LWE.DiscreteGaussian(3.2),
            tag=name,
        )
        t0 = time.perf_counter()
        try:
            rep = LWE.estimate(params, deny_list=["arora-gb", "bkw"])
            # Cost 是 collections.UserDict 子类——**非** dict 子类，
            # isinstance(v, dict) 恒 False（P5-R1 二次踩坑实证：两次"已修"
            # 均栽在同一过滤器上）；以 hasattr("get") 守卫 + v["rop"] 直取。
            # 不可行攻击的 rop 为 +Infinity（sage oo）——必须过滤，否则
            # min() 得 inf（P5-R1 实证：small-case 探针 dual 攻击返回 inf）。
            import math
            costs = []
            for v in rep.values():
                if not hasattr(v, "get") or v.get("rop") is None:
                    continue
                fv = float(v["rop"])
                if math.isfinite(fv) and fv > 0:
                    costs.append(fv)
            if not costs:
                raise ValueError(f"estimate 返回空（attacks={list(rep) if isinstance(rep, dict) else rep}）")
            best = min(costs)
            bits_sec = int(best).bit_length()   # P6 重试修正：float 无 bit_length
            results[name] = {"n": c["n"], "chain_bits": c["bits"],
                             "security_bits": bits_sec,
                             "wall_s": round(time.perf_counter() - t0, 1),
                             "detail": {k: (v.get("rop") if isinstance(v, dict) else v)
                                        for k, v in rep.items()}}
            print(f"{name}: n={c['n']} chain={c['bits']}bit -> ~{bits_sec}bit 安全 "
                  f"({results[name]['wall_s']}s)")
        except Exception as exc:
            results[name] = {"n": c["n"], "chain_bits": c["bits"], "error": repr(exc)}
            print(f"{name}: FAILED {exc!r}")

    # 双输出路径（P6-R2 第 1 项修复）：smoke 数据仅存 estimator_logs/
    if args.smoke:
        out_path = (REPO_ROOT / "benchmarks" / "results" / "estimator_logs"
                    / f"smoke_{time.strftime('%Y%m%d_%H%M%S')}.json")
    else:
        out_path = OUT
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
