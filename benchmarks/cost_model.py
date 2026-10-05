# -*- coding: utf-8 -*-
"""D5′ 转换成本先验模型（P9-2c；EncFormer cost-model 同思路）。

用途：布局/参数改造**先算后跑**——给定 (参数组, m, n, 优化开关) 输出每次
请求的转换成本预测（在线通信/转换次数），与实测曲线（mparty_sweep_m2_15.json、
mparty_cost_curve_store.json）互校。模型常数来自实测（密文尺寸=poc_2e16/
masked_ct_bytes 实测，z=8B/槽，10n/层=docs/01 §7.4）；预测值为先验估计，
落地后仍须实测重录（D8）。

运行：python -m benchmarks.cost_model --m 3
"""
from __future__ import annotations

import argparse
import json

CT_MIB = {"2e15": 1.64, "2e16": 2.51}    # 实测：masked_ct_bytes / poc_2e16
Z_MIB = 0.125                             # z 向量 8B×16384 槽
CONV_POINTS_PER_LAYER = 10                # 实测口径：classify conversions=10/层
LAYERS = 12
N_CTS = {"2e15": 2, "2e16": 1}            # L=16 所需密文数（docs/01 §5.3：10+6 块 / 单密文）
# ⚠ 口径疑点（P9 登记）：docs/01 §5.3 的「转换 192 次」与本模型 240（=10 点×12 层×2 密文）
# 及「10n/层」公式存在口径出入——待对照管线实测 conversions 核实（P9 提案 §遗留）。


def predict(params: str, m: int, n: int = 16, mask_store: bool = False,
            modulus_trimming: float = 1.0, complex_packing: float = 1.0) -> dict:
    """单请求转换成本预测。

    - 入口通信/转换 = m×ct（掩码链 m 跳）+ z（y 分发）
    - 出口通信/转换 = m×z + [库存关: m×ct + ct（各片 Enc(sᵢ)+fresh）]
                       [库存开: ct（fresh）]        —— Enc(sᵢ) 移到离线
    - modulus_trimming：EncFormer 式解密前模数裁剪系数（实测 0.3×，先验假设）
    - complex_packing：复数打包跨域载荷减半系数（EncFormer 实测 0.5×，先验假设）
    """
    ct = CT_MIB[params] * modulus_trimming * complex_packing
    conv_per_layer = CONV_POINTS_PER_LAYER * N_CTS[params]
    conv_total = conv_per_layer * LAYERS
    entry_per_conv = m * ct + Z_MIB
    if mask_store:
        exit_per_conv = m * Z_MIB + ct
    else:
        exit_per_conv = m * (Z_MIB + ct) + ct
    per_conv = entry_per_conv + exit_per_conv
    return {
        "params": params, "m": m, "n_tokens": n,
        "conversions_per_layer": conv_per_layer,
        "conversions_total": conv_total,
        "comm_per_conv_mib": round(per_conv, 3),
        "comm_per_request_mib": round(per_conv * conv_total, 1),
        "comm_per_request_gib": round(per_conv * conv_total / 1024, 2),
        "levers": {"mask_store": mask_store, "modulus_trimming": modulus_trimming,
                   "complex_packing": complex_packing},
    }


def main() -> int:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, default=3)
    args = ap.parse_args()
    rows = []
    scenarios = ((False, 1.0, 1.0, "现状"), (True, 1.0, 1.0, "库存"),
                 (True, 0.6, 1.0, "库存+裁剪"), (True, 0.6, 0.5, "库存+裁剪+复数"))
    for params in ("2e15", "2e16"):
        for store, trim, cplx, tag in scenarios:
            r = predict(params, args.m, mask_store=store, modulus_trimming=trim,
                        complex_packing=cplx)
            r["scenario"] = f"{params}/{tag}"
            rows.append(r)
    print(f"{'场景':<20} {'m':>3} {'转换/请求':>8} {'通信/请求':>12}")
    for r in rows:
        print(f"{r['scenario']:<20} {r['m']:>3} {r['conversions_total']:>8} "
              f"{r['comm_per_request_gib']:>10.2f} GiB")
    print("\n模型常数：ct 尺寸=实测（poc_2e16/masked_ct_bytes）；z=8B×16384 槽；"
          "10n/层=docs/01 §7.4；裁剪/复数系数=EncFormer 实测先验（落地须重测）")
    return 0


if __name__ == "__main__":
    main()
