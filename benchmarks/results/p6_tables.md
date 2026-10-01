# P6 表格（由 run_full_bench.py 自动生成——禁止手抄数字）

## 1. 代码规模（分模块）

| 模块 | 文件 | 代码行 | 注释 | 空行 |
|---|---|---|---|---|
| crypto | 6 | 789 | 21 | 182 |
| protocol | 7 | 1590 | 54 | 342 |
| model | 10 | 1181 | 36 | 247 |
| nodes | 8 | 746 | 18 | 125 |
| common | 4 | 422 | 6 | 75 |
| benchmarks | 16 | 1870 | 47 | 318 |
| tests | 34 | 3351 | 87 | 638 |
| **合计** | 85 | 9949 | 269 | 1927 |

## 2. 密文推理矩阵（模式 B × 层数；L=2；D8：JSON 可追溯）

| 配置 | 轮数 | 端到端 P50 (s) | P95 (s) | 最长轮 (s) | 内存增量 P50 (MiB) | 内存增量 max (MiB) | 转换数/轮 |
|---|---|---|---|---|---|---|---|
| 模式B/4层 | 20/20 | 121.7 | 133.5 | 136.8 | 5016.3 | 8345.5 | 40 |
| 模式B/8层 | 20/20 | 237.7 | 438.2 | 445.5 | 4451.7 | 11382.1 | 80 |
| 模式B/12层 | 20/20 | 371.8 | 628.9 | 1981.6 | 5333.2 | 15671.0 | 120 |
| 模式A/4·8·12层 | - | N/A | N/A | N/A | N/A | N/A |

模式 A：本栈不可实例化（docs/01 §5.3）——N/A 如实申报。

## 3. 明文基线（同机同模型，≥20 轮）

| 基线 | P50 (ms) | P95 (ms) | 来源 |
|---|---|---|---|
| BERT 情感分类（满配 12 层） | 23.6 | 35.7 | 本机实测（p6_full_bench.json） |
| GPT-2 生成（16 token） | 677.2 | 723.0 | 本机实测（p6_full_bench.json） |

满配精度（INT8+Q16，12 层）：accuracy = 99.00%（n=200，精度口径，与截断性能分表）

## 4. C3 双实现 A/B（对角线+BSGS vs row-sum，真实 mode-b 参数）

| 维度 | 实现 | 时间 (s) | 旋转 | 明文乘 | 最大误差 |
|---|---|---|---|---|---|
| d256 | diagonal_bsgs | 2.603 | 33 | 256 | 9.8e-08 |
| d256 | rowsum | 14.031 | 256 | 256 | 2.4e-06 |
| d512 | diagonal_bsgs | 5.693 | 45 | 512 | 9.3e-07 |
| d512 | rowsum | 29.236 | 512 | 512 | 2.2e-06 |

结论：对角线+BSGS 胜出（实测 5× 量级）——P2 推测『row-sum 或快 1.3×』被否定：BSGS 将旋转压到 2√d，而 row-sum 的 d 次旋转在 ρ≈1.7 下不可回收；解析单位模型低估了环上全宽明文编码成本（P6 实测口径）

## 5. C2 参数自适应 A/B（GPT-2；20 prompts × 16 token；docs/05 注入源）

| 配置 | token 一致率 | 体积 (MiB) | 时延 P50 (ms) | 判定 | 方案分布 |
|---|---|---|---|---|---|
| default_q22 | 1.0000 | 474.7 | 915 | ✅ P0 默认档（可复现锚点） | {"q22": "全部 Conv1D/Linear/Embedding"} |
| pure_int8 | 0.8906 | 268.1 | 890 | ✗ 一致率不足 | {"int8": 48} |
| adaptive | 0.8906 | 268.1 | 878 | ✗ 退化为 pure_int8（阈值判据失效） | {"int8": 48} |
| ladder_per_layer | 0.8906 | 459.5 | 712 | ✗ 升级无效（级联定型） | {"int8": 0, "fp16": 48} |
| conv_int8_emb_q22 | 0.8906 | 231.7 | 686 | ✗ 一致率不足 | {"embedding": "q22", "conv1d": "int8"} |
| conv_fp16_emb_q22 | 1.0000 | 312.7 | 699 | ✅ 合规第二点（−34.1% 体积） | {"embedding": "q22", "conv1d": "fp16"} |
| conv_q22_emb_fp16 | 0.9437 | 399.6 | 685 | ✗ 差 0.63pp | {"embedding": "fp16", "conv1d": "q22"} |
| conv_q22_emb_int8 | 0.3063 | 362.0 | 733 | ✗ 灾难 | {"embedding": "int8", "conv1d": "q22"} |

层级升级 trace：[{"upgraded_layer": 0, "agreement": 0.8906}, {"upgraded_layer": 1, "agreement": 0.8906}, {"upgraded_layer": 2, "agreement": 0.8906}, {"upgraded_layer": 3, "agreement": 0.8906}, {"upgraded_layer": 4, "agreement": 0.8906}, {"upgraded_layer": 5, "agreement": 0.8906}, {"upgraded_layer": 6, "agreement": 0.8906}, {"upgraded_layer": 7, "agreement": 0.8906}, {"upgraded_layer": 8, "agreement": 0.8906}, {"upgraded_layer": 9, "agreement": 0.8906}, {"upgraded_layer": 10, "agreement": 0.8906}, {"upgraded_layer": 11, "agreement": 0.8906}]


FP32 参考体积：474.7 MiB
