# A1-22 模块代码导读（看懂这套系统的一份地图）

> 目的：答辩前把"哪个模块干什么、文件在哪、数字从哪来"一次性装进脑子。
> 仓库根：`a122-llm-privacy/`；代码约 **9,949 行**（p6_full_bench.json 实测口径）。
> 配套阅读：`docs/00-方案概述.md`（方案总述）｜`docs/06-复现手册.md`（怎么跑）。

---

## 0. 一张图看懂系统形态

```
                ┌──────────── 离线 CA（provision.py）────────────┐
                │  签发 SM2 证书 ×3 + 分发 CKKS 公钥/重线性化密钥   │
                └───────────────────┬───────────────────────────┘
                                    ▼
 P0 客户端 ──SM2 双向认证── P1 密钥节点 ──SM2── P2 推理节点
 (client.py)  (keynode.py)              (infernode.py)
    │              │                          │
    │ ①本地加密输入(明文不出P0)               │
    ├──── ②密文/秘密分享经 channel 流转 ──────┤
    │              │   ③ 密文线性层(CKKS对角线+BSGS) │
    │              │   ④ 非线性(MPC+minimax多项式)  │
    │              │   ⑤ Beaver 三元组在线乘法      │
    ├──── ⑥白名单掩码解密(仅情感标签/概率)────┤
    └── ⑦结果返回；全程 SM3 哈希链审计 + 密钥销毁覆写
```

**一轮演示的生命周期**（src/demo/runner.py 状态机）：
PROVISION → AUTHENTICATING → SESSION_READY → INFERRING ↔ DECRYPTING（逐轮）
→ DESTROYING（密钥覆写+审计核验）→ DONE。

**两种模式**：
- **模式 A**：纯 MPC 秘密分享路线——本栈如实申报"不可实例化"（演示里
  有专门的 MODE_A_UNAVAILABLE 状态，不装能跑）；
- **模式 B（主线）**：CKKS 密文线性层 + MPC 非线性 + 密文↔分享转换协议
  （D5 白名单解密在转换点执行）。演示与精度主张全部走模式 B。

---

## 1. 三条阅读路线（按你有多少时间）

| 路线 | 时间 | 顺序 |
|---|---|---|
| **答辩线**（讲得清） | 30 分钟 | 本图 → §2 目录总览 → §6 概念小词典 → §7 追问速查表 |
| **代码线**（答得深） | 2 小时 | §3 分层详解自上而下，每个模块只看文件头 docstring + grep class/def |
| **数字线**（问数字不慌） | 15 分钟 | §8"哪里看什么数字"——所有数字只认 `benchmarks/results/*.json` |

---

## 2. 目录总览（每行一句话）

| 目录 | 一句话 | 关键内容 |
|---|---|---|
| `docs/` | 七份终稿 + 阶段记录 | 00~06 终稿；`phases/` P0~P7 工作记录+总结报告；`data_dict.json` **93 条数字权威** |
| `src/crypto/` | 密码原语层（P2） | Beaver 三元组、CKKS 封装、国密、秘密分享、门限解密模拟 |
| `src/protocol/` | 协议控制面（P1 规格/P4 实现） | 消息格式、会话通道、SM2 认证、密钥生命周期、转换协议、审计链 |
| `src/nodes/` | 三方节点编排（P4） | client/keynode/infernode + orchestrator + 离线供给 provision |
| `src/model/` | 模型与密文算子（P0/P3） | 模型加载、模式 B 管线、量化；`ops/` 线性层/打包/非线性/rowsum |
| `src/common/` | 观测设施（F9） | perf 统计、envinfo 环境块、netmeter 字节计数 |
| `src/demo/` | 演示系统（P7） | FastAPI app + runner 状态机 + 双场景定义（政务/企业） |
| `benchmarks/` | 一切性能/精度数字的**唯一产地**（D8 纪律） | 12 个脚本；`results/` 全部 JSON 证据 |
| `tests/` | 243 个自动化用例 | unit（含并发 append 200）+ e2e（10 轮稳定性）+ attack（五类攻击） |
| `scripts/` | 一键复现与一致性门禁 | `reproduce.sh`（6 步）+ `check_docs_consistency.py`（四检） |
| `data/` | 数据与系数 | `sentiment/` 自建 600 条数据集；`minimax/poly_approx.json` 非线性系数 |

---

## 3. 分层详解（自上而下）

### 3.1 `src/crypto/` —— 密码原语层（P2，评审通过）

| 文件 | 是什么 | 答辩一句话 |
|---|---|---|
| `gm_cipher.py` | SM2/SM3/SM4-GCM 国密封装（gmssl 库 + 标准参数，D2/S3 纪律） | "身份认证、完整性、传输加密全部国密，参数不魔改" |
| `beaver.py` | Beaver 三元组生成与在线乘法协议 | "乘法不开密文：离线备三元组，在线只交换掩码" |
| `ckks_ops.py` | CKKS 封装（基于 TenSEAL 0.3.14 的原生 SEAL 绑定） | "线性层在密文域算，CKKS 噪声预算有规格控制" |
| `secret_sharing.py` | 模 2^k 环上加法秘密分享（模式 B 分享域，docs/01 §7.4） | "非线性部分转分享值，两方各持一半" |
| `threshold.py` | CKKS 门限解密的**协议逻辑级模拟层**（主线不用） | "如实申报：这是逻辑模拟，不是主线路径"（D5 修订诚实标注） |

### 3.2 `src/protocol/` —— 协议控制面（P1 定规格 / P4 实现）

| 文件 | 是什么 |
|---|---|
| `messages.py` | 消息格式**权威代码载体**——docs/01 §3 的每个字段在这里一一对应 |
| `session.py` | 会话与传输通道（含 ratchet 密钥轮换、销毁） |
| `auth.py` | SM2 双向身份认证（F3 载体）——证书链、挑战应答 |
| `keylifecycle.py` | 密钥全生命周期（F5/F6 载体）——生成/分发/轮换/销毁全零覆写 |
| `conversion.py` | **模式 B 密文↔分享转换真实路径**（D5 白名单解密的执行点） |
| `audit_log.py` | SM3 哈希链审计日志（每条 append 带 seq；并发安全——P7-R1 有 4×50 并发测试） |

> 想追"一轮消息怎么走"：`session.py` → `messages.py` → `nodes/orchestrator.py`。

### 3.3 `src/nodes/` —— 三方节点编排（P4）

| 文件 | 是什么 |
|---|---|
| `base.py` | 节点基座与链路（公共设施） |
| `client.py` | **P0 客户端**：输入在本地加密/分享，结果掩码解密——明文永不出 P0 |
| `keynode.py` | **P1 密钥节点**：密钥材料、转换协议一方 |
| `infernode.py` | **P2 推理节点**：跑密文推理 |
| `orchestrator.py` | 三方编排总指挥：会话建立→推理数据面→ratchet→销毁→审计核验 |
| `provision.py` | 离线身份供给（CA 根密钥+节点证书+CKKS 密钥分发，docs/01 §8.7） |
| `offline_triple_gen.py` | 离线 Beaver 三元组生产（D7 离线假设的落点） |

### 3.4 `src/model/` —— 模型与密文算子（P0 预研 / P3 管线）

| 文件 | 是什么 |
|---|---|
| `loader.py` | 模型加载：GPT-2 124M（生成）+ BERT-base-chinese（分类，99.00% 那个） |
| `pipeline.py` | **模式 B 端到端密文推理管线**（层数可截断；P4 转换真实路径接线） |
| `quantize.py` | 模拟定点化与权重量化（P0 明文域预研，P2-R1 校准） |
| `ops/packing.py` | SIMD 打包与密文线性层（选型：对角线+gap-block+BSGS） |
| `ops/linear.py` | 密文线性层与嵌入（吃 packing 的选型） |
| `ops/nonlinear_approx.py` | 非线性近似与 MPC 求值（系数来自 minimax） |
| `ops/minimax.py` | Remez 多项式逼近系数生成器（P2-R1 补做评审 B1 项，系数入库 data/minimax/） |
| `ops/rowsum.py` | row-sum 线性层（C3 双实现的对照实现——**实测慢 5.1×，落选**，诚实负结果） |

### 3.5 `src/common/` —— 观测设施（F9）

| 文件 | 是什么 |
|---|---|
| `perf.py` | 性能采集框架 + **write_report（D8 门禁：所有报告只准从这进 results/，路径强制归一化）** |
| `envinfo.py` | 环境信息采集（性能数字必须带环境块） |
| `netmeter.py` | socket 字节计数（网络流量口径的**唯一采集点**——346,472,232 B 那类数字的家） |

### 3.6 `src/demo/` —— 演示系统（P7）

| 文件 | 是什么 |
|---|---|
| `app.py` | FastAPI 服务：`/api/run` `/api/status` `/api/stop` `/api/health`（409 冲突拒绝） |
| `runner.py` | 状态机（本文 §0 的生命周期）+ **明文扫描钩子**（每轮扫 P1/P2 全字段视图断言 0 命中） |
| `scenarios.py` | 双场景：政务"市民诉求情绪研判" / 企业工单分派；**全部模拟数据，页面如实标注** |

看懂演示只需读这三件：先 `scenarios.py`（30 行），再 `runner.py` 的 `run()`/`_run_inner()`，最后 `app.py` 路由。

### 3.7 `benchmarks/` —— 数字唯一产地（D8）

| 脚本 | 产出 |
|---|---|
| `run_full_bench.py` | `p6_full_bench.json`（四项资源指标+C2/C3+五维对比源数据）+ `p6_tables.md` 自动注入 |
| `finetune_bert.py` | `data/models/bert-base-chinese-sentiment/`（99.00% 的模型本体） |
| `eval_pipeline_accuracy.py` | 管线精度三口径（明文/量化/密文 vs FP32 一致率） |
| `eval_quantize.py` / `primitive_bench.py` / `perf_runner.py` | 量化档位 / 原语微基准 / 端到端分段 |
| `baselines/plaintext_baseline.py` / `security_estimator.py` | 本机明文锚点 / 安全强度估算 |
| `download_models.py` / `gen_prompts.py` / `run_ci.py` / `package_phase.py` | 模型下载（HF+镜像探测）/ 评测 prompt / 基准门禁 / 阶段打包 |

`results/` 里的关键 JSON（答辩最常被问的五个）：
`p6_full_bench.json`（性能总表）、`p7_demo_stability.json`（10 轮稳定性）、
`pipeline_accuracy_*.json`（三口径精度）、`p4_protocol_bench.json`（协议开销）、
`attack_verdicts.json`（30 项攻击判定）。

### 3.8 `tests/` —— 243 个用例的防线

- `tests/unit/`：单测全绿（含 `test_audit_log.py::test_concurrent_append_200`——4×50 并发，P7-R1 补的"声明与交付对齐"用例）；
- `tests/e2e/test_demo_stability.py`：真实 uvicorn+HTTP，10 轮出口断言（**带 stall 取证与 409 教训的鲁棒性口径**，文件头 docstring 写了首跑事故）；
- `tests/attack/`：五类攻击自动化断言——窃听/中间人/结果篡改/重放/合谋，30 项判定=28 防御成功+2 边界演示（模式 B t=2 合谋，预先声明的构造边界）+0 失败。

### 3.9 `scripts/` —— 一键复现与门禁

- `reproduce.sh`：干净目录 6 步一次通过（证据 `reproduce_logs/`）；
- `check_docs_consistency.py`：**四检**（字典↔JSON / 注入表 / 黑名单 / 互引），任何文档改动后必须跑，EXIT=0 才算数。

---

## 4. 关键概念小词典（10 个高频词）

| 词 | 一句话解释 | 深入 |
|---|---|---|
| **模式 A / 模式 B** | A=纯秘密分享（本栈如实申报不可实例化）；B=CKKS+MPC 混合（主线） | 01 §7.4 |
| **D5 白名单解密** | 解密只允许输出预登记的结果类型（情感标签/概率），原始文本密文不解 | 01 §8.4 |
| **Beaver 三元组** | 让两方在不开密文的情况下完成乘法的离线预备材料 | 02 §1.3 |
| **对角线+BSGS** | 密文矩阵乘的打包选型（C3 双实现 A/B 实测胜者，row-sum 慢 5.1×） | 01 §6 |
| **minimax/Remez** | 把 GELU/Softmax 近似成多项式的系数生成法（P2-R1 补做） | data/minimax/ |
| **D8 纪律** | 所有性能数字只准由脚本写入 results/，文档引用必与 JSON 一致，禁手填 | README |
| **四检** | 字典↔JSON、注入表、过期黑名单、文档互引的一致性门禁 | scripts/ |
| **F1~F10** | 需求验收表编号（F2 明文扫描、F3 认证、F8 运行时钩子、F9 环境块…） | 00 |
| **S8** | 同机回环模拟的部署形态（演示即此形态，页面如实标注） | 01 |
| **诚实负结果** | 失败实验照登（rowsum 落选、选择器三形态失效、模式 A 不可用） | 05 |

## 5. 主要决策速记（D 系列）

- **D2/S3**：国密参数只用标准参数，不魔改；
- **D4**：双模型分工——GPT-2 生成 + BERT 分类；
- **D5**：白名单解密 + threshold 降级为逻辑模拟（诚实申报）；
- **D7**：Beaver 三元组离线生成假设（供给脚本落地）；
- **D8**：性能数据纪律（见上表）。

## 6. 追问速查表（"评委问 X → 打开 Y"）

| 评委可能追问 | 打开 |
|---|---|
| "精度 99.00% 怎么来的？和谁比？" | `04 §6` + `pipeline_accuracy_*.json`（n=200，vs FP32 0.00pp；自建 600 条同源对照，公开集锚点见 00 §3 差距 6） |
| "并发写审计会不会竞态？" | `tests/unit/test_audit_log.py::test_concurrent_append_200`（4×50 并发，seq 严格 1..200） |
| "明文真的不出本地？口说无凭" | `src/demo/runner.py::_plaintext_scan` + `tests/attack/` F8 钩子（每轮全字段扫描 0 命中） |
| "合谋攻击防住了吗？" | `03 攻击测试报告`——t=2 合谋是**预先声明的构造边界**（28+2+0） |
| "为什么不用 row-sum？" | `05 §C3`——双实现 A/B 实测慢 5.1×，诚实负结果 |
| "量化后掉精度吗？" | `05 §C2`——conv_fp16+emb_q22 一致率 1.000，体积 −34.1%（312.7 MiB） |
| "流量/内存多大？" | `p7_demo_stability.json`（10 轮 346,472,232 B）+ `p6_full_bench.json` |
| "怎么复现？" | `docs/06` + `scripts/reproduce.sh`（6 步，reproduce_logs/ 有证据） |
| "文档数字会不会手抄错？" | `scripts/check_docs_consistency.py` 四检 EXIT=0（93 条字典全对账） |

## 7. 哪里看什么数字（数字只认 JSON）

| 数字 | 权威来源 |
|---|---|
| 代码规模 9,949 行 | `p6_full_bench.json::code_size` |
| 满配精度 99.00%（n=200） | `pipeline_accuracy_*.json` + 04 §6 |
| C2 最优点 312.7 MiB / −34.1% | `p6_full_bench.json::c2_selector` |
| C3 rowsum 慢 5.1× | `p6_full_bench.json::c3`（如字段名有出入以 p6_tables.md 为准） |
| 演示 10 轮流量 346,472,232 B（330.4 MiB） | `p7_demo_stability.json` |
| 攻击 28+2+0 | `attack_verdicts.json` |
| 测试 243 passed + 16 deselected + 3 xfailed | 亲跑 `pytest`（总结报告 §0） |

> 记住一条铁律：**答辩里报的每个数字，都能在 `benchmarks/results/` 里按
> `docs/data_dict.json` 的 93 条指针回读**——这就是这套项目的底气。
