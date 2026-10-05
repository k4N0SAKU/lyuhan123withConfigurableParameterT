# 文献核验笔记：EncFormer（外部审计推荐，已核验）

> 核验方式：arXiv HTML 全文（arXiv:2604.09975v1）+ 检索交叉确认；核验日期 2026-10-02。
> 状态：**已核验**——元数据与技术内容可直接进报告书文献表。

## 1. 元数据

- 题名：EncFormer: Secure and Efficient Transformer Inference over Encrypted Data
- 作者：Yufan Zhu, Chao Jin, Khin Mi Mi Aung, Xiaokui Xiao（A*STAR / NUS）
- 发表：IEEE TDSC 2026（arXiv:2604.09975）
- 开源：**论文页未提及 EncFormer 本体代码发布**；其评测基于开源第三方组件（PhantomFHE、Liberate.FHE、EzPC/SCI、CrypTen）。"代码开源"需修正为"评测栈开源、本体未见发布"。

## 2. 系统模型

- **两方（2PC）半诚实**：P0=客户端（持输入/CKKS 私钥，收结果）、P1=服务器（持模型）；密钥材料 (pk, evk) 客户端→服务器，解密只在客户端。安全性=CKKS+MPC+转换组件的流水线组合定理（Theorem 1）。
- **与本方案对照**：信任形态同构（客户端独钥+计算方无钥），但 EncFormer 固定 2PC；本方案 1+m 方、计算方间门限 t=m−1（P8）+全链国密控制面。

## 3. 三项核心技术（可移植性评估）

### 3.1 Stage Compatible Patterns（SCP）——布局兼容纪律
- 全管线限定三种规范打包（segment-column / folded-diagonal / head-major），三条规则：FHE→FHE 产出即消费者所需格式（免重打包）；跨域边界用最小打包 K_min。违反规则 1 代价 Ω(m log m) 次旋转/密文。
- kernel 实例：共享投影 kernel（BSGS+复数打包，权重预置换使 Q/K 按得分友好段序、V 按 head-major 输出）；folded-diagonal 得分 kernel（两条注意力对角线塞进同一槽的实/虚部）；旋转银行（预计算段内/段间移位，免冗余 key-switch）。**量化对照：BERT-base 注意力 630 旋转/448 乘 vs BOLT 13824/1536（约 20×）**。
- **对本方案**：我们的 gap-block 布局（块前 d 槽有效+BSGS 垃圾）与 I4 双拷贝 fresh 约束，每段出口靠 RECRYPT 重置布局——RECRYPT 兼有深度重置与 D5′ 掩码双重职责**不可去除**；SCP 的可移植点=①QKV 三线性共享输入布局②旋转银行/旋转集复用③folded-diagonal 得分 kernel（把两条对角线装进实虚部，旋转/乘法数直降）。登记为 1a GPU 化之后的第二优先布局改造评估项。

### 3.2 复数打包 + 模数裁剪的 CKKS–MPC 转换（与本方案 D5′ 直接对标）
- **复数打包**：两条实向量装进一个密文的实/虚部 ⇒ 跨域载荷减半（k 条实向量 ⌈k/2⌉ 个密文）。
- **C2M**（密文→分享）：服务器在**明文环**上采均匀 r̂ 对明文掩蔽（BLB 式），客户端解密一次一密密文 → 环份额 → Π_Field2Ring → 各自解码得实/虚部加性份额。**与本方案入口同构**（我们：计算方密文域加 Enc(rᵢ)、P0 白名单解密）。
- **M2C**（分享→密文）：份额编码为明文环份额，Π_Ring2Field，客户端加密自己份额、服务器本地加明文份额——**与本方案 RECRYPT 同构**。
- **转换载荷实测**：20.97 MB（实数基线）→ 10.49 MB（复数打包）→ **6.41 MB（模数裁剪）**，≈3.3×。
- **模数裁剪（最高价值移植项）**：C2M 前 level down 到最小模数，安全条件 log₂(q) ≥ ℓ+σ+1 且 q/2 > Δ·B_max。**对本方案**：我们的 masked-ct 携带当前层剩余全链模数——解密只需解码正确性（scale 2⁴⁰+掩码窗 2⁴⁹），C2M 前把 masked-ct mod_switch 到最小层可显著缩载荷（每降一级 ~30-40%）；需按我们的 64 位环/2⁴⁹ 掩码窗重推安全条件后实测。

### 3.3 非线性的交互轮次压缩
- Softmax（MBMax）：Powerformer 式 (X+c)⁵/R_d——3 次安全乘法/3 轮；LayerNorm（MBLN）：**固定公开尺度**替代方差归一化——0 交互轮；GELU：分段多项式+MPC 选择 4 轮；每层共 7 MPC 轮。定点域 Z₂^ℓ（ℓ=43，F=13 小数位）。
- **对本方案**：我们的 MPC 为模拟口径（S8 申报），轮次不直接进时延；可移植的是**MBLN 固定公开尺度**——若精度可接受，可把 LN 从"转换点"降为纯本地仿射（减少域穿越次数），需 A/B 精度实测。登记为候选。

## 4. 评测数据（已核验，可引用）

- 模型：GPT2-base（m=64）、BERT-base（m=128）、BERT-large（m=128）；GLUE SST-2/MRPC/RTE 精度近明文（SST-2 91.78% vs 明文 92.43%）。
- 环境：PhantomFHE（GPU）+ EzPC/SCI，A100；LAN 1Gbps/0.3ms。
- 关键数：BERT-base LAN **2.1 min / 在线通信 2.2 GB**；对比 BLB 2.5min/3.0GB、BumbleBee 4.3min/5.8GB、BOLT 15.5min/63.6GB；对混合基线平均 **1.3×–9.8× 时延 / 1.4×–30.4× 通信**；对 FHE-only 1.9×–3.5×；匹配后端单层 179.73s（THOR 623.39s、Powerformer 344.33s）。
- **安全推理 ≈3×10⁴× 于明文**（BERT-base 明文 4.5ms）——与本方案实测 ~1.6×10⁴×（docs/00 记"约 3 个数量级"）同量级，互证"数量级开销是该技术路线的当前共性"。
- 参数：leveled RNS-CKKS 免 bootstrap，N=2^15/2^16（深度 10/7/6/4），scale 2⁴²/2⁴⁰。

## 5. 对本方案的移植清单（按收益/成本排序）

1. **模数裁剪进 D5′ 转换**（高收益/中成本）：masked-ct 发 P0 前 mod_switch 到最小层；需重推 64 位环安全条件（对应 log₂(q) ≥ ℓ+σ+1 的等价式）并实测精度。
2. **复数打包跨域载荷减半**（高收益/中成本）：入口/出口把两条实向量装实虚部；需改 packing 布局与转换配对逻辑。
3. **旋转银行 + folded-diagonal 得分 kernel**（高收益/高成本）：attention kernel 旋转/乘法数对标 630/448；依赖 1a GPU 化后更有意义。
4. **MBLN 固定公开尺度**（中收益/低成本+精度风险）：LN 转换点降为本地仿射；先 A/B 精度。
5. **成本分析模型**（低成本）：先验评估布局改造，配套 2c。

> 红线提示：EncFormer 为 2PC、国际算法栈、无国密控制面——移植其**技术**（布局/转换/裁剪）不等于采用其系统形态；本方案的门限语义（t=m−1）与白名单两触点在所有移植中保持。
