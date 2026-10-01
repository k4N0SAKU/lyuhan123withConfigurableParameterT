# data 目录说明

## sentiment/ —— 自建中文情感二分类演示集

| 文件 | 条数 | 用途 |
|---|---|---|
| `train.tsv` | 400（正 200 / 负 200） | 微调 BERT-base-chinese 分类头 |
| `eval.tsv` | 200（正 100 / 负 100） | FP32 vs 定点化 accuracy 对比、密文推理精度实验（S6） |

### 格式

TSV 三列：`id 	 label 	 text`；`id` 前缀 S=train、E=eval；`label` 1=正面、0=负面。

### 来源与隐私声明

- **来源**：参赛团队自建的合成短句（2026-09 编写），围绕政务办事、购物电商、餐饮外卖、
  出行交通、医疗就诊、教育培训、物业社区、客服售后等 A1-22 题设场景；
- **隐私**：全部为通用化合成语句，**不含任何真实个人信息**（无姓名、电话、证件号、地址；
  含 ≥11 位连续数字的语句在数据校验测试中被禁止）；
- **已知局限（S8 诚实声明）**：本集为演示与量化损耗测量用，分布单一（短句、二分类），
  不能代表开放域基准；与公开基准的对比在 P6 的文献口径部分另行说明，不混用数字。
- **重建注记（2026-09-26）**：本文件为搬迁事故后按最终提交内容忠实重建的版本。

### 使用入口

- 微调：`python -m benchmarks.finetune_bert`
- 精度对比（FP32 vs 定点化）：`python -m benchmarks.eval_quantize`
- 结构校验：`tests/unit/test_dataset.py`（列数、平衡性、去重、隐私检查）

## demo/prompts.txt

GPT-2 贪心解码评测用英文短 prompt（20 条），用于明文基线时延与定点化生成
token 一致率实验。GPT-2 为英文模型，中文 prompt 会产生无意义输出，故生成
任务的演示语料使用英文；中文任务由 BERT 分类承担。

## minimax/poly_approx.json

Remez（minimax）近似多项式系数（由 `python -m src.model.ops.minimax` 生成），
P3 非线性算子使用；`tests/unit/test_minimax.py` 独立复算锚定。
