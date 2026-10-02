"""管线 m 方等价性 e2e（P8；slow——真实 BERT + mode-b 全管线）。

等价口径（与 [K] 一致率口径相同）：
- label 必须一致（m=2/3 与原管线）；
- prob 差 ≤1e-4——环算术重构精确（单测层已证逐位），但 CKKS 解码存在
  ±1 ulp 边界抖动（conversion.py 申报的 +e_ckks ≤1 ulp；fresh 掩码每跑
  独立，舍入边界命中随机）→ 管线级prob 本就不逐位稳定；
- 转换计数一致（10n/层 口径不变）。
"""
from __future__ import annotations

import pytest

from src.model.loader import BertSentimentPipeline
from src.model.pipeline import ModeBPipeline, PipelineConfig
from src.model.pipeline_m import ModeBPipelineM, PipelineConfigM

pytestmark = pytest.mark.slow

TEXT = "这家店的分量足味道好服务也热情，下次还会再来。"


def test_pipeline_m_equivalent_to_legacy():
    plain = BertSentimentPipeline()
    base = ModeBPipeline(plain, PipelineConfig(n_layer=1, seq_tokens=8))
    ref = base.classify(TEXT)

    for m in (2, 3):
        pipe = ModeBPipelineM(plain, PipelineConfigM(n_layer=1, seq_tokens=8,
                                                     n_compute=m))
        got = pipe.classify(TEXT)
        assert got["label"] == ref["label"], f"m={m} 标签漂移"
        assert abs(got["prob"] - ref["prob"]) <= 1e-4, \
            f"m={m} 概率超出 ±1 ulp 等价窗：{got['prob']} vs {ref['prob']}"
        # 转换数与门数：m 方转换数公式不变（10n/层 口径的管线计数）
        assert got["conversions"] == ref["conversions"]


def test_m15_boundary_classify():
    """m=15（mode-b 上限，t=14）边界：全管线 classify 与原管线 label 一致。"""
    plain = BertSentimentPipeline()
    ref = ModeBPipeline(plain, PipelineConfig(n_layer=1, seq_tokens=2)).classify(TEXT)
    pipe = ModeBPipelineM(plain,
                          PipelineConfigM(n_layer=1, seq_tokens=2, n_compute=15))
    got = pipe.classify(TEXT)
    assert got["label"] == ref["label"], "m=15 边界标签漂移"
    assert abs(got["prob"] - ref["prob"]) <= 1e-4
