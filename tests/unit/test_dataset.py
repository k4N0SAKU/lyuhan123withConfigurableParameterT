"""数据集完整性测试（P0 陷阱：数据避免真实个人信息）。"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SENTIMENT_DIR = REPO_ROOT / "data" / "sentiment"


def load_rows(name: str) -> list:
    lines = (SENTIMENT_DIR / name).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id\tlabel\ttext"
    rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
    for r in rows:
        assert len(r) == 3, f"{name} 列数错误: {r}"
    return rows


class TestDataset:
    def test_minimum_size(self):
        train, ev = load_rows("train.tsv"), load_rows("eval.tsv")
        assert len(train) >= 200 and len(ev) >= 200  # 任务要求 ≥200 条
        assert len(train) == 400 and len(ev) == 200  # 当前锁定规模

    def test_binary_and_balanced(self):
        for name, n_pos in (("train.tsv", 200), ("eval.tsv", 100)):
            rows = load_rows(name)
            labels = [r[1] for r in rows]
            assert set(labels) == {"0", "1"}
            assert labels.count("1") == n_pos
            assert labels.count("0") == len(rows) - n_pos

    def test_unique_ids_and_prefix(self):
        for name, prefix in (("train.tsv", "S"), ("eval.tsv", "E")):
            rows = load_rows(name)
            ids = [r[0] for r in rows]
            assert len(set(ids)) == len(ids)
            assert all(i.startswith(prefix) for i in ids)

    def test_text_sane(self):
        for name in ("train.tsv", "eval.tsv"):
            for sid, _, text in load_rows(name):
                assert 5 <= len(text) <= 60, f"{sid} 文本长度异常: {len(text)}"
                assert text == text.strip() and "　" not in text

    def test_no_personal_info_patterns(self):
        """禁止形似手机号/证件号的长数字串（隐私红线，全部为合成短句）。"""
        for name in ("train.tsv", "eval.tsv"):
            for sid, _, text in load_rows(name):
                assert not re.search(r"\d{11,}", text), f"{sid} 疑似敏感数字串"
                assert "@" not in text, f"{sid} 疑似邮箱"
