"""攻击套件共享框架（F7/F8；每类攻击独立可运行，输出结构化判定结论）。

组件：
- RecordingEndpoint：链路记录端点（窃听者=两端发出字节的全集）；
- collect_view：节点/角色「内存视图」规范化收集（F8 钩子证明的取证基础）；
- VerdictRecorder：判定记录器（会话结束写入 benchmarks/results/attack_verdicts.json，
  D8 口径：结论由测试脚本产出、可复现）；
- 统计工具：chi2_uniformity（密文字节均匀性）、pearson（明文-密文相关，
  已知明文侧区分攻击的检验统计量）。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List

from src.common.perf import write_report

REPO_ROOT = Path(__file__).resolve().parents[2]
VERDICT_PATH = REPO_ROOT / "benchmarks" / "results" / "attack_verdicts.json"


class RecordingEndpoint:
    """链路包装：记录本端发出的全部帧（窃听者视角=两端记录的并集）。"""

    def __init__(self, inner, tag: str) -> None:
        self.inner = inner
        self.tag = tag
        self.frames: List[bytes] = []

    def send(self, data: bytes) -> None:
        self.frames.append(bytes(data))
        self.inner.send(data)

    def recv(self, timeout_s: float = 30.0) -> bytes:
        return self.inner.recv(timeout_s)


def wire_bytes(endpoints: List[RecordingEndpoint]) -> bytes:
    """窃听者捕获的全部线缆字节（两方向并集）。"""
    return b"".join(fr for ep in endpoints for fr in ep.frames)


def _walk(obj, depth: int = 0, seen: frozenset = frozenset()):
    """递归收集对象可达的简单类型值（内存视图取证；防环）。"""
    if depth > 6 or id(obj) in seen:
        return []
    out: List = []
    if isinstance(obj, (bytes, bytearray)):
        out.append(bytes(obj))
    elif isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, int):
        out.append(obj)
    elif isinstance(obj, float):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += _walk(k, depth + 1, seen)
            out += _walk(v, depth + 1, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            out += _walk(v, depth + 1, seen)
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if k.startswith("_") and k not in ("_pending_r", "_secret_ctx"):
                continue
            out += _walk(v, depth + 1, seen | {id(obj)})
    return out


def collect_view(obj) -> Dict:
    """节点/角色内存视图：简单值全集 + 长度摘要（F8 钩子证明数据源）。

    取证口径：`_secret_ctx` 等下划线私有成员**计入**（证明持钥边界），
    其余下划线内部簿记跳过。返回值可直接序列化比对。"""
    values = _walk(obj)
    return {
        "n_values": len(values),
        "bytes_items": [v.hex() for v in values if isinstance(v, bytes)],
        "str_items": [v for v in values if isinstance(v, str)],
        "int_items_sample": [v for v in values if isinstance(v, int)][:64],
        "float_items_sample": [v for v in values if isinstance(v, float)][:64],
    }


def view_contains_token(view: Dict, token: int, width: int = 8) -> bool:
    """视图内是否出现某 token id 的定宽字节表示（大端）。"""
    pat = token.to_bytes(width, "big")
    hexpat = pat.hex()
    if any(hexpat in b for b in view["bytes_items"]):
        return True
    return token in view["int_items_sample"]


def scan_pattern(haystack: bytes, needle: bytes) -> int:
    """字节串出现次数（明文模式扫描）。"""
    if not needle:
        return 0
    return len(re.findall(re.escape(needle), haystack))


def chi2_uniformity(data: bytes) -> float:
    """256 桶字节频率卡方统计量（GCM 密文应接近自由度 255 的期望）。

    返回统计量本身；判读阈值由调用方设定（均匀分布期望 255±扰动）。"""
    if not data:
        return float("inf")
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    exp = len(data) / 256
    return sum((c - exp) ** 2 / exp for c in counts)


def pearson(xs, ys) -> float:
    """Pearson 相关（已知明文侧区分攻击的检验统计量；|r|≈0 ⇒ 无区分力）。"""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs, ys = list(xs[:n]), list(ys[:n])
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    vy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


class VerdictRecorder:
    """判定记录器：collect(attack_id, name, verdict, evidence) → 会话结束落盘。"""

    def __init__(self) -> None:
        self.entries: List[dict] = []

    def collect(self, attack_id: str, name: str, verdict: str,
                evidence: dict) -> None:
        assert verdict in ("防御成功", "防御失败", "边界演示")
        self.entries.append({"attack": attack_id, "case": name,
                             "verdict": verdict, "evidence": evidence})

    def dump(self) -> str:
        report = {
            "schema": "a122-attack/1",
            "kind": "attack-verdicts",
            "timestamp_utc": __import__("time").strftime(
                "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
            "environment": {"python": __import__("sys").version.split()[0]},
            "verdicts": self.entries,
            "summary": {
                "defended": sum(1 for e in self.entries
                                if e["verdict"] == "防御成功"),
                "demonstrated_boundary": sum(1 for e in self.entries
                                             if e["verdict"] == "边界演示"),
                "failed": sum(1 for e in self.entries
                              if e["verdict"] == "防御失败"),
            },
        }
        write_report(report, VERDICT_PATH)
        return json.dumps(report["summary"], ensure_ascii=False)


VERDICTS = VerdictRecorder()
