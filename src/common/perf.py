"""A1-22 性能采集框架（对应需求 F9）。

设计约束（决策 D8：所有性能结论必须来自本框架产出的 JSON）：
- 时间口径：``time.perf_counter``，单位毫秒，跨轮次统计 P50/P95/均值/最值；
- 内存口径：psutil 对进程 RSS 后台采样（默认 20ms），报告峰值与轮次基线；
- 网络口径：字节计数由传输封装层统一调用（含协议头），明文基线无网络流量时如实记 0；
- 统计粒度：一轮（round）= 一次完整工作负载调用；``round_total`` 为整轮墙钟时间。

标准分段名（D8：口径统一，各模块禁止自造同义名，新增分段须先在此登记）：
认证 AUTH / 密钥协商 KEYNEG / 加密 ENCRYPT / 传输 TRANSPORT / 线性层 COMPUTE_LINEAR /
非线性层 COMPUTE_NONLINEAR / 门限解密 THRESH_DECRYPT / 分词 TOKENIZE / 反解码 DETOKENIZE。
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import psutil

# ---- 标准分段名（后续阶段的协议埋点必须使用这些常量） ----
SEG_AUTH = "auth"                        # SM2 双向身份认证
SEG_KEYNEG = "key_negotiation"           # SM2 密钥协商 + 会话密钥派生
SEG_ENCRYPT = "encrypt"                  # 客户端加密 / 通道加密
SEG_TRANSPORT = "transport"              # 网络传输（纯链路时间）
SEG_COMPUTE_LINEAR = "compute_linear"    # 线性层密文计算
SEG_COMPUTE_NONLINEAR = "compute_nonlinear"  # 非线性层（近似多项式 / MPC）
SEG_THRESH_DECRYPT = "threshold_decrypt"     # 门限解密
SEG_TOKENIZE = "tokenize"                # 分词 / 定点化
SEG_DETOKENIZE = "detokenize"            # 反解码
SEG_NOOP = "noop"                        # 仅空转自检使用

SCHEMA_VERSION = "a122-perf/1"


def percentile(values: List[float], q: float) -> float:
    """线性插值百分位（与 numpy 默认 linear 方法一致），q ∈ [0, 100]。"""
    if not values:
        raise ValueError("percentile() 需要非空样本")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q 必须在 [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _stats_block(samples_ms: List[float]) -> dict:
    """对一组毫秒样本输出统一统计块。"""
    return {
        "n": len(samples_ms),
        "unit": "ms",
        "mean": sum(samples_ms) / len(samples_ms),
        "p50": percentile(samples_ms, 50),
        "p95": percentile(samples_ms, 95),
        "min": min(samples_ms),
        "max": max(samples_ms),
    }


class SegmentTimer:
    """命名分段计时器。每轮新建一个实例，段名→该轮各次采样的毫秒列表。"""

    def __init__(self) -> None:
        self._samples: Dict[str, List[float]] = {}

    @contextmanager
    def segment(self, name: str):
        """``with timer.segment(SEG_ENCRYPT): ...`` 记录一段耗时。"""
        if not name:
            raise ValueError("分段名不能为空")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._samples.setdefault(name, []).append(elapsed_ms)

    def samples_ms(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self._samples.items()}


class NetworkMeter:
    """网络字节计数器。后续阶段由 socket 发送/接收封装层统一调用（陷阱 5：含协议头）。"""

    def __init__(self) -> None:
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.messages_sent = 0
        self.messages_recv = 0

    def on_send(self, num_bytes: int) -> None:
        if num_bytes < 0:
            raise ValueError("字节数不能为负")
        self.bytes_sent += num_bytes
        self.messages_sent += 1

    def on_recv(self, num_bytes: int) -> None:
        if num_bytes < 0:
            raise ValueError("字节数不能为负")
        self.bytes_recv += num_bytes
        self.messages_recv += 1

    def snapshot(self) -> dict:
        return {
            "bytes_sent": self.bytes_sent,
            "bytes_recv": self.bytes_recv,
            "messages_sent": self.messages_sent,
            "messages_recv": self.messages_recv,
        }

    def reset(self) -> None:
        self.__init__()  # noqa: PLW0201 —— 显式复位，语义与新建实例一致


class MemoryTracker:
    """进程 RSS 采样：进入时记录基线，后台线程周期采样峰值，退出时返回结果。"""

    def __init__(self, interval_s: float = 0.02) -> None:
        self._interval_s = interval_s
        self._proc = psutil.Process()
        self._baseline_rss = 0
        self._peak_rss = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "MemoryTracker":
        self._baseline_rss = self._proc.memory_info().rss
        self._peak_rss = self._baseline_rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self._peak_rss:
                self._peak_rss = rss
            time.sleep(self._interval_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        rss = self._proc.memory_info().rss
        if rss > self._peak_rss:
            self._peak_rss = rss

    def result(self) -> dict:
        return {
            "baseline_rss_bytes": self._baseline_rss,
            "peak_rss_bytes": self._peak_rss,
            "peak_delta_bytes": max(0, self._peak_rss - self._baseline_rss),
        }


@dataclass
class RoundContext:
    """传给工作负载的单轮上下文：工作负载通过它埋点。"""

    index: int
    timer: SegmentTimer = field(default_factory=SegmentTimer)
    net: NetworkMeter = field(default_factory=NetworkMeter)


@dataclass
class RoundRecord:
    index: int
    round_total_ms: float
    segments_ms: Dict[str, List[float]]
    network: dict
    memory: dict


class PerfSession:
    """性能采集会话：驱动工作负载 N 轮，产出符合 a122-perf/1 模式的 JSON 报告。"""

    def __init__(
        self,
        workload: str,
        mode: str,
        config: dict,
        environment: dict,
        track_memory: bool = True,
        keep_rounds: bool = True,
    ) -> None:
        self.workload = workload
        self.mode = mode
        self.config = config
        self.environment = environment
        self.track_memory = track_memory
        self.keep_rounds = keep_rounds
        self.records: List[RoundRecord] = []

    def run(self, workload_fn: Callable[[RoundContext], None], rounds: int) -> dict:
        if rounds < 1:
            raise ValueError("rounds 必须 ≥ 1")
        self.records = []
        for i in range(rounds):
            ctx = RoundContext(index=i)
            if self.track_memory:
                with MemoryTracker() as mem:
                    t0 = time.perf_counter()
                    workload_fn(ctx)
                    total_ms = (time.perf_counter() - t0) * 1000.0
                memory = mem.result()
            else:
                t0 = time.perf_counter()
                workload_fn(ctx)
                total_ms = (time.perf_counter() - t0) * 1000.0
                memory = {"baseline_rss_bytes": None, "peak_rss_bytes": None,
                          "peak_delta_bytes": None}
            self.records.append(RoundRecord(
                index=i,
                round_total_ms=total_ms,
                segments_ms=ctx.timer.samples_ms(),
                network=ctx.net.snapshot(),
                memory=memory,
            ))
        return self.build_report()

    # ---- 汇总 ----
    def build_report(self) -> dict:
        segment_flat: Dict[str, List[float]] = {}
        for rec in self.records:
            for name, samples in rec.segments_ms.items():
                segment_flat.setdefault(name, []).extend(samples)
        totals_ms = [r.round_total_ms for r in self.records]
        net_sent = [r.network["bytes_sent"] for r in self.records]
        net_recv = [r.network["bytes_recv"] for r in self.records]
        peaks = [r.memory["peak_delta_bytes"] for r in self.records
                 if r.memory["peak_delta_bytes"] is not None]

        summary: dict = {
            "round_total": _stats_block(totals_ms),
            "segments": {name: _stats_block(v) for name, v in sorted(segment_flat.items())},
            "network": {
                "bytes_sent_per_round": _stats_block([float(x) for x in net_sent]),
                "bytes_recv_per_round": _stats_block([float(x) for x in net_recv]),
                "bytes_sent_total": sum(net_sent),
                "bytes_recv_total": sum(net_recv),
            },
        }
        if peaks:
            summary["memory"] = {
                "unit": "bytes",
                "peak_delta_mean": sum(peaks) / len(peaks),
                "peak_delta_max": max(peaks),
                "note": "进程 RSS 相对轮次基线的增量；绝对峰值另见 rounds 记录",
            }
        else:
            summary["memory"] = {"note": "未启用内存采样"}

        report = {
            "schema": SCHEMA_VERSION,
            "kind": "performance",
            "workload": self.workload,
            "mode": self.mode,
            "config": self.config,
            "environment": self.environment,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rounds_completed": len(self.records),
            "summary": summary,
        }
        if self.keep_rounds:
            report["rounds"] = [
                {
                    "index": r.index,
                    "round_total_ms": r.round_total_ms,
                    "segments_ms": r.segments_ms,
                    "network": r.network,
                    "memory": r.memory,
                }
                for r in self.records
            ]
        return report


def _sanitize_paths(obj, repo_root: str):
    """递归把含仓库绝对路径的字符串归一化为仓库相对路径（评审 G 项：
    results/*.json 不得携带本机绝对路径）。"""
    if isinstance(obj, str):
        if repo_root in obj:
            return obj.replace(repo_root + "\\", "").replace(repo_root + "/", "")
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_paths(v, repo_root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_paths(v, repo_root) for v in obj]
    return obj


def write_report(report: dict, path) -> None:
    """报告只允许经由本函数写入 results/ 目录（决策 D8）；
    写入前强制做绝对路径归一化（评审 G 项）。"""
    repo_root = str(Path(__file__).resolve().parents[2])
    report = _sanitize_paths(report, repo_root)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
