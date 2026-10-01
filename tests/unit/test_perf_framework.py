"""性能采集框架单元测试（F9 框架层验收）。"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.common import perf
from src.common.envinfo import collect_environment

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPercentile:
    def test_known_values(self):
        vals = [float(i) for i in range(1, 101)]  # 1..100
        assert perf.percentile(vals, 50) == pytest.approx(50.5)
        assert perf.percentile(vals, 95) == pytest.approx(95.05)
        assert perf.percentile(vals, 0) == pytest.approx(1.0)
        assert perf.percentile(vals, 100) == pytest.approx(100.0)

    def test_single_and_empty(self):
        assert perf.percentile([7.0], 95) == 7.0
        with pytest.raises(ValueError):
            perf.percentile([], 50)
        with pytest.raises(ValueError):
            perf.percentile([1.0], 101)

    def test_order_insensitive(self):
        vals = [3.0, 1.0, 2.0]
        assert perf.percentile(vals, 50) == pytest.approx(2.0)


class TestSegmentTimer:
    def test_measures_sleep(self):
        timer = perf.SegmentTimer()
        with timer.segment("s1"):
            time.sleep(0.01)
        samples = timer.samples_ms()["s1"]
        assert len(samples) == 1
        assert samples[0] >= 9.0  # 至少约 10ms

    def test_multiple_samples_and_names(self):
        timer = perf.SegmentTimer()
        for _ in range(3):
            with timer.segment("a"):
                pass
            with timer.segment("b"):
                pass
        assert len(timer.samples_ms()["a"]) == 3
        assert len(timer.samples_ms()["b"]) == 3

    def test_empty_name_rejected(self):
        timer = perf.SegmentTimer()
        with pytest.raises(ValueError):
            with timer.segment(""):
                pass


class TestNetworkMeter:
    def test_counts(self):
        net = perf.NetworkMeter()
        net.on_send(100)
        net.on_send(24)   # 模拟"含协议头"的累计口径
        net.on_recv(64)
        snap = net.snapshot()
        assert snap["bytes_sent"] == 124
        assert snap["bytes_recv"] == 64
        assert snap["messages_sent"] == 2
        assert snap["messages_recv"] == 1

    def test_negative_rejected(self):
        net = perf.NetworkMeter()
        with pytest.raises(ValueError):
            net.on_send(-1)


class TestMemoryTracker:
    def test_peak_ge_baseline(self):
        with perf.MemoryTracker(interval_s=0.01) as mem:
            blob = bytearray(4 * 1024 * 1024)  # 分配 4MiB 抬高 RSS
            blob[:4] = b"a122"
        result = mem.result()
        assert result["peak_rss_bytes"] >= result["baseline_rss_bytes"]
        assert result["peak_rss_bytes"] > 0


class TestPerfSession:
    def _collect(self, track_memory):
        session = perf.PerfSession(
            workload="unit-test", mode="test-mode",
            config={"k": "v"}, environment={"fake": "env"},
            track_memory=track_memory, keep_rounds=True)

        def workload(ctx: perf.RoundContext) -> None:
            with ctx.timer.segment(perf.SEG_NOOP):
                time.sleep(0.002)
            ctx.net.on_send(10 + ctx.index)

        return session.run(workload, rounds=5)

    def test_report_schema(self):
        report = self._collect(track_memory=False)
        assert report["schema"] == perf.SCHEMA_VERSION
        for key in ("workload", "mode", "config", "environment",
                    "timestamp_utc", "rounds_completed", "summary", "rounds"):
            assert key in report
        assert report["rounds_completed"] == 5

        summary = report["summary"]
        assert summary["round_total"]["n"] == 5
        assert summary["round_total"]["p95"] >= summary["round_total"]["p50"]
        assert summary["segments"][perf.SEG_NOOP]["n"] == 5
        net = summary["network"]
        assert net["bytes_sent_total"] == 10 + 11 + 12 + 13 + 14
        assert net["bytes_recv_total"] == 0

    def test_report_json_serializable(self):
        report = self._collect(track_memory=True)
        text = json.dumps(report, ensure_ascii=True)  # 不抛异常即通过
        assert '"a122-perf/1"' in text

    def test_invalid_rounds(self):
        session = perf.PerfSession("w", "m", {}, {})
        with pytest.raises(ValueError):
            session.run(lambda ctx: None, rounds=0)


class TestEnvInfo:
    def test_schema(self):
        env = collect_environment()
        for key in ("platform", "python_version", "cpu", "ram_total_gb",
                    "gpu", "packages", "network_topology",
                    "cpu_load_percent_at_collect", "torch_threads"):
            assert key in env
        assert env["cpu"]["logical_cores"] >= 1
        # 稳定性协议支撑字段（P1-R1 补记）
        assert 0.0 <= env["cpu_load_percent_at_collect"] <= 100.0
        assert env["torch_threads"] is None or env["torch_threads"] >= 1
        assert env["packages"]["torch"] is None or isinstance(env["packages"]["torch"], str)

    def test_tracked_packages_present(self):
        env = collect_environment()
        for pkg in ("torch", "transformers", "tenseal", "gmssl", "numpy"):
            assert pkg in env["packages"]


class TestDryRunRunner:
    """P0 出口条件：性能采集框架空转通过（子进程运行 CLI，校验 JSON 模式）。"""

    def test_dryrun_produces_valid_report(self, tmp_path):
        out = tmp_path / "dryrun.json"
        proc = subprocess.run(
            [sys.executable, "-m", "benchmarks.perf_runner",
             "--workload", "dryrun", "--rounds", "3",
             "--output", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["workload"] == "dryrun"
        assert report["mode"] == "dryrun-selfcheck"
        assert report["rounds_completed"] == 3
        # 全部标准分段（含协议占位段）埋点通路有效
        from src.common.perf import (SEG_AUTH, SEG_COMPUTE_LINEAR,
                                     SEG_COMPUTE_NONLINEAR, SEG_ENCRYPT,
                                     SEG_KEYNEG, SEG_NOOP, SEG_THRESH_DECRYPT,
                                     SEG_TRANSPORT)
        segments = set(report["summary"]["segments"])
        assert {SEG_AUTH, SEG_KEYNEG, SEG_ENCRYPT, SEG_TRANSPORT,
                SEG_COMPUTE_LINEAR, SEG_COMPUTE_NONLINEAR,
                SEG_THRESH_DECRYPT, SEG_NOOP} <= segments
        assert report["summary"]["network"]["bytes_sent_total"] == 0
