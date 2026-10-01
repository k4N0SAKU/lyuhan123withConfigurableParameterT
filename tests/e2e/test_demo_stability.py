"""演示系统稳定性（P7 出口条件：连续 10 轮无故障）。

真实 uvicorn 服务器 + 真实 HTTP。判定：10 轮全部完成、白名单解密逐轮成功、
明文扫描逐轮 0 命中、审计链核验通过、指标来自 perf 统计模块（陷阱 2 断言）。
证据落 benchmarks/results/p7_demo_stability.json（D8）。

鲁棒性口径（首跑实证教训）：
- status 请求在 PROVISION 阶段偶发 >10s 卡顿（SEAL 键生成的 GIL 尾部，
  观测 2.4s~偶发更长）——测试以 60s 超时 + 重试容忍并记录 stall 证据，
  单次慢响应不构成"演示故障"（UI 轮询本身天然容忍）；
- 任何测试失败路径必须 stop + 等待 idle，避免污染后续用例（首跑 409 连锁）。

运行：python -m pytest tests/e2e/test_demo_stability.py -m slow -s
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
ROUNDS = 10
DEADLINE_S = 45 * 60
STALL_TOL_S = 60          # 单请求容忍（SEAL keygen GIL 尾部）


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


STALLS: list = []          # 全会话 stall 取证（进入 JSON 证据）


def _get(port, path, timeout=STALL_TOL_S):
    t0 = time.time()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=timeout) as r:
        body = json.loads(r.read())
    ms = (time.time() - t0) * 1000
    if ms > 1000:
        STALLS.append({"path": path, "ms": round(ms),
                       "phase": body.get("phase") if isinstance(body, dict) else None})
    return body


def _post(port, path, body, timeout=STALL_TOL_S):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _wait_idle(port, deadline_s=240):
    """等待运行结束（stop 后当前轮收尾）；返回最终状态。"""
    t0 = time.time()
    st = {"phase": "INIT", "busy": False}
    while time.time() - t0 < deadline_s:
        try:
            st = _get(port, "/api/status")
            if not st.get("busy") and st.get("phase") in (
                    "DONE", "FAILED", "INIT", "MODE_A_UNAVAILABLE"):
                return st
        except Exception:
            time.sleep(2)
        time.sleep(2)
    return st


@pytest.fixture(scope="module")
def demo_server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.demo.app", "--port", str(port)],
        cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                if _get(port, "/api/health", timeout=10)["ok"]:
                    break
            except Exception:
                time.sleep(1)
        else:
            pytest.fail("demo 服务 120s 内未就绪")
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_ten_rounds_stable(demo_server):
    """出口条件：连续 10 轮无故障。失败路径强制停跑隔离（防污染后续用例）。"""
    port = demo_server
    started = _post(port, "/api/run", {"scenario": "gov", "sample": 0,
                                       "rounds": ROUNDS, "mode": "B"})
    assert started["accepted"], started
    t0 = time.time()
    st = None
    polls = 0
    try:
        while time.time() - t0 < DEADLINE_S:
            polls += 1
            try:
                st = _get(port, "/api/status")
            except Exception as exc:            # 偶发慢响应：容忍并取证
                STALLS.append({"path": "/api/status", "poll": polls,
                               "err": type(exc).__name__})
                time.sleep(5)
                continue
            if st["phase"] in ("DONE", "FAILED", "MODE_A_UNAVAILABLE"):
                break
            time.sleep(5)
        elapsed = round(time.time() - t0, 1)
        assert st is not None, "无状态"
        rounds = st.get("rounds") or []
        evidence = {
            "schema": "a122-perf/1", "kind": "demo-stability",
            "workload": "demo_10_rounds", "mode": "p7-demo",
            "config": {"rounds_target": ROUNDS, "scenario": "gov",
                       "pipeline": st.get("pipeline_cfg"),
                       "port": port, "elapsed_s": elapsed, "polls": polls},
            "environment": {"python": sys.version.split()[0], "platform": os.name,
                            "note": "uvicorn 真实 HTTP；stats 来自 src.common.perf"},
            "result": {
                "phase": st.get("phase"),
                "rounds_done": len(rounds),
                "error": st.get("error"),
                "audit_chain_ok": st.get("audit_chain_ok"),
                "destroy_all_zeroed": all(
                    v for d in (st.get("destroy") or {}).values() for v in d.values()),
                "plaintext_all_ok": all(r.get("plaintext_ok") for r in rounds),
                "final_decrypt_all_ok": all(r.get("final_decrypt_ok") for r in rounds),
                "wall_stats": st.get("wall_stats"),
                "segment_stats_keys": sorted((st.get("segment_stats") or {}).keys()),
                "traffic": st.get("traffic"),
                "memory": st.get("memory"),
                "stalls_over_1s": STALLS,
            },
            "rounds": rounds,
        }
        out = REPO / "benchmarks" / "results" / "p7_demo_stability.json"
        from src.common.perf import write_report
        write_report(evidence, out)

        # ---- 出口条件断言 ----
        assert st["phase"] == "DONE", f"phase={st['phase']} error={st.get('error')}"
        assert len(rounds) == ROUNDS, f"轮数 {len(rounds)} != {ROUNDS}"
        assert evidence["result"]["plaintext_all_ok"], "存在明文扫描命中"
        assert evidence["result"]["final_decrypt_all_ok"], "存在白名单②未成功轮"
        assert st.get("audit_chain_ok") is True, "审计链核验失败"
        assert evidence["result"]["destroy_all_zeroed"], "销毁未全部清零"
        # 陷阱 2：指标来自 perf 统计模块
        ws = st["wall_stats"]
        assert ws["unit"] == "ms" and ws["n"] == ROUNDS and {"p50", "p95"} <= set(ws)
        assert st["segment_stats"], "分段统计缺失"
    finally:
        # 隔离纪律：无论成败都停跑并等 idle，后续用例从干净状态开始
        try:
            _post(port, "/api/stop", {})
        except Exception:
            pass
        _wait_idle(port)


def test_mode_a_honest_unavailable(demo_server):
    """模式 A 一键切换 = 如实不可用面板（不做假运行）。"""
    _wait_idle(demo_server)
    out = _post(demo_server, "/api/run", {"scenario": "gov", "rounds": 1,
                                          "mode": "A"})
    assert out.get("unavailable") is True
    st = _get(demo_server, "/api/status")
    assert st["phase"] == "MODE_A_UNAVAILABLE"
    assert "不可实例化" in "".join(l["msg"] for l in st["log"])


def test_conflict_rejected(demo_server):
    """运行中重复启动 → 409 明确反馈；停止后可再次启动。"""
    import urllib.error
    _wait_idle(demo_server)
    started = _post(demo_server, "/api/run", {"scenario": "enterprise",
                                              "rounds": 2, "mode": "B"})
    assert started["accepted"]
    try:
        _post(demo_server, "/api/run", {"scenario": "gov", "rounds": 2})
        pytest.fail("应 409 拒绝")
    except urllib.error.HTTPError as e:
        assert e.code == 409
    _post(demo_server, "/api/stop", {})
    st = _wait_idle(demo_server)
    assert not st.get("busy"), f"停止后仍未 idle: {st.get('phase')}"
