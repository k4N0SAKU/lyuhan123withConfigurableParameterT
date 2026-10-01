"""演示运行器（P7 任务 1 核心）。

流程（一次演示 run）：
  PROVISION → AUTHENTICATING（三链路真实握手，玩具 CKKS 通道）→ SESSION_READY
  → 逐轮 INFERRING（真实管线 n_layer=2 / L=8 密文推理）→ DECRYPTING
  （D5 白名单② + 标签）→ … 轮次耗尽 → DESTROYING（密钥覆写+审计链核验）→ DONE
  任一环节异常/超时 → FAILED（error 携带可展示信息，UI 明确反馈）。

口径纪律（陷阱 2）：所有指标统计一律经 src.common.perf 的统计模块
（_stats_block / NetworkMeter / MemoryTracker）——与 benchmarks/results JSON
同源；基线对照直接读取 p6_full_bench.json（文档同款数字）。

明文不出本地（陷阱 1 配套 + F2 运行时钩子）：每轮扫描 P1/P2 角色全字段视图，
断言输入文本/敏感字段字节 0 命中（与 tests/attack F8 钩子同技法）。
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

from src.common.perf import MemoryTracker, NetworkMeter, _stats_block

REPO = Path(__file__).resolve().parents[2]
P6_JSON = REPO / "benchmarks" / "results" / "p6_full_bench.json"

# 演示管线配置（P3 精度协议同款：截断 2 层 / L=8——runtime 与 L=2 相当）
DEMO_N_LAYER = 2
DEMO_SEQ_TOKENS = 8
ROUND_TIMEOUT_S = 900          # 单轮软超时（CPU 慢机裕量；超时→FAILED UI 反馈）

PHASE_LABELS = {
    "INIT": "初始化",
    "PROVISION": "离线供给（CA/证书/CKKS）",
    "AUTHENTICATING": "认证中",
    "SESSION_READY": "会话建立",
    "INFERRING": "密文推理中",
    "DECRYPTING": "结果解密（白名单）",
    "DESTROYING": "密钥销毁+审计核验",
    "DONE": "完成",
    "FAILED": "失败",
    "MODE_A_UNAVAILABLE": "模式 A 不可用（本栈如实申报）",
}


class DemoTimeout(Exception):
    pass


def _scan_view(obj, needle_hex: str, depth: int = 0, seen: frozenset = frozenset()) -> int:
    """递归扫描对象视图中的明文命中数（hex 字符串/bytes；与 F8 钩子同技法）。"""
    if depth > 6 or id(obj) in seen:
        return 0
    hits = 0
    if isinstance(obj, (bytes, bytearray)):
        if needle_hex and needle_hex in bytes(obj).hex():
            hits += 1
    elif isinstance(obj, str):
        if needle_hex and needle_hex in obj:
            hits += 1
    elif isinstance(obj, dict):
        for k, v in obj.items():
            hits += _scan_view(k, needle_hex, depth + 1, seen) + \
                _scan_view(v, needle_hex, depth + 1, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            hits += _scan_view(v, needle_hex, depth + 1, seen)
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            hits += _scan_view(v, needle_hex, depth + 1, seen | {id(obj)})
    return hits


class DemoRunner:
    """单例运行器（FastAPI 进程内）；heavy init 惰性、跨 run 复用。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.state: dict = {"phase": "INIT", "run_id": 0, "error": None,
                            "rounds": [], "log": []}
        self._pipeline = None                 # 惰性 ModeBPipeline
        self._session = None                  # (client, keynode, infer) 生命周期三节点
        self._baseline = self._load_baseline()

    # ---------- 静态数据 ----------
    @staticmethod
    def _load_baseline() -> dict:
        try:
            d = json.loads(P6_JSON.read_text(encoding="utf-8"))
            rt = d["plaintext"]["bert_latency_full12"]["round_total"]
            return {"source": "benchmarks/results/p6_full_bench.json :: plaintext",
                    "bert_p50_ms": round(rt["p50"], 1),
                    "bert_p95_ms": round(rt["p95"], 1),
                    "note": "同机明文基线（满配 12 层，20 轮）——演示管线为截断 2 层，"
                            "对照仅供量级参考（docs/04 §2/§3）"}
        except Exception:
            return {"source": "unavailable", "bert_p50_ms": None}

    # ---------- 惰性重资产 ----------
    def _ensure_pipeline(self):
        if self._pipeline is None:
            self._log("加载 BERT 管线 + CKKS 上下文（一次性，~60s）…")
            from src.model.loader import BertSentimentPipeline
            from src.model.pipeline import ModeBPipeline, PipelineConfig
            plain = BertSentimentPipeline()
            self._pipeline = ModeBPipeline(
                plain, PipelineConfig(n_layer=DEMO_N_LAYER,
                                      seq_tokens=DEMO_SEQ_TOKENS))
            self._log(f"管线就绪（n_layer={DEMO_N_LAYER}, L={DEMO_SEQ_TOKENS}）")
        return self._pipeline

    def _ensure_session(self):
        """生命周期三节点（玩具 CKKS 通道）；已销毁则重建。"""
        if self._session is not None and self._session[3]:
            return self._session
        import tempfile
        from src.crypto.ckks_ops import PARAMS_P4_TOY
        from src.nodes.orchestrator import build_local_nodes, establish_all
        from src.nodes.provision import provision_demo
        d = tempfile.mkdtemp(prefix="a122_demo_")
        provision_demo(d, ckks_params=PARAMS_P4_TOY)
        client, keynode, infer = build_local_nodes(d, PARAMS_P4_TOY)
        self._set_phase("AUTHENTICATING")
        establish_all(client, keynode, infer, ratchet_threshold=64)
        self._set_phase("SESSION_READY")
        self._log("三链路双向认证 + SM2DH 会话建立完成（真实通道，玩具参数）")
        self._session = (client, keynode, infer, True)
        return self._session

    # ---------- 状态（工作线程与 status() 并发——统一加锁） ----------
    def _log(self, msg: str) -> None:
        with self._lock:
            self.state.setdefault("log", []).append(
                {"ts": round(time.time(), 1), "msg": msg})
            self.state["log"] = self.state["log"][-60:]

    def _set_phase(self, phase: str, detail: str = "") -> None:
        with self._lock:
            self.state["phase"] = phase
            self.state["phase_label"] = PHASE_LABELS.get(phase, phase)
        if detail:
            self._log(f"[{phase}] {detail}")

    # ---------- 明文钩子（F2 运行时证据） ----------
    def _plaintext_scan(self, texts: list) -> dict:
        client, keynode, infer, _ = self._session
        hits = {"P1": 0, "P2": 0}
        needles = [t.encode("utf-8").hex() for t in texts if t]
        for name, view in (("P1", keynode), ("P2", infer)):
            for nd in needles:
                hits[name] += _scan_view(view, nd)
        # 管线侧 P1/P2 角色（转换持有方）
        if self._pipeline is not None:
            for name, role in (("P1", self._pipeline.keynode),
                               ("P2", self._pipeline.infer)):
                for nd in needles:
                    hits[name] += _scan_view(role, nd)
        ok = all(v == 0 for v in hits.values())
        return {"ok": ok, "hits": hits,
                "scope": "P1/P2 全字段视图（生命周期节点+管线角色）",
                "evidence": f"scan hits={hits}；线缆抓包证据见 tests/attack F7-1"}

    # ---------- 对外 API ----------
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run(self, scenario_id: str, sample_index: int, rounds: int,
            mode: str = "B", overrides: dict | None = None) -> dict:
        """启动一次演示（后台线程）。返回 {run_id, mode, accepted}。"""
        if mode != "B":
            # 一键切换 A：如实状态（不做假运行）——模式 A 本栈不可实例化
            if self.busy():
                return {"run_id": self.state.get("run_id", 0), "mode": mode,
                        "accepted": False, "reason": "B 演示运行中，先停止再切换"}
            with self._lock:
                self.state = {
                    "phase": "MODE_A_UNAVAILABLE", "run_id": 0, "error": None,
                    "rounds": [],
                    "log": [{"ts": round(time.time(), 1),
                             "msg": "模式 A（全密文深链）本栈不可实例化：SEAL 校验"
                                    "上限 2^15，需 2^17/84 层链（docs/01 §5.3）；"
                                    "按团队决策保留理论推演，不做模拟冒充。"}],
                    "mode": "A"}
            return {"run_id": 0, "mode": "A", "accepted": True,
                    "unavailable": True}
        with self._lock:
            if self.busy():
                return {"run_id": self.state.get("run_id", 0), "mode": mode,
                        "accepted": False, "reason": "已有演示在运行"}
            self.state = {"phase": "INIT", "run_id": int(time.time()),
                          "error": None, "mode": "B", "rounds": [],
                          "log": [], "input": None,
                          "wall_stats": None, "segment_stats": {},
                          "traffic": None, "memory": None,
                          "plaintext": None, "nodes": {}}
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_inner, daemon=True,
                args=(scenario_id, sample_index, rounds, overrides))
            self._thread.start()
            return {"run_id": self.state["run_id"], "mode": mode, "accepted": True}

    def stop(self) -> dict:
        self._stop.set()
        self._log("收到停止请求（当前轮结束后停止）")
        return {"stopped": True}

    def status(self) -> dict:
        with self._lock:
            st = dict(self.state)
        # 拓扑节点状态
        sess_alive = bool(self._session and self._session[3])
        ph = st.get("phase")
        node_state = {}
        for n in ("P0", "P1", "P2"):
            if ph in ("AUTHENTICATING",):
                node_state[n] = "认证中"
            elif ph in ("SESSION_READY", "INFERRING", "DECRYPTING"):
                node_state[n] = "会话建立" if ph != "INFERRING" else "密文推理中"
            elif ph in ("DESTROYING",):
                node_state[n] = "销毁中"
            elif ph == "DONE":
                node_state[n] = "已销毁"
            elif ph == "FAILED":
                node_state[n] = "失败"
            else:
                node_state[n] = "待机"
        st["nodes"] = node_state
        st["busy"] = self.busy()
        st["baseline"] = self._baseline
        st["pipeline_cfg"] = {"n_layer": DEMO_N_LAYER, "seq_tokens": DEMO_SEQ_TOKENS,
                              "note": "截断演示管线（精度主张以满配口径为准，"
                                      "docs/04 §6；标签对截断模型的可信度见"
                                      " docs/04 §2 注记）"}
        return st

    # ---------- 主流程 ----------
    def _run_inner(self, scenario_id: str, sample_index: int,
                   rounds: int, overrides: dict | None) -> None:
        try:
            from src.demo.scenarios import build_input
            inp = build_input(scenario_id, sample_index, overrides)
            texts = [inp["text"]] + [f["value"] for f in inp["display_fields"] if f["value"]]
            with self._lock:
                self.state["input"] = inp
            self._set_phase("PROVISION")
            pipe = self._ensure_pipeline()
            ch_start = self._channel_snapshot()   # 建立前基线（含握手字节语义：
            self._ensure_session()                # 会话复用时为 0——如实）
            wall_samples: list = []
            seg_samples: dict = {}
            mem_peaks: list = []
            net_start = pipe.stats.net.snapshot()
            with self._lock:
                self.state["traffic"] = {"bytes_sent": 0, "bytes_recv": 0}

            for i in range(rounds):
                if self._stop.is_set():
                    self._log(f"第 {i+1} 轮前停止请求生效")
                    break
                t_round = time.perf_counter()
                with self._lock:
                    self.state["current_round"] = i + 1
                    self.state["rounds_total"] = rounds
                self._set_phase("INFERRING", f"第 {i+1}/{rounds} 轮")
                with MemoryTracker() as mem:
                    try:
                        r = pipe.classify(inp["text"])
                    except Exception as exc:
                        raise RuntimeError(f"第 {i+1} 轮推理异常: {exc}") from exc
                    wall_ms = (time.perf_counter() - t_round) * 1000.0
                if wall_ms > ROUND_TIMEOUT_S * 1000:
                    raise DemoTimeout(f"第 {i+1} 轮耗时 {wall_ms/1000:.0f}s 超过 "
                                      f"软超时 {ROUND_TIMEOUT_S}s")
                peak = mem.result()["peak_delta_bytes"]
                wall_samples.append(wall_ms)
                mem_peaks.append(peak)
                for name, v in (r["segments_ms"] or {}).items():
                    seg_samples.setdefault(name, []).append(v)

                self._set_phase("DECRYPTING", "白名单②最终解密 + 标签")
                events = pipe._decrypt_events
                final_ok = any(e.get("kind") == "final_output" for e in events)
                net_now = pipe.stats.net.snapshot()
                ch_now = self._channel_snapshot()
                light = self._plaintext_scan(texts)
                with self._lock:
                    self.state["rounds"].append({
                        "round": i + 1,
                        "label": r["label"], "prob": round(r["prob"], 4),
                        "wall_ms": round(wall_ms, 1),
                        "conversions": r["conversions"],
                        "mpc_gates": r["mpc_gates"],
                        "peak_rss_delta_bytes": peak,
                        "final_decrypt_ok": final_ok,
                        "plaintext_ok": light["ok"],
                        "elapsed_s": round(time.perf_counter() - t_round, 1),
                    })
                    # 指标条：一律经 perf 统计模块（陷阱 2）
                    self.state["wall_stats"] = _stats_block(wall_samples)
                    self.state["segment_stats"] = {
                        k: _stats_block(v) for k, v in sorted(seg_samples.items())}
                    self.state["memory"] = {
                        "peak_delta_bytes_p50": int(_stats_block(mem_peaks)["p50"]),
                        "peak_delta_bytes_max": int(_stats_block(mem_peaks)["max"]),
                        "rounds": len(mem_peaks)}
                    self.state["traffic"] = {
                        "bytes_sent": net_now["bytes_sent"] - net_start["bytes_sent"],
                        "bytes_recv": net_now["bytes_recv"] - net_start["bytes_recv"],
                        "channel_bytes_sent": ch_now[0] - ch_start[0],
                        "channel_bytes_recv": ch_now[1] - ch_start[1]}
                    self.state["plaintext"] = light

            self._set_phase("DESTROYING", "通道密钥两遍覆写 + 审计链核验")
            client, keynode, infer, alive = self._session
            destroy_report = {}
            if alive:
                for n in (client, keynode, infer):
                    destroy_report[n.node_id] = {
                        k: v["all_zeroed"] for k, v in n.destroy_all("demo_run").items()}
                self._session = (client, keynode, infer, False)
            from src.protocol.audit_log import verify_audit_entries
            audit_ok = all(verify_audit_entries(n.audit.entries)[0]
                           for n in (client, keynode, infer))
            with self._lock:
                self.state["destroy"] = destroy_report
                self.state["audit_chain_ok"] = audit_ok
            done = len(self.state.get("rounds", []))
            self._set_phase("DONE", f"{done} 轮完成，审计链核验={'通过' if audit_ok else '失败'}")
            if done < rounds and not self._stop.is_set():
                self.state["error"] = f"仅完成 {done}/{rounds} 轮"
                self._set_phase("FAILED", self.state["error"])
        except Exception as exc:
            with self._lock:
                self.state["error"] = f"{type(exc).__name__}: {exc}"
            self._set_phase("FAILED", self.state["error"])
            self._log(traceback.format_exc(limit=6))

    def _channel_snapshot(self) -> tuple:
        if not self._session or not self._session[3]:
            return (0, 0)
        sent = recv = 0
        for n in self._session[:3]:
            for ch in n.channels.values():
                m = ch.meter.snapshot()
                sent += m["bytes_sent"]
                recv += m["bytes_recv"]
        return (sent, recv)
