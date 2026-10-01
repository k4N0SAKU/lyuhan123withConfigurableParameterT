"""SM3 哈希链审计日志（docs/01 §8.6；F5 审计防篡改的实现载体，P4 实现）。

链式结构：entry_i.hash = SM3(entry_{i-1}.hash ‖ canonical(entry_i))，创世
prev_hash = 32B 零。防篡改性质：改动任一条目任何字段 ⇒ 该条 hash 重算不符；
删除/重排 ⇒ 序列号断裂 / prev_hash 不接（verify_audit_lines 逐项定位）。

纪律：审计事件**禁止记录密钥材料**（只记名称/长度/状态快照）；时间戳经注入
时钟获取（陷阱 1：测试可注入虚拟时钟）。
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from src.crypto.gm_cipher import sm3_hash

ZERO32 = bytes(32)


def _default_clock_ms() -> int:
    return int(time.time() * 1000)


def canonical_event_bytes(ev: "AuditEvent") -> bytes:
    """链上覆盖段：{seq, ts_ms, actor, event, detail} 规范 JSON（sort_keys）。

    prev_hash/hash 不入覆盖段（hash 覆盖自身除外，链式递归）。"""
    return json.dumps(
        {"seq": ev.seq, "ts_ms": ev.ts_ms, "actor": ev.actor,
         "event": ev.event, "detail": ev.detail},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


@dataclass
class AuditEvent:
    """审计事件（canonical 编码后入链；禁止记录密钥材料明文）。"""

    seq: int = 0                             # 链内单调序号（从 1 起）
    ts_ms: int = 0                           # UTC 毫秒
    actor: str = ""                          # "P0" / "P1" / "P2" / "system"
    event: str = ""                          # 事件类型（如 AUTH_OK / KEY_ROTATED / DESTROYED）
    detail: dict = field(default_factory=dict)
    prev_hash: bytes = ZERO32
    hash: bytes = ZERO32

    def compute_hash(self, prev_hash: bytes) -> bytes:
        self.prev_hash = prev_hash
        self.hash = sm3_hash(prev_hash + canonical_event_bytes(self))
        return self.hash


class AuditLog:
    """单节点审计链（每节点一个 JSONL 文件；docs/01 §8.6：P1 主链 + P2 副本）。"""

    def __init__(self, actor: str = "system",
                 clock_ms: Optional[Callable[[], int]] = None) -> None:
        self.actor = actor
        self._clock_ms = clock_ms or _default_clock_ms
        self.entries: List[AuditEvent] = []
        # P7：单节点可有多条链路并发 establish（各线程均可能写审计）——
        # append 必须原子（seq/prev_hash 读改写），否则哈希链被交错破坏
        # （冒烟实证：audit_ok=False，两链路并发同节点竞态）
        self._append_lock = threading.Lock()

    def append(self, actor: str, event: str, detail: dict | None = None) -> AuditEvent:
        """追加事件并接入哈希链（canonical 序列化 -> SM3；线程安全）。"""
        with self._append_lock:
            ev = AuditEvent(seq=len(self.entries) + 1, ts_ms=self._clock_ms(),
                            actor=actor or self.actor, event=event,
                            detail=dict(detail or {}))
            prev = self.entries[-1].hash if self.entries else ZERO32
            ev.compute_hash(prev)
            self.entries.append(ev)
            return ev

    def verify_chain(self) -> bool:
        """全链重算校验；任一条目被改 -> False（F5 审计防篡改测试）。"""
        ok, _ = verify_audit_entries(self.entries)
        return ok

    def export_json(self) -> str:
        """JSONL（每行一个事件；hash/prev_hash 以 hex 编码）。"""
        lines = []
        for ev in self.entries:
            lines.append(json.dumps(
                {"seq": ev.seq, "ts_ms": ev.ts_ms, "actor": ev.actor,
                 "event": ev.event, "detail": ev.detail,
                 "prev_hash": ev.prev_hash.hex(), "hash": ev.hash.hex()},
                sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else "")

    @classmethod
    def from_json(cls, text: str) -> "AuditLog":
        """从 JSONL 重建（不重算链——校验用 verify_audit_lines）。"""
        log = cls()
        for line in text.splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            ev = AuditEvent(seq=d["seq"], ts_ms=d["ts_ms"], actor=d["actor"],
                            event=d["event"], detail=d.get("detail") or {},
                            prev_hash=bytes.fromhex(d["prev_hash"]),
                            hash=bytes.fromhex(d["hash"]))
            log.entries.append(ev)
        return log


def verify_audit_entries(entries: List[AuditEvent]) -> Tuple[bool, List[str]]:
    """全链校验并定位问题（F5 测试断言的证据输出）。

    检测三类完整性破坏：
    - 篡改：entry_i 的 canonical 重算 hash ≠ 记录 hash；
    - 删除：seq 断裂（k → k+2）/ prev_hash 不接；
    - 重排：seq 非严格递增。
    """
    errors: List[str] = []
    prev_hash = ZERO32
    expect_seq = 1
    prev_ts = -1
    for ev in entries:
        if ev.seq != expect_seq:
            errors.append(f"seq 断裂: 期望 {expect_seq} 实得 {ev.seq}（疑似删除/重排）")
        if ev.ts_ms < prev_ts:
            errors.append(f"seq {ev.seq}: 时间戳回退（疑似重排）")
        if ev.prev_hash != prev_hash:
            errors.append(f"seq {ev.seq}: prev_hash 不接（链断裂）")
        recomputed = sm3_hash(ev.prev_hash + canonical_event_bytes(ev))
        if recomputed != ev.hash:
            errors.append(f"seq {ev.seq}: hash 重算不符（该条目被篡改）")
        expect_seq = ev.seq + 1
        prev_ts = ev.ts_ms
        prev_hash = ev.hash
    return (not errors), errors
