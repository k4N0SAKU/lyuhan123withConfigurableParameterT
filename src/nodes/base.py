"""节点基座与链路（P4：client/keynode/infernode 编排的公共设施）。

链路抽象（S8 口径：同机回环，无真实 socket）：
- LoopbackLink：进程内 queue.Queue 对（单元测试/进程内 e2e）；
- MpLink：multiprocessing.Queue 对（多进程编排 e2e，Windows spawn 安全）。
接口：send(bytes) / recv(timeout_s) -> bytes / close()——SecureChannel 仅依赖
该鸭子类型；F9 字节计数由 SecureChannel 统一在帧层执行（含协议头，陷阱 5）。

BaseNode：身份 + 审计文件（按节点分文件）+ 通道表。审计回调把通道/密钥事件
写入本节点 JSONL 链（docs/01 §8.6：P1 主链、P2 副本、P0 客户端链）。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Dict, Optional

from src.protocol.audit_log import AuditLog
from src.protocol.auth import NodeIdentity
from src.protocol.keylifecycle import KeyManager
from src.protocol.session import (SecureChannel, Session, SessionState)

ROLE_CLIENT, ROLE_KEYNODE, ROLE_INFER = 0, 1, 2


class LoopbackLink:
    """进程内全双工链路：两端共享一对反向队列（a.send→b.recv，b.send→a.recv，
    每方向 FIFO 一帧一元素，边界保持）。"""

    def __init__(self, q_send: "queue.Queue", q_recv: "queue.Queue",
                 name: str = "link") -> None:
        self._send_q = q_send
        self._recv_q = q_recv
        self.name = name
        self._open = True

    def send(self, data: bytes) -> None:
        if not self._open:
            raise ConnectionError("链路已关闭")
        self._send_q.put(data)

    def recv(self, timeout_s: float = 30.0) -> bytes:
        return self._recv_q.get(timeout=timeout_s)

    def close(self) -> None:
        self._open = False

    @classmethod
    def create_pair(cls, name: str = "link") -> tuple:
        q_ab, q_ba = queue.Queue(), queue.Queue()
        return (cls(q_ab, q_ba, name + "-a"), cls(q_ba, q_ab, name + "-b"))


class MpLink:
    """跨进程全双工链路（multiprocessing.Queue 对；Windows spawn 兼容）。"""

    def __init__(self, q_send: "mp.Queue", q_recv: "mp.Queue",
                 name: str = "mp-link") -> None:
        self._send_q = q_send
        self._recv_q = q_recv
        self.name = name

    def send(self, data: bytes) -> None:
        self._send_q.put(data)

    def recv(self, timeout_s: float = 30.0) -> bytes:
        return self._recv_q.get(timeout=timeout_s)

    def close(self) -> None:
        pass

    @classmethod
    def create_pair(cls, name: str = "mp-link", ctx=None) -> tuple:
        ctx = ctx or mp.get_context("spawn")
        q_ab, q_ba = ctx.Queue(), ctx.Queue()
        return (cls(q_ab, q_ba, name + "-a"), cls(q_ba, q_ab, name + "-b"))


class BaseNode:
    """节点基座：身份 + 审计链 + 通道表（会话由编排器逐链路建立）。"""

    def __init__(self, node_id: str, role: int, identity: NodeIdentity,
                 ca_pub: bytes, audit_path: Optional[str] = None,
                 clock_ms=None) -> None:
        self.node_id = node_id
        self.role = role
        self.identity = identity
        self.ca_pub = ca_pub
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.audit = AuditLog(actor=node_id, clock_ms=self.clock_ms)
        self.audit_path = audit_path
        self.channels: Dict[str, SecureChannel] = {}
        self.kms: Dict[str, KeyManager] = {}

    def audit_event(self, event: str, detail: dict) -> None:
        self.audit.append(self.node_id, event, detail)

    def flush_audit(self) -> Optional[str]:
        """审计链落盘（按节点分文件）；返回写入的事件数。"""
        if self.audit_path is None:
            return None
        p = Path(self.audit_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.audit.export_json(), encoding="utf-8")
        return str(p)

    def make_channel(self, name: str, session_id: bytes,
                     ratchet_mode: str = "messages",
                     ratchet_threshold: int = 64) -> SecureChannel:
        """建一条到某对端的通道（Session + KeyManager + SecureChannel）。"""
        session = Session(session_id, role=self.role)
        km = KeyManager(events=self.audit_event)
        chan = SecureChannel(
            session, self.identity, km, clock_ms=self.clock_ms,
            ratchet_mode=ratchet_mode, ratchet_threshold=ratchet_threshold,
            audit=self.audit_event)
        self.channels[name] = chan
        self.kms[name] = km
        return chan

    def destroy_all(self, reason: str = "session_end") -> dict:
        """销毁本节点全部会话密钥（终态；逐链审计）。"""
        out = {}
        for name, chan in self.channels.items():
            out[name] = chan.destroy(reason)
        return out

    def state_summary(self) -> dict:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "channels": {n: {"state": c.session.state.name,
                             "send_seq": c.current_seq("send"),
                             "recv_seq": c.current_seq("recv"),
                             "km": c.km.audit_state(),
                             "net": c.meter.snapshot()}
                         for n, c in self.channels.items()},
            "audit_events": len(self.audit.entries),
        }

    def export_report(self) -> dict:
        """节点级报告（编排器汇总用；不含任何密钥材料）。"""
        return {
            "state_summary": self.state_summary(),
            "audit_events": [
                {"seq": e.seq, "ts_ms": e.ts_ms, "event": e.event,
                 "detail": e.detail}
                for e in self.audit.entries
            ],
        }


def save_report(report: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                 encoding="utf-8")
