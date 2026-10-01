"""socket 字节计数封装（F9 网络口径的唯一采集点）。

陷阱 5 要求：流量统计在 socket 发送/接收封装层统一计数，包含协议头。
后续 P3/P4 的传输层必须经由 :class:`CountingSocket` 收发；本模块不含任何
密码逻辑，只做字节与消息计数。
"""
from __future__ import annotations

from typing import Optional

from src.common.perf import NetworkMeter


class CountingSocket:
    """包装一个 socket 风格对象，将收发字节数如实记入 NetworkMeter。

    只包装本项目用到的最小接口：send / sendall / recv。
    计数口径：传给 send 的字节数（含协议头，由调用方组装）与 recv 实际收到的字节数。
    """

    def __init__(self, sock, meter: Optional[NetworkMeter] = None) -> None:
        self._sock = sock
        self.meter = meter if meter is not None else NetworkMeter()

    # ---- 发送 ----
    def send(self, data: bytes, flags: int = 0) -> int:
        n = self._sock.send(data, flags)
        self.meter.on_send(n)
        return n

    def sendall(self, data: bytes, flags: int = 0) -> None:
        self._sock.sendall(data, flags)
        self.meter.on_send(len(data))

    # ---- 接收 ----
    def recv(self, bufsize: int, flags: int = 0) -> bytes:
        data = self._sock.recv(bufsize, flags)
        self.meter.on_recv(len(data))
        return data

    # ---- 透传其余 socket 方法（settimeout、close 等） ----
    def __getattr__(self, name):
        return getattr(self._sock, name)
