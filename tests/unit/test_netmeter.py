"""socket 字节计数封装单元测试（陷阱 5：流量统计含协议头，由封装层统一计数）。"""
from __future__ import annotations

import socket

from src.common.netmeter import CountingSocket
from src.common.perf import NetworkMeter


class TestCountingSocket:
    def test_send_recv_counted(self):
        meter = NetworkMeter()
        a, b = socket.socketpair()
        try:
            ca = CountingSocket(a, meter)
            cb = CountingSocket(b, meter)
            payload = b"x" * 100 + b"HEADER"  # 调用方组装的"含协议头"报文
            n = ca.send(payload)
            assert n == len(payload)
            assert meter.bytes_sent == len(payload)
            assert meter.messages_sent == 1
            got = cb.recv(len(payload) + 1)
            assert got == payload
            assert meter.bytes_recv == len(payload)
            assert meter.messages_recv == 1
        finally:
            a.close()
            b.close()

    def test_sendall_counted(self):
        meter = NetworkMeter()
        a, b = socket.socketpair()
        try:
            ca = CountingSocket(a, meter)
            cb = CountingSocket(b, meter)
            ca.sendall(b"AB" * 50)
            assert meter.bytes_sent == 100
            assert cb.recv(200) == b"AB" * 50
        finally:
            a.close()
            b.close()

    def test_passthrough_attrs(self):
        a, _ = socket.socketpair()
        try:
            cs = CountingSocket(a, NetworkMeter())
            cs.settimeout(1.0)  # __getattr__ 透传
            assert cs.gettimeout() == 1.0
            assert cs.meter.bytes_sent == 0
        finally:
            a.close()

    def test_default_meter(self):
        a, _ = socket.socketpair()
        try:
            cs = CountingSocket(a)
            cs.send(b"hi")
            assert isinstance(cs.meter, NetworkMeter)
            assert cs.meter.bytes_sent == 2
        finally:
            a.close()
