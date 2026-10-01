"""Beaver 三元组与在线乘法协议（P2 实现；D7 离线假设，docs/02 §1.3）。

安全声明（S8 如实）：三元组由可信离线生成器进程预生成（src/nodes/
offline_triple_gen.py），生成器不在线、不与参与方合谋；本方案为半诚实安全，
不声称恶意安全；三元组一次性使用（fetch 即标记消耗，用后即焚）。

在线协议（2-party，定点域 mod 2^k）：
    open 阶段：P1 广播 d₁=x₁-a₁, e₁=y₁-b₁；P2 广播 d₂=x₂-a₂, e₂=y₂-b₂；
    组合 d=d₁+d₂, e=e₁+e₂（公开值）；
    本地：P1: z₁=c₁+e·a₁+d·b₁+d·e；P2: z₂=c₂+e·a₂+d·b₂
    ⇒ z₁+z₂ = c+e·a+d·b+d·e = (a+e)(b+d) = x·y (mod m)。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from gmssl import sm3

from src.crypto.secret_sharing import DEFAULT_MODULUS


@dataclass
class BeaverTriple:
    gate_id: int = 0
    a: int = 0            # 完整值（生成器侧）
    b: int = 0
    c: int = 0            # c ≡ a·b (mod m)


@dataclass
class BeaverTripleShare:
    """一方的三元组份额视图（真实部署中由生成器分别为两方写独立库存文件）。"""

    gate_id: int = 0
    a: int = 0
    b: int = 0
    c: int = 0


def split_triple(triple: BeaverTriple,
                 modulus: int = DEFAULT_MODULUS) -> Tuple[BeaverTripleShare, BeaverTripleShare]:
    """把完整三元组拆成两方份额（模拟层：真实部署由离线生成器分别写两份库存，
    完整值不出现在任一参与方——D7 口径）。"""
    a1 = int.from_bytes(os.urandom(8), "big") % modulus
    b1 = int.from_bytes(os.urandom(8), "big") % modulus
    c1 = int.from_bytes(os.urandom(8), "big") % modulus
    s1 = BeaverTripleShare(triple.gate_id, a1, b1, c1)
    s2 = BeaverTripleShare(triple.gate_id,
                           (triple.a - a1) % modulus,
                           (triple.b - b1) % modulus,
                           (triple.c - c1) % modulus)
    return s1, s2


class OfflineDepletedError(RuntimeError):
    """在线阶段三元组不足：按协议中止（不得在线补生成——D7 假设边界）。"""


class TripleStore:
    """三元组库存文件（JSON 行 + 全文件 SM3 校验和）。

    生成器进程独占写；在线节点只读消费（fetch 原子标记已用）。
    """

    def __init__(self, store_path: str) -> None:
        self.store_path = Path(store_path)
        self._used: set = set()
        if self.store_path.exists():
            self._load()

    def _load(self) -> None:
        self._used = set()
        data = json.loads(self.store_path.read_text(encoding="utf-8"))
        self._rows = data["triples"]
        digest = sm3.sm3_hash(
            list("".join(json.dumps(t, sort_keys=True) for t in self._rows)
                 .encode("utf-8")))
        if digest != data["checksum"]:
            raise ValueError("三元组库存校验和不匹配——文件被篡改")
        for i, t in enumerate(self._rows):
            if t.get("used"):
                self._used.add(i)

    def remaining(self) -> int:
        return len(self._rows) - len(self._used)

    def fetch(self, n: int = 1, modulus: int = DEFAULT_MODULUS) -> List[BeaverTriple]:
        """取 n 个未用三元组并持久化标记（不足时抛 OfflineDepletedError）。"""
        if self.remaining() < n:
            raise OfflineDepletedError(
                f"剩余 {self.remaining()} < 请求 {n}（D7：不得在线补生成）")
        out = []
        taken: List[int] = []
        for i, t in enumerate(self._rows):
            if len(taken) == n:
                break
            if i in self._used:
                continue
            self._used.add(i)
            taken.append(i)
            out.append(BeaverTriple(gate_id=int(t["gate_id"]),
                                    a=int(t["a"]) % modulus,
                                    b=int(t["b"]) % modulus,
                                    c=int(t["c"]) % modulus))
        self._mark(taken)
        return out

    def _mark(self, indices: List[int]) -> None:
        data = json.loads(self.store_path.read_text(encoding="utf-8"))
        for i in indices:
            data["triples"][i]["used"] = True
        rows = data["triples"]
        data["checksum"] = sm3.sm3_hash(
            list("".join(json.dumps(t, sort_keys=True) for t in rows).encode("utf-8")))
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.store_path)


def beaver_multiply(x1: int, y1: int, shares1: "BeaverTripleShare",
                    x2: int, y2: int, shares2: "BeaverTripleShare",
                    modulus: int = DEFAULT_MODULUS) -> Tuple[int, int]:
    """在线 Beaver 乘法（双方份额视图入参）：返回 (z₁, z₂)，z₁+z₂ ≡ x·y (mod m)。

    通信口径（F9）：open 阶段每方广播 2 个域元素（d、e）＝2×log₂m 比特/门。"""
    d1 = (x1 - shares1.a) % modulus
    e1 = (y1 - shares1.b) % modulus
    d2 = (x2 - shares2.a) % modulus
    e2 = (y2 - shares2.b) % modulus
    d = (d1 + d2) % modulus                  # 公开值 x - a
    e = (e1 + e2) % modulus                  # 公开值 y - b
    z1 = (shares1.c + e * shares1.a + d * shares1.b + d * e) % modulus
    z2 = (shares2.c + e * shares2.a + d * shares2.b) % modulus
    return z1, z2
