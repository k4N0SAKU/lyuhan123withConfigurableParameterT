# -*- coding: utf-8 -*-
"""离线出口掩码库存生成器与存储（P9 优化 3a；D7 可信离线假设的延伸）。

动机（通信优化）：D5′ 出口每个转换在线需传 m×Enc(sᵢ)（各 1.64MB）——把
Enc(sᵢ) 的生成移到**离线阶段**：可信生成器预采样 sᵢ 并加密入库，在线阶段
各方只发 zᵢ=aᵢ+sᵢ（8B/槽），合成方从自己的库存取 Enc(sᵢ)。在线出口通信
从 m×密文 降为 m×z 向量（实测见 benchmarks/results/mparty_cost_curve_store.json）。

安全边界（如实申报，docs/02 假设体系内）：
- 生成器**可信且离线**（D7 同款——Beaver 三元组生成器同样知悉完整 a,b,c），
  不在线、不与参与方合谋；生成器被攻破或参与合谋则出口掩码全泄——与三元组
  假设同一信任级别，不属于既有 t=m−1 声明的变更；
- sᵢ 仍逐转换独立均匀采样（os.urandom 拒绝采样，S3），**一次性使用**（fetch
  原子标记已用，复用即破坏 OTP——按协议中止，禁止在线补生成）；
- 合成方库存只含 Enc(sᵢ)（无私钥不可解）；各方库存只含自己的 sᵢ——
  size-(m−1) 合谋仍恰缺一片，门限语义不变；
- 库存按 conversion 计数索引对齐（各方与合成方按同序 fetch_next 消费）。

存储布局（性能教训固化为架构）：条目内含 MB 级密文 hex——若整库单文件，
每次 fetch 的 used 标记都要对全库重算哈希（gmssl 纯 Python SM3 ≈0.25MB/s，
实测单库数分钟）。故改为**清单 + 分条目文件**：清单极小（fetch 只重写清单），
每条目独立文件独立校验和；哈希用 SM3 的 OpenSSL C 实现（hashlib.new('sm3')，
≈388MB/s，与 gmssl 参考实现交叉验证一致），不可用时回退 gmssl。

用法：
    python -m src.nodes.offline_mask_gen --dir data/mask_store --m 3 --count 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from gmssl import sm3 as gmssl_sm3

from src.crypto.ckks_ops import CKKSContext
from src.protocol.conversion import FIXED_ONE, policy_for, sample_exit_mask
from src.protocol.conversion_m import check_m


def _sm3(data: bytes) -> str:
    """SM3 摘要：优先 OpenSSL C 实现（hashlib），回退 gmssl 纯 Python。

    两者在同一输入上必须产出相同摘要（模块级交叉验证一次，见文末）。"""
    try:
        return hashlib.new("sm3", data).hexdigest()
    except Exception:
        return gmssl_sm3.sm3_hash(list(data))


def _sm3_json(obj) -> str:
    return _sm3(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"))


# 交叉验证：C 实现与 gmssl 参考实现对同一输入必须一致（国密合规锚点）
assert _sm3(b"abc") == gmssl_sm3.sm3_hash(list(b"abc")), \
    "hashlib SM3 与 gmssl 参考实现摘要不一致——禁止继续"


def generate_exit_mask_inventory(pub: CKKSContext, dir_path: str, m: int,
                                 count: int) -> dict:
    """离线生成 count 次转换的出口掩码库存（清单 + 分条目文件）。

    产物：
      <dir>/composer_store.json            清单（条目指针+校验和，极小）
      <dir>/composer/enc_{k}.json          第 k 次转换的 m 个 Enc(sᵢ)（holder 序）
      <dir>/party_{i}_store.json           第 i 方清单
      <dir>/party_{i}/s_{k}.json           第 i 方第 k 次转换的 sᵢ 向量
    """
    if m < 2:
        raise ValueError("m 必须 ≥ 2")
    policy = policy_for(pub)
    check_m(policy, m)
    slots = pub.slot_count
    root = Path(dir_path)
    (root / "composer").mkdir(parents=True, exist_ok=True)

    comp_entries, party_entries = [], {i: [] for i in range(1, m + 1)}
    for k in range(count):
        encs, ss = [], {}
        for i in range(1, m + 1):
            s = sample_exit_mask(slots, policy)
            enc = pub.encrypt_vector([si / FIXED_ONE for si in s])
            ss[i] = s
            encs.append(pub.serialize_ct_bytes(enc).hex())
        comp_file = {"index": k, "encs": encs}
        cf = root / "composer" / f"enc_{k}.json"
        cf.write_text(json.dumps(comp_file, ensure_ascii=False), encoding="utf-8")
        comp_entries.append({"index": k, "file": cf.name,
                             "checksum": _sm3_json(comp_file), "used": False})
        for i in range(1, m + 1):
            (root / f"party_{i}").mkdir(parents=True, exist_ok=True)
            s_file = {"index": k, "s": ss[i]}
            sf = root / f"party_{i}" / f"s_{k}.json"
            sf.write_text(json.dumps(s_file, ensure_ascii=False), encoding="utf-8")
            party_entries[i].append({"index": k, "file": sf.name,
                                     "checksum": _sm3_json(s_file), "used": False})
    meta = {"m": m, "count": count, "slots": slots}
    (root / "composer_store.json").write_text(
        json.dumps({**meta, "kind": "composer", "entries": comp_entries,
                    "checksum": _sm3_json(comp_entries)}, ensure_ascii=False),
        encoding="utf-8")
    for i in range(1, m + 1):
        (root / f"party_{i}_store.json").write_text(
            json.dumps({**meta, "kind": "party", "holder": i,
                        "entries": party_entries[i],
                        "checksum": _sm3_json(party_entries[i])}, ensure_ascii=False),
            encoding="utf-8")
    return {**meta, "dir": str(root)}


class ExitMaskStore:
    """出口掩码库存（清单极小；条目独立文件独立校验；fetch 只触本条目）。

    view="party"：fetch_next() → 自己的 sᵢ 向量（int 列表）；
    view="composer"：fetch_next() → [Enc(s₁),…,Enc(s_m)]（bytes，holder 序）。
    一次性使用：fetch 原子标记 used（清单重写为 KB 级），复用即破坏 OTP。
    """

    def __init__(self, path: str, view: str) -> None:
        if view not in ("party", "composer"):
            raise ValueError("view 必须为 party 或 composer")
        self.view = view
        self.path = Path(path)
        self.base = self.path.parent
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.m, self.slots, self.count = data["m"], data["slots"], data["count"]
        rows = data["entries"]
        if _sm3_json(rows) != data["checksum"]:
            raise ValueError(f"{view} 清单校验和不匹配——文件被篡改")
        self.rows = {r["index"]: r for r in rows}
        self.holder = data.get("holder", 0)                  # party 视图的持方
        self._subdir = "composer" if view == "composer" else f"party_{self.holder}"
        self._verified: set = set()
        self._cursor = 0

    def fetch_next(self):
        while self._cursor in self.rows and self.rows[self._cursor].get("used"):
            self._cursor += 1
        if self._cursor not in self.rows:
            raise RuntimeError("出口掩码库存耗尽——按协议中止（D7：不得在线补生成）")
        row = self.rows[self._cursor]
        entry_path = self.base / self._subdir / row["file"]
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        if row["index"] not in self._verified:
            if _sm3_json(entry) != row["checksum"]:
                raise ValueError(f"条目 {row['file']} 校验和不匹配——文件被篡改")
            self._verified.add(row["index"])
        row["used"] = True
        self._persist()
        self._cursor += 1
        if self.view == "party":
            return entry["s"]
        return [bytes.fromhex(h) for h in entry["encs"]]

    def _persist(self) -> None:
        rows = [self.rows[k] for k in sorted(self.rows)]
        payload = {"m": self.m, "count": self.count, "slots": self.slots,
                   "kind": self.view, "holder": self.holder,
                   "entries": rows, "checksum": _sm3_json(rows)}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)                       # 原子替换（TripleStore 同款）


class PartyExitMaskStore(ExitMaskStore):
    """计算方 i 的库存视图：fetch_next() → 自己的 sᵢ 向量（int 列表）。"""

    def __init__(self, path: str) -> None:
        super().__init__(path, "party")


class ComposerExitMaskStore(ExitMaskStore):
    """合成方库存视图：fetch_next() → [Enc(s₁),…,Enc(s_m)]（bytes，holder 序）。"""

    def __init__(self, path: str) -> None:
        super().__init__(path, "composer")


def main(argv: list | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="库存输出目录")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--count", type=int, default=100,
                        help="预生成转换次数（每次数=一次出口转换的量）")
    args = parser.parse_args(argv)
    from src.crypto.ckks_ops import PARAMS_MODE_B
    pub = CKKSContext(PARAMS_MODE_B, public_only=True)
    info = generate_exit_mask_inventory(pub, args.dir, args.m, args.count)
    print(f"exit-mask inventory: m={info['m']} count={info['count']} "
          f"slots={info['slots']} -> {info['dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
