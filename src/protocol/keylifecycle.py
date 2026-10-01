"""密钥全生命周期管理（docs/01 §8；F5/F6 的实现载体，P4 实现）。

ratchet 规则（docs/01 §8.3）：
    K_{i+1} = SM3(K_i ‖ H_transcript_i)      双方向独立密钥链
    H_transcript_i = SM3(H_{i-1} ‖ msg_header_i)   会话消息头哈希链
前向安全论证：SM3 单向性 ⇒ 攻破 K_{i+1} 不可恢复 K_i；销毁即覆写（S3）。

新旧密钥共存窗口（P4 陷阱 3 的显式语义）：
    发送方在触发条件到达时先发 KEY_ROTATE 再切换发送密钥；接收方收到
    KEY_ROTATE 后立即前向派生接收密钥。因 FIFO 传输严格有序，正常路径无
    并存；为容忍非 FIFO 传输的乱序到达，接收侧保留上一纪元密钥直至本纪元
    收满 ``ACCEPT_WINDOW_MESSAGES`` 条后**立即覆写**——窗口外到达的旧纪元
    消息因密钥已销毁而以 MAC_INVALID 拒绝并审计告警。

销毁语义（docs/01 §8.5）：登记表内全部密钥材料（含 KDF 中间值）逐条
「随机覆写 + 清零」两遍，状态置 DESTROYED（终态不可逆），逐条核对结果入审计。
Python 局限声明：bytes/str 拷贝无法覆写，密钥一律以 bytearray 驻留，仅在调用
密码原语边界做即时 bytes 转换（短生命周期拷贝，风险可接受并如实申报）。
"""
from __future__ import annotations

import enum
import os
from typing import Callable, Dict, List, Optional

from src.crypto.gm_cipher import sm3_hash, sm3_kdf

RATCHET_RULE = "K_{i+1} = SM3(K_i || H_transcript_i)"
TRANSCRIPT_RULE = "H_i = SM3(H_{i-1} || msg_header_i)"
ROTATE_INTERVAL_ROUNDS = 64      # 每 64 条业务消息触发一次 ratchet（可配置）

CHANNEL_KDF_SALT = b"A1-22-CHAN"
ROOT_KDF_SALT = b"A1-22-ROOT"
SM4_KEY_LEN = 16


class KeyLifecycleState(enum.IntEnum):
    """密钥生命周期状态（与 SessionState 分离但联动守卫）。"""

    IDLE = 0
    NEGOTIATED = 1        # 根会话密钥已派生，通道密钥未启用
    ACTIVE = 2            # 通道密钥使用中
    ROTATING = 3          # 轮换事务进行中（新旧密钥并存窗口）
    DESTROYED = 4         # 已覆写销毁（终态）


class KeyLifecycleError(Exception):
    """状态守卫拒绝 / 销毁后操作。携带 ErrorCode 数值供 ERR 消息引用。"""

    def __init__(self, error_code: int, detail: str) -> None:
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail


def _overwrite_bytes(buf: bytearray) -> None:
    """两遍覆写：随机一遍 + 全零一遍（docs/01 §8.5）。"""
    for i in range(len(buf)):
        buf[i] = os.urandom(1)[0]
    for i in range(len(buf)):
        buf[i] = 0


class KeyManager:
    """会话密钥全生命周期：协商 → 派生 → ratchet 轮换 → 覆写销毁。

    实现纪律（S3）：随机数一律 os.urandom/secrets；密钥以 bytearray 驻留，
    销毁 = 显式清零 + 状态置 DESTROYED + 审计事件；禁止硬编码任何密钥。
    审计经 events 回调外发（节点侧接 AuditLog；单测收集到列表），
    事件 detail 只含名称/长度/纪元——不含任何密钥材料。
    """

    def __init__(self, accept_window: int = 8,
                 events: Optional[Callable[[str, dict], None]] = None) -> None:
        self.state = KeyLifecycleState.IDLE
        self.epoch = 0
        self.accept_window = max(1, int(accept_window))
        self._events = events
        # 密钥登记表（陷阱 2）：所有派生树节点（含中间值）登记于此，
        # destroy 时逐条覆写核对。
        self._registry: Dict[str, bytearray] = {}
        self._zeroed: Dict[str, bool] = {}
        # 每方向密钥链：{"epoch": int, "key": bytearray,
        #                "prev": {"epoch", "key", "since_rotation"} | None}
        self.directions: Dict[str, dict] = {}

    # ---- 内部 ----
    def _emit(self, event: str, detail: dict) -> None:
        if self._events is not None:
            self._events(event, detail)

    def _register(self, name: str, material: bytes) -> bytearray:
        buf = bytearray(material)
        self._registry[name] = buf
        self._zeroed[name] = False
        return buf

    def _require(self, *states: KeyLifecycleState, op: str) -> None:
        if self.state not in states:
            raise KeyLifecycleError(
                8,  # ErrorCode.STATE_ILLEGAL
                f"{op} 在状态 {self.state.name} 下非法（允许: "
                f"{[s.name for s in states]}）")

    # ---- 生命周期 ----
    def register_intermediate(self, name: str, material: bytes) -> None:
        """外部 KDF 中间值（如 SM2DH 共享坐标）登记入销毁清单。"""
        self._register(name, material)

    def negotiate_root(self, transcript: bytes) -> None:
        """SM2DH 协商 → SM3-KDF 派生根会话密钥（状态 IDLE→NEGOTIATED）。

        共享秘密须先经 :meth:`register_intermediate`（"sm2dh_shared"）登记，
        本方法只消费登记表内的材料——保证中间值必然进入销毁清单。"""
        self._require(KeyLifecycleState.IDLE, op="negotiate_root")
        if "sm2dh_shared" not in self._registry:
            raise KeyLifecycleError(10, "negotiate_root 需先登记 sm2dh_shared")
        root = sm3_kdf(bytes(self._registry["sm2dh_shared"]), 32,
                       salt=ROOT_KDF_SALT + transcript)
        self._register("root_key", root)
        self.state = KeyLifecycleState.NEGOTIATED
        self._emit("KEY_NEGOTIATED", {"root_key_len": len(root)})

    def derive_channel_keys(self, session_id: bytes,
                            send_label: str = "d0",
                            recv_label: str = "d1") -> None:
        """根密钥 → 双方向 SM4-GCM 通道密钥（NEGOTIATED→ACTIVE）。

        send_label/recv_label 为**方向规范标签**（由 node_id 全序决定，见
        SecureChannel._key_agreement）：同一条方向的两端必须派生同一密钥——
        A 的 send 与 B 的 recv 用同一 label，故盐一致。（若用端点本地名
        "send"/"recv" 作盐，两端同方向会派生出不同密钥——P4 实测踩坑。）"""
        self._require(KeyLifecycleState.NEGOTIATED, op="derive_channel_keys")
        for direction, label in (("send", send_label), ("recv", recv_label)):
            key = sm3_kdf(bytes(self._registry["root_key"]), SM4_KEY_LEN,
                          salt=CHANNEL_KDF_SALT + label.encode() + session_id)
            buf = self._register(f"chan_{direction}_e0", key)
            self.directions[direction] = {
                "epoch": 0, "key": buf, "prev": None,
            }
        self.state = KeyLifecycleState.ACTIVE
        self._emit("CHANNEL_KEYS_ACTIVE",
                   {"directions": ["send", "recv"],
                    "labels": [send_label, recv_label]})

    def channel_key(self, direction: str, epoch: Optional[int] = None
                    ) -> bytes:
        """当前（或指定纪元）通道密钥的即时 bytes 视图（仅密码原语调用边界）。

        epoch 缺省取当前纪元；epoch=当前-1 且仍在共存窗口内返回上一纪元。"""
        if self.state not in (KeyLifecycleState.ACTIVE,
                              KeyLifecycleState.ROTATING):
            raise KeyLifecycleError(8, f"channel_key 在 {self.state.name} 下不可用")
        entry = self.directions[direction]
        if epoch is None or epoch == entry["epoch"]:
            return bytes(entry["key"])
        if entry["prev"] is not None and epoch == entry["prev"]["epoch"]:
            return bytes(entry["prev"]["key"])
        raise KeyLifecycleError(7, f"纪元 {epoch} 密钥不存在或已销毁")

    def current_epoch(self, direction: str) -> int:
        return self.directions[direction]["epoch"] if direction in self.directions else -1

    def ratchet(self, direction: str, transcript_hash: bytes) -> int:
        """K_{i+1} = SM3(K_i ‖ H_i)（ACTIVE→ROTATING→ACTIVE；旧密钥立即覆写）。

        返回新纪元号。共存窗口语义见模块 docstring：旧密钥不立即销毁而是降级
        为 prev（计数清零），由 :meth:`on_epoch_message` 在窗口满后覆写。"""
        self._require(KeyLifecycleState.ACTIVE, KeyLifecycleState.ROTATING,
                      op="ratchet")
        entry = self.directions[direction]
        old_epoch = entry["epoch"]
        new = sm3_hash(bytes(entry["key"]) + transcript_hash)[:SM4_KEY_LEN]
        prev_state = self.state
        self.state = KeyLifecycleState.ROTATING
        entry["prev"] = {"epoch": old_epoch, "key": entry["key"],
                         "since_rotation": 0}
        name = f"chan_{direction}_e{old_epoch + 1}"
        entry["key"] = self._register(name, new)
        entry["epoch"] = old_epoch + 1
        self.epoch = max(self.epoch, entry["epoch"])
        self.state = KeyLifecycleState.ACTIVE
        self._emit("KEY_ROTATED", {
            "direction": direction, "old_epoch": old_epoch,
            "new_epoch": entry["epoch"], "prev_retained": prev_state is not None,
            "rule": RATCHET_RULE,
        })
        return entry["epoch"]

    def on_epoch_message(self, direction: str, epoch: int) -> bool:
        """接收侧记账：本纪元消息计数；窗口满 → 覆写上一纪元密钥。

        返回 True 表示该纪元密钥当前可用（本纪元恒真；上一纪元在窗口内真）。"""
        if self.state != KeyLifecycleState.ACTIVE:
            raise KeyLifecycleError(8, f"on_epoch_message 在 {self.state.name} 下非法")
        entry = self.directions[direction]
        if epoch == entry["epoch"]:
            if entry["prev"] is not None:
                entry["prev"]["since_rotation"] += 1
                if entry["prev"]["since_rotation"] >= self.accept_window:
                    old = entry["prev"]
                    _overwrite_bytes(old["key"])
                    self._zeroed[f"chan_{direction}_e{old['epoch']}"] = True
                    entry["prev"] = None
                    self._emit("PREV_EPOCH_DESTROYED", {
                        "direction": direction, "epoch": old["epoch"],
                        "window": self.accept_window})
            return True
        if entry["prev"] is not None and epoch == entry["prev"]["epoch"]:
            return True
        return False

    def destroy(self, reason: str) -> dict:
        """任意状态 → DESTROYED：覆写全部密钥材料并写审计（终态不可逆）。

        返回逐条核对快照（名称/长度/是否全零——不含材料本身）。"""
        if self.state == KeyLifecycleState.DESTROYED:
            raise KeyLifecycleError(8, "destroy 只能执行一次（终态）")
        report: List[dict] = []
        for name, buf in self._registry.items():
            _overwrite_bytes(buf)
            all_zero = all(b == 0 for b in buf)
            self._zeroed[name] = all_zero
            report.append({"name": name, "len": len(buf), "zeroed": all_zero})
        for entry in self.directions.values():
            if entry.get("prev") is not None:
                _overwrite_bytes(entry["prev"]["key"])
                entry["prev"] = None
        self.directions.clear()
        self.state = KeyLifecycleState.DESTROYED
        detail = {"reason": reason, "keys_destroyed": len(report),
                  "all_zeroed": all(r["zeroed"] for r in report),
                  "registry": report}
        self._emit("KEYS_DESTROYED", detail)
        return detail

    def audit_state(self) -> dict:
        """供审计日志记录的密钥状态快照（不含密钥材料本身）。"""
        return {
            "state": self.state.name,
            "max_epoch": self.epoch,
            "registered_keys": len(self._registry),
            "zeroed_keys": sum(1 for v in self._zeroed.values() if v),
            "directions": {d: {"epoch": e["epoch"],
                               "prev_epoch": (e["prev"]["epoch"]
                                              if e.get("prev") else None)}
                           for d, e in self.directions.items()},
        }
