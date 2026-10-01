"""密钥全生命周期测试（P4；F5/F6 锚点：状态守卫 / 覆写销毁 / ratchet 窗口）。"""
from __future__ import annotations

import pytest

from src.protocol.keylifecycle import (KeyLifecycleError, KeyLifecycleState,
                                       KeyManager)
from src.crypto.gm_cipher import sm4_gcm_decrypt, sm4_gcm_encrypt

SID = b"\x11" * 16


def _active_km(events=None, window=2) -> KeyManager:
    cb = (lambda e, d: events.append((e, d))) if events is not None else None
    km = KeyManager(accept_window=window, events=cb)
    km.register_intermediate("sm2dh_shared", b"\x22" * 32)
    km.negotiate_root(transcript=b"\x00" * 32)
    km.derive_channel_keys(SID)
    return km


class TestLifecycleStates:
    def test_negotiate_and_derive_states(self):
        events = []
        km = _active_km(events)
        assert km.state == KeyLifecycleState.ACTIVE
        assert any(e[0] == "CHANNEL_KEYS_ACTIVE" for e in events)
        assert km.current_epoch("send") == 0

    def test_illegal_transitions_rejected(self):
        km = KeyManager()
        with pytest.raises(KeyLifecycleError):
            km.derive_channel_keys(SID)          # IDLE 下直接派生
        km.register_intermediate("sm2dh_shared", b"\x22" * 32)
        km.negotiate_root(b"\x00" * 32)
        with pytest.raises(KeyLifecycleError):
            km.negotiate_root(b"\x00" * 32)      # 重复协商
        km.destroy("test")
        with pytest.raises(KeyLifecycleError):
            km.ratchet("send", b"\x00" * 32)     # 终态后操作

    def test_missing_intermediate_rejected(self):
        km = KeyManager()
        with pytest.raises(KeyLifecycleError):
            km.negotiate_root(b"\x00" * 32)      # 未登记 sm2dh_shared


class TestRatchet:
    def test_ratchet_forward_secrecy_semantics(self):
        events = []
        km = _active_km(events)
        k0 = km.channel_key("send")
        km.ratchet("send", b"\x0a" * 32)
        k1 = km.channel_key("send")
        assert k0 != k1 and len(k1) == 16
        assert km.current_epoch("send") == 1
        assert any(e[0] == "KEY_ROTATED" for e in events)

    def test_prev_epoch_window_then_destroyed(self):
        """共存窗口显式语义：窗口内旧纪元密钥可解，窗口满后立即覆写（陷阱 3）。"""
        km = _active_km(window=2)
        old_key = bytes(km.channel_key("recv"))
        km.ratchet("recv", b"\x0b" * 32)
        # 窗口内：旧纪元可用
        assert km.channel_key("recv", 0) == old_key
        ct, tag = sm4_gcm_encrypt(old_key, b"old-epoch-msg", nonce=b"\x01" * 12)
        assert km.on_epoch_message("recv", 1)
        assert km.on_epoch_message("recv", 1)      # 第 2 条 → 窗口满 → 覆写
        with pytest.raises(KeyLifecycleError):
            km.channel_key("recv", 0)              # 旧纪元已销毁
        with pytest.raises(Exception):
            sm4_gcm_decrypt(bytes(km.channel_key("recv")), ct, tag,
                            nonce=b"\x01" * 12)    # 新钥解不开旧密文（前向安全）

    def test_old_epoch_message_outside_window_rejected(self):
        km = _active_km(window=1)
        km.ratchet("recv", b"\x0c" * 32)
        km.on_epoch_message("recv", 1)             # 窗口满 → prev 销毁
        assert km.on_epoch_message("recv", 0) is False   # 旧纪元密钥不存在


class TestDestroy:
    def test_destroy_zeroes_all_registered_material(self):
        events = []
        km = _active_km(events, window=4)
        km.ratchet("send", b"\x0d" * 32)
        detail = km.destroy("test_destroy")
        assert km.state == KeyLifecycleState.DESTROYED
        assert detail["all_zeroed"] and detail["keys_destroyed"] >= 5
        # 派生树中间值（sm2dh_shared/root_key/两方向各 2 纪元）全部在册且清零
        names = {r["name"] for r in detail["registry"]}
        assert {"sm2dh_shared", "root_key", "chan_send_e0", "chan_recv_e0",
                "chan_send_e1"} <= names
        assert any(e[0] == "KEYS_DESTROYED" for e in events)

    def test_destroy_terminal_no_reuse(self):
        km = _active_km()
        km.destroy("x")
        with pytest.raises(KeyLifecycleError):
            km.destroy("again")
        with pytest.raises(KeyLifecycleError):
            km.channel_key("send")

    def test_destroyed_key_cannot_decrypt(self):
        """F6：销毁后系统内旧密钥不可再用于解密（registry 清零 + channel_key 拒绝）。

        Python 语义申报：bytes 拷贝无法回收覆写——本断言针对**系统持有的**
        密钥材料（keylifecycle 模块 docstring 的 S3 纪律声明）。"""
        km = _active_km()
        ct, tag = sm4_gcm_encrypt(km.channel_key("recv"), b"secret",
                                  nonce=b"\x02" * 12)
        km.destroy("test")
        assert all(b == 0 for b in km._registry["chan_recv_e0"])
        with pytest.raises(KeyLifecycleError):
            km.channel_key("recv")               # 系统侧解密路径已失效

    def test_audit_state_has_no_material(self):
        km = _active_km()
        snap = km.audit_state()
        assert snap["state"] == "ACTIVE"
        assert all(k in ("state", "max_epoch", "registered_keys",
                         "zeroed_keys", "directions") for k in snap)
