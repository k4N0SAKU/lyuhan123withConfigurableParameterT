"""规格锚定测试（P2 更新）：协议层骨架仍 NotImplementedError；密码层为 P2
实现，参数锚定断言改为 P2 实测修订值（docs/01 §5.3 表 7 修订版）。

本测试是规格的"可执行锚点"——规格文档修订而未同步代码时在此失败。
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest

from src.crypto.ckks_ops import (PARAMS_DEEP19, PARAMS_MODE_A, PARAMS_MODE_B,
                                 CKKSContext)
from src.model.ops.packing import DEFAULT_SPEC, PackingScheme, PackingSpec
from src.model.ops.nonlinear_approx import (EXP_SPEC, GELU_SPEC, INV_SQRT_SPEC,
                                            INV_SPEC, MAX_SPEC)
from src.protocol import messages as M
from src.protocol.audit_log import AuditLog
from src.protocol.auth import Authenticator, NodeIdentity
from src.protocol.keylifecycle import (ROTATE_INTERVAL_ROUNDS, KeyLifecycleState,
                                       KeyManager)
from src.protocol.session import SecureChannel, Session, SessionState


# ---- 枚举与状态机完整性（P4 实现，锚点不变）----

class TestEnums:
    def test_session_states(self):
        assert {s.name for s in SessionState} == {
            "IDLE", "NEGOTIATED", "ACTIVE", "ROTATING", "DESTROYED", "ERROR"}
        assert {s.name for s in KeyLifecycleState} == {
            "IDLE", "NEGOTIATED", "ACTIVE", "ROTATING", "DESTROYED"}

    def test_msg_types_structure(self):
        ctrl = {m for m in M.MsgType if 0x0100 <= m.value < 0x0200}
        data = {m for m in M.MsgType if 0x0200 <= m.value < 0x0300}
        err = {m for m in M.MsgType if m.value >= 0x0300}
        assert {"HELLO", "AUTH_CHALLENGE", "AUTH_RESPONSE", "KEY_NEGOTIATE",
                "KEY_ROTATE", "KEY_DESTROY"} <= {m.name for m in ctrl}
        assert {"INFER_REQUEST", "INFER_RESULT", "THRESHOLD_PARTIAL",
                "CONVERT_MASKED_CT", "RECRYPT_SHARES"} <= {m.name for m in data}
        assert err == {M.MsgType.ERR}

    def test_error_codes_cover_five_attacks(self):
        codes = {e.name for e in M.ErrorCode}
        assert {"CERT_INVALID", "CERT_EXPIRED", "AUTH_FAILURE",
                "MAC_INVALID", "SEQ_REPLAY", "TS_EXPIRED",
                "DECRYPT_FAILURE"} <= codes

    def test_ratchet_constants_documented(self):
        from src.protocol import keylifecycle as K
        assert "SM3" in K.RATCHET_RULE and "H_transcript" in K.RATCHET_RULE
        assert ROTATE_INTERVAL_ROUNDS == 64


# ---- 消息字段级定义（docs/01 §3 字段表的代码映射）----

class TestMessages:
    def test_common_header_field_widths_documented(self):
        h = M.CommonHeader(msg_type=int(M.MsgType.HELLO),
                           session_id=b"\x01" * M.SESSION_ID_LEN, seq=7,
                           timestamp_ms=1_700_000_000_000, payload_len=10)
        assert h.version == M.PROTOCOL_VERSION == 1
        assert len(h.session_id) == 16

    def test_envelope_and_payload_registry(self):
        env = M.MessageEnvelope(
            header=M.CommonHeader(msg_type=int(M.MsgType.AUTH_RESPONSE)),
            payload_type="AuthResponsePayload",
            payload=dataclasses.asdict(M.AuthResponsePayload()),
            auth_kind=int(M.AuthKind.SM2_SIG))
        assert env.payload_type in M.PAYLOAD_REGISTRY
        assert M.PAYLOAD_REGISTRY["RecryptSharesPayload"] is M.RecryptSharesPayload
        recrypt = M.RecryptSharesPayload()
        assert recrypt.enc_mask is not None and recrypt.masked_share_ints is not None

    def test_convert_payload_carries_mask_semantics(self):
        conv = M.ConvertMaskedCtPayload()
        assert hasattr(conv, "masked_ct") and hasattr(conv, "level")
        rec = M.RecryptSharesPayload()
        assert hasattr(rec, "enc_mask") and hasattr(rec, "masked_share_ints")

    def test_all_payloads_are_dataclasses(self):
        for name, cls in M.PAYLOAD_REGISTRY.items():
            assert dataclasses.is_dataclass(cls), name
            dataclasses.asdict(cls())


# ---- 协议层实现锚点（P4 已实现；行为测试见 test_auth/test_session/
#      test_keylifecycle/test_audit_log/test_conversion 与 tests/attack）----

IMPLEMENTED_TARGETS = [
    ("SecureChannel",
     lambda: SecureChannel(Session(b"\x00" * 16, role=0),
                           NodeIdentity(), KeyManager()),
     ["establish", "send_message", "recv_message", "current_seq"]),
    ("Session", lambda: Session(b"\x00" * 16, role=1),
     ["transition", "transcript_hash"]),
    ("Authenticator", lambda: Authenticator(b"\x00" * 65, NodeIdentity()),
     ["verify_cert", "issue_challenge", "respond", "verify_response"]),
    ("KeyManager", lambda: __import__("src.protocol.keylifecycle",
                                      fromlist=["KeyManager"]).KeyManager(),
     ["negotiate_root", "derive_channel_keys", "ratchet", "destroy",
      "audit_state"]),
    ("AuditLog", lambda: AuditLog(), ["append", "verify_chain", "export_json"]),
]


class TestProtocolLayerImplemented:
    """P4 锚点：协议层接口已从骨架进入实现（可调用、非 NotImplementedError）。

    各方法的行为正确性由专门测试文件覆盖；本类只锚定"实现已落地"。"""

    @pytest.mark.parametrize("label,factory,methods", IMPLEMENTED_TARGETS,
                             ids=[t[0] for t in IMPLEMENTED_TARGETS])
    def test_methods_no_longer_not_implemented(self, label, factory, methods):
        import inspect as _inspect
        obj = factory()
        for name in methods:
            src = _inspect.getsource(getattr(obj, name))
            assert "NotImplementedError" not in src, f"{label}.{name} 仍是骨架"

    def test_serialize_envelope_implemented(self):
        import dataclasses as _dc
        env = M.MessageEnvelope(payload_type="ErrorPayload",
                                payload={"error_code": 1})
        env.header.payload_len = len(M.payload_to_json(env.payload))
        env.auth_value = b"\x00" * 16
        data = M.serialize_envelope(env)
        env2 = M.deserialize_envelope(data)
        assert env2.payload == {"error_code": 1}


# ---- CKKS 参数表锚定（docs/01 §5.3 表 7，P2 实测修订值）----

class TestCKKSParamsSpec:
    def test_mode_b_segment(self):
        """P2 探针实测：本栈 SEAL 校验上限 2^15 → mode-b 主线 16384 槽。"""
        p = PARAMS_MODE_B
        assert p.poly_modulus_degree == 1 << 15
        assert p.coeff_mod_bit_sizes == (60, 40, 40, 60)
        assert len(p.coeff_mod_bit_sizes) - 2 == 2      # 段内最深两个线性层
        assert p.scale_log2 == 40
        assert sum(p.coeff_mod_bit_sizes) == 200
        assert p.slots == 1 << 14

    def test_deep19_regression_chain(self):
        """本栈最大深度链：19 层 = 880bit ≤ 881（128-bit classical@2^15）。"""
        p = PARAMS_DEEP19
        assert p.poly_modulus_degree == 1 << 15
        assert len(p.coeff_mod_bit_sizes) - 2 == 19
        assert sum(p.coeff_mod_bit_sizes) == 880

    def test_mode_a_theoretical(self):
        """评审 A/探针结论：2^17 被本栈拒绝，mode-a 仅保留理论参数。"""
        assert PARAMS_MODE_A.poly_modulus_degree == 1 << 17
        assert "不可实例化" in PARAMS_MODE_A.name or "OPENFHE" in PARAMS_MODE_A.name
        assert len(PARAMS_MODE_A.coeff_mod_bit_sizes) - 2 == 84

    def test_mode_b_actually_instantiates(self):
        ctx = CKKSContext(PARAMS_MODE_B, public_only=True)
        assert ctx.slot_count == 1 << 14

    def test_mode_a_fails_to_instantiate(self):
        with pytest.raises(ValueError):
            CKKSContext(PARAMS_MODE_A)


# ---- 打包与近似规格锚定（docs/01 §6/§7，P2 修订值）----

class TestOpsSpec:
    def test_packing_default_spec(self):
        assert DEFAULT_SPEC.block_elems == 768
        assert DEFAULT_SPEC.gap == 2
        assert DEFAULT_SPEC.slots == 1 << 14
        assert DEFAULT_SPEC.blocks_per_ct == 10
        assert int(PackingScheme.DIAGONAL_BSGS) == 1

    def test_poly_specs_shapes(self):
        """P2-R1 B1 实测校准后的规格锚定（P1 理论界作废，见 minimax.py）。"""
        assert GELU_SPEC.name == "gelu_deg15"
        assert GELU_SPEC.depth == 4 and GELU_SPEC.num_mults == 15
        assert GELU_SPEC.max_error <= 5.5e-3            # 实测 5.067e-3（P1 目标 5e-3 的 +1.3%，域内最优）
        assert "exp_deg" in EXP_SPEC.name
        assert EXP_SPEC.domain[0] <= -10.0           # max 减法后的单侧域
        assert EXP_SPEC.depth == 4
        assert EXP_SPEC.max_error <= 1e-2                # 实测 6.8e-5
        assert MAX_SPEC.depth == 1 and MAX_SPEC.num_mults >= 2
        assert INV_SPEC.extra.get("newton_rounds", 5) >= 5      # P1 的 3-4 轮不收敛
        assert INV_SQRT_SPEC.extra.get("newton_rounds", 3) >= 3
        assert INV_SQRT_SPEC.max_error < 0.2

    def test_linear_signature_takes_spec(self):
        from src.model.ops.packing import linear_cipher
        sig = inspect.signature(linear_cipher)
        assert "spec" in sig.parameters and "diagonals" in sig.parameters
