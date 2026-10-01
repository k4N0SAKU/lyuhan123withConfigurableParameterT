"""m 方计算节点（P8；D5′ 协议族的通道层载体，零改动既有节点文件）。

与 KeyNode/InferNode 的关系：m 方族里 P1..Pm 全部是「public_only 计算方」，
差异只在角色位——index=1 锚点（额外接收 P0 分发的 y）、index=2 合成方
（出口汇总）、其余纯掩码方。本节点统一承载三者，通道面/审计面/销毁面
完全继承 BaseNode（SM2DH 会话、SM4-GCM 帧、ratchet、SM3 审计链不变）。

安全边界（D5′ 申报）：节点上不存在任何 CKKS 私钥材料（public_only 强制）；
本方掩码片 rᵢ/sᵢ 只存在本进程内存（audit 不记录片值）。
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from src.crypto.ckks_ops import CKKSContext, CKKSCiphertext, CKKSParams
from src.crypto.secret_sharing import DEFAULT_MODULUS
from src.nodes.base import BaseNode, ROLE_INFER
from src.protocol.auth import NodeIdentity
from src.protocol.conversion_m import ComputePartyRole


class MaskPartyNode(BaseNode):
    """m 方计算节点：掩码链成员 + 份额持有方（锚点/合成方为角色位）。"""

    def __init__(self, identity: NodeIdentity, ca_pub: bytes,
                 ckks_public_dir: str, params: CKKSParams, index: int, m: int,
                 audit_path: Optional[str] = None,
                 clock_ms: Optional[Callable[[], int]] = None) -> None:
        super().__init__(identity.node_id, ROLE_INFER, identity, ca_pub,
                         audit_path=audit_path, clock_ms=clock_ms)
        self.ctx = CKKSContext(params, public_only=True,
                               keys_dir=ckks_public_dir)
        self.party_index = index
        self.party_count = m
        self.role = ComputePartyRole(self.ctx, index=index, m=m,
                                     is_anchor=(index == 1),
                                     audit=self.audit_event)

    @property
    def is_anchor(self) -> bool:
        """锚点位（P1）：接收 P0 分发的掩码解密值 y。"""
        return self.role.is_anchor

    # ---- 入口：掩码链加片 / 份额形成 ----
    def entry_apply_piece(self, ct: CKKSCiphertext,
                          request_id: bytes) -> CKKSCiphertext:
        masked = self.role.entry_apply_piece(ct, request_id)
        self.audit_event("MPARTY_ENTRY_MASKED",
                         {"request_id": request_id.hex(),
                          "index": self.party_index, "level": masked.level})
        return masked

    def entry_own_share(self, request_id: bytes,
                        y: Optional[List[int]] = None) -> List[int]:
        share = self.role.entry_own_share(request_id, y=y)
        self.audit_event("MPARTY_ENTRY_SHARE",
                         {"request_id": request_id.hex(),
                          "index": self.party_index, "n": len(share)})
        return share

    # ---- MPC 中间（S8 模拟口径延续）：公开常数乘 ----
    @staticmethod
    def share_const_mult(share: List[int], c: int) -> List[int]:
        """公开常数乘（本地零交互；各方同乘保持分享一致性）。"""
        return [(v * c) % DEFAULT_MODULUS for v in share]

    # ---- 出口：片准备（发合成方） / 合成（仅合成方 P2 调用） ----
    def set_result_share(self, request_id: bytes, a_i: List[int]) -> None:
        self.role.set_result_share(request_id, a_i)

    def exit_piece(self, request_id: bytes) -> Tuple[List[int], bytes]:
        z, enc_s = self.role.exit_piece(request_id)
        self.audit_event("MPARTY_EXIT_PIECE",
                         {"request_id": request_id.hex(),
                          "index": self.party_index})
        return z, enc_s
