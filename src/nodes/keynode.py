"""P1 密钥服务节点（public_only；D5：无任何 CKKS 私钥材料）。

职责（docs/01 §2/§7.4）：入口分享 y₁ 接收与保管、出口再随机化准备
（fresh 掩码 s + OTP 分享 z = a₁+s + Enc(s)）。P1 全程只见 OTP 掩码值
（t=1 论证：P1 知 y₁=x+r 不知 r，知 z=a₁+s 不知 a₂/s 分量）。
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from src.crypto.ckks_ops import CKKSContext, CKKSParams
from src.nodes.base import ROLE_KEYNODE, BaseNode
from src.protocol.auth import NodeIdentity
from src.protocol.conversion import exit_keynode_prepare


class KeyNode(BaseNode):
    """P1：密钥服务。入口分享保管 + 出口再随机化准备。"""

    def __init__(self, identity: NodeIdentity, ca_pub: bytes,
                 ckks_public_dir: str, params: CKKSParams,
                 audit_path: Optional[str] = None,
                 clock_ms: Optional[Callable[[], int]] = None) -> None:
        super().__init__(identity.node_id, ROLE_KEYNODE, identity, ca_pub,
                         audit_path=audit_path, clock_ms=clock_ms)
        self.ctx = CKKSContext(params, public_only=True,
                               keys_dir=ckks_public_dir)
        self.entry_shares: dict = {}          # request_id → y₁（定点 mod 2⁶⁴）

    def take_entry_share(self, request_id: bytes, y1: List[int]) -> None:
        """接收 P0 分发的掩码解密值 y₁（P1 的入口分享）。"""
        self.entry_shares[request_id] = y1
        self.audit_event("ENTRY_SHARE_TAKEN",
                         {"request_id": request_id.hex(), "n": len(y1)})

    def exit_prepare(self, request_id: bytes,
                     const_mult: int = 1) -> Tuple[List[int], bytes]:
        """出口准备：对（可选常数乘后的）a₁ 加掩码 s 并加密。

        const_mult：MPC 中间对分享的公开常数乘（本地操作，双方同乘保持
        分享一致性——e2e 的 2·x 演示用）。"""
        a1 = [(v * const_mult) % (1 << 64) for v in
              self.entry_shares.pop(request_id)]
        z, enc_s = exit_keynode_prepare(self.ctx, a1)
        self.audit_event("EXIT_PREPARED",
                         {"request_id": request_id.hex(), "const": const_mult})
        return z, enc_s
