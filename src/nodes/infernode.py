"""P2 推理服务节点（public_only；D5：无任何 CKKS 私钥材料）。

职责：密文计算主体 + 转换入口掩码（fresh r → ct′=ct+Enc(r)）+ 转换出口
密文域合成（w=(z+a₂) mod 2⁶⁴ → Enc(w)−Enc(s)=Enc(v)）。P2 视角全程只有
OTP 掩码值（t=1 论证：P2 知 −r 与 w=v+s，不知 v）。
"""
from __future__ import annotations

from typing import Callable, List, Optional

from src.crypto.ckks_ops import CKKSContext, CKKSCiphertext, CKKSParams
from src.crypto.secret_sharing import DEFAULT_MODULUS
from src.nodes.base import ROLE_INFER, BaseNode
from src.protocol.auth import NodeIdentity
from src.protocol.conversion import (entry_make_masked_ct, entry_p2_share,
                                     exit_infernode_compose)


class InferNode(BaseNode):
    """P2：推理服务。密文计算 + 转换入口掩码 + 出口密文域合成。"""

    def __init__(self, identity: NodeIdentity, ca_pub: bytes,
                 ckks_public_dir: str, params: CKKSParams,
                 audit_path: Optional[str] = None,
                 clock_ms: Optional[Callable[[], int]] = None) -> None:
        super().__init__(identity.node_id, ROLE_INFER, identity, ca_pub,
                         audit_path=audit_path, clock_ms=clock_ms)
        self.ctx = CKKSContext(params, public_only=True,
                               keys_dir=ckks_public_dir)
        self._pending_r: dict = {}            # request_id → r（入口掩码）

    # ---- 转换入口：密文域加 Enc(r)（D5 白名单① 的对侧） ----
    def entry_mask(self, ct: CKKSCiphertext, request_id: bytes) -> CKKSCiphertext:
        masked, r = entry_make_masked_ct(self.ctx, ct)
        self._pending_r[request_id] = r
        self.audit_event("ENTRY_MASKED", {"request_id": request_id.hex(),
                                          "level": masked.level})
        return masked

    def entry_p2_share(self, request_id: bytes) -> List[int]:
        """P2 的入口分享：−r mod 2⁶⁴。"""
        share = entry_p2_share(self._pending_r.pop(request_id))
        return share

    # ---- MPC 中间（S8 模拟口径：真实 MPC 定点求值为 P3 xfail 项） ----
    @staticmethod
    def share_const_mult(share: List[int], c: int) -> List[int]:
        """公开常数乘（本地零交互；双方同乘保持分享一致性）。"""
        return [(v * c) % DEFAULT_MODULUS for v in share]

    # ---- 转换出口：密文域合成 fresh 顶层密文 ----
    def exit_compose(self, request_id: bytes, enc_s_bytes: bytes,
                     z: List[int], a2: List[int]) -> CKKSCiphertext:
        ct = exit_infernode_compose(self.ctx, enc_s_bytes, z, a2)
        self.audit_event("EXIT_COMPOSED",
                         {"request_id": request_id.hex(), "level": ct.level})
        return ct
