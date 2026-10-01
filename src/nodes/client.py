"""P0 客户端节点（D5 单钥持有方：CKKS 私钥仅存在于此节点）。

密钥纪律（D5 修订）：私钥上下文私有属性 `_secret_ctx`，对外**唯一**解密入口
是 DecryptionWhitelist 门（masked_conversion / final_output 两类）——节点上
不存在任何绕过白名单的解密路径（tests/attack 断言）。
"""
from __future__ import annotations

from typing import Callable, List, Optional

from src.crypto.ckks_ops import CKKSContext, CKKSCiphertext, CKKSParams
from src.nodes.base import ROLE_CLIENT, BaseNode
from src.protocol.auth import NodeIdentity
from src.protocol.conversion import DecryptionWhitelist, from_fixed


class ClientNode(BaseNode):
    """P0：数据属主。嵌入加密 / 掩码解密（白名单①）/ 最终解密（白名单②）。"""

    def __init__(self, identity: NodeIdentity, ca_pub: bytes,
                 ckks_secret_dir: str, params: CKKSParams,
                 audit_path: Optional[str] = None,
                 clock_ms: Optional[Callable[[], int]] = None) -> None:
        super().__init__(identity.node_id, ROLE_CLIENT, identity, ca_pub,
                         audit_path=audit_path, clock_ms=clock_ms)
        self._secret_ctx = CKKSContext(params, public_only=False,
                                       keys_dir=ckks_secret_dir)
        self.whitelist = DecryptionWhitelist(self._secret_ctx,
                                             node_id=identity.node_id,
                                             audit=self.audit_event)

    # ---- 白名单①：掩码转换值解密（CONVERT_MASKED_CT 处理） ----
    def masked_decrypt(self, ct_masked: CKKSCiphertext,
                       request_id: bytes) -> List[int]:
        return self.whitelist.decrypt("masked_conversion", ct_masked,
                                      request_id)

    # ---- 白名单②：最终输出解密（INFER_RESULT 处理） ----
    def final_decrypt(self, ct: CKKSCiphertext,
                      request_id: bytes) -> List[float]:
        vals = self.whitelist.decrypt("final_output", ct, request_id)
        return [from_fixed(v) for v in vals]

    def encrypt_vector(self, values: List[float]) -> CKKSCiphertext:
        """P0 本地嵌入/输入加密（深度 0）。"""
        return self._secret_ctx.encrypt_vector(values)
