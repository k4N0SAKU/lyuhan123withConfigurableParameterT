"""CKKS 封装（P2 实现；基于 TenSEAL 0.3.14 内置的原生 SEAL 绑定 tenseal.sealapi）。

实现依据（P2 API 探针实测，见 benchmarks/results/primitives_*.json 与工作记录）：
- **本栈（TenSEAL 0.3.14 捆绑 SEAL）参数校验上限 poly_modulus_degree = 2^15**：
  n≥2^16 一律 ``invalid_parameters_insecure``（与链长无关，实测连 [60,40,60]
  极小链亦被拒）；n=2^15 实测最大链 [60]+[40]*19+[60]=880bit 通过（≤881，
  128-bit classical）。P1 §5.3 表 7 据此修订：mode-b 降为 2^15（16384 槽），
  mode-a-demo 在本栈不可实例化（需 OpenFHE 等完整后端，团队决策项）。
- 旋转语义：``Evaluator.rotate_vector(ct, steps)`` 为**左移**（res[s]=v[s+steps]）；
- 显式 rescale：本封装的每个乘法 API 都在乘后调用 ``rescale_to_next``（P1 §5.2
  的深度记账以"每乘恰一次 rescale"为前提）；
- ``Decryptor.invariant_noise_budget`` 不支持 CKKS → 噪声观测用
  ``decrypt_error_rmse``（已知明文对照，P2 实测口径）。

TenSEAL 高层 API（ts.context/ckks_vector）不暴露旋转与显式 rescale，故本模块
直接使用 sealapi 原语——仍在 D6 技术栈（TenSEAL 分发包）内，无新增依赖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from tenseal import sealapi


@dataclass(frozen=True)
class CKKSParams:
    """一套 CKKS 参数（docs/01 §5.3 参数表的代码映射；P2 实测修订版）。"""

    name: str = "mode-b-segment"
    poly_modulus_degree: int = 1 << 15
    coeff_mod_bit_sizes: tuple = (60, 40, 40, 60)
    scale_log2: int = 40
    slots: int = 1 << 14                      # N/2 = 16384
    note: str = ""


# 规格参数表（P2 实测修订——探针确认 2^15 为本栈上限，见模块 docstring）
# 每密文容纳 ⌊16384/1536⌋ = 10 个 1536-槽 token 块（gap-block 布局，docs/01 §6.2）
PARAMS_MODE_B = CKKSParams(
    name="mode-b-segment", poly_modulus_degree=1 << 15,
    coeff_mod_bit_sizes=(60, 40, 40, 60), scale_log2=40, slots=1 << 14,
    note="模式 B 主线：段内深度 2；密文 1.64MB；L=16 需 2 密文（10+6 块）")

# 2^16 评审路径 PoC（P9-1b）：SEC_LEVEL_TYPE.NONE 下 SEAL 接受 n=2^16；
# 安全性由 lattice-estimator 承载（docs/01 §5.3 决策记录，C≈2^130.2≥128）。
# 单密文 3.3MB 装 L=16 全序列 ⇒ 转换次数 96（较 2^15 主线减半）。非提交主线。
PARAMS_MODE_B_2E16 = CKKSParams(
    name="mode-b-2e16-REVIEW-PATH", poly_modulus_degree=1 << 16,
    coeff_mod_bit_sizes=(60, 40, 40, 60), scale_log2=40, slots=1 << 15,
    note="2^16 评审路径 PoC（需 sec_level='NONE' 构造）；密文 3.28MB；转换减半")

# 深链变体（仅用于 S5 精度回归与噪声曲线：19 层 = 本栈最大深度 880bit@2^15）
PARAMS_DEEP19 = CKKSParams(
    name="deep19-regression", poly_modulus_degree=1 << 15,
    coeff_mod_bit_sizes=(60,) + (40,) * 19 + (60,), scale_log2=40, slots=1 << 14,
    note="深度 19（本栈上限）；用于多层链精度回归，非推理配置")

# P4 协议层玩具参数（仅协议/转换测试与演示：密钥生成快、密文 ~KB 级。
# TC128 校验对 CKKS 强制 HomomorphicEncryption.org 合规——N=1024/2048 全链
# 被拒（实测），N=4096 起 100bit 可用。非推理配置，不进入性能口径。）
PARAMS_P4_TOY = CKKSParams(
    name="p4-protocol-toy", poly_modulus_degree=1 << 12,
    coeff_mod_bit_sizes=(30, 30, 40), scale_log2=30, slots=1 << 11,
    note="P4 协议测试/通道 e2e 玩具参数；转换协议全流程等价可验")

# 模式 A 理论参数：**本栈不可实例化**（SEAL 校验表上限 2^15，需 84 层深链）。
# 保留常量供理论推演引用；实例化会抛 ValueError（test 断言该行为）。
PARAMS_MODE_A = CKKSParams(
    name="mode-a-THEORETICAL-NEEDS-OPENFHE", poly_modulus_degree=1 << 17,
    coeff_mod_bit_sizes=(60,) + (40,) * 84 + (60,), scale_log2=40, slots=1 << 16,
    note="本栈不可用：SEAL 校验对 n≥2^16 一律 invalid_parameters_insecure（实测）")


class OpNotAllowedError(PermissionError):
    """public_only 上下文尝试解密/持钥操作（t=1 防线之一，D5 修订）。"""


class CKKSContext:
    """CKKS 上下文与全套同态原语（显式 rescale 口径）。

    密钥纪律（D5 修订）：public_only=True 构造的上下文不含私钥（P1/P2 视角），
    decrypt 会抛 OpNotAllowedError；私钥仅 P0 侧实例化并可序列化保存。
    """

    def __init__(self, params: CKKSParams = PARAMS_MODE_B,
                 public_only: bool = False,
                 keys_dir: Optional[str] = None,
                 sec_level: str = "TC128",
                 galois: bool = True) -> None:
        """sec_level：TC128（默认）/NONE——NONE 仅供 2^16 评审路径 PoC
        （SEAL 校验表止于 N=32768；安全性改由 lattice-estimator 数据承载，
        docs/01 §5.3 决策记录）。galois=False 跳过 Galois 密钥生成
        （2^16 全集 keygen ≈45min；PoC 只验参数接受性/加解密，不做旋转）。"""
        self.params = params
        self.public_only = public_only
        n = params.poly_modulus_degree
        bits = list(params.coeff_mod_bit_sizes)
        self._parms = sealapi.EncryptionParameters(sealapi.SCHEME_TYPE.CKKS)
        self._parms.set_poly_modulus_degree(n)
        self._parms.set_coeff_modulus(sealapi.CoeffModulus.Create(n, bits))
        _lvl = getattr(sealapi.SEC_LEVEL_TYPE, sec_level)
        self._ctx = sealapi.SEALContext(self._parms, True, _lvl)
        if not self._ctx.first_context_data().qualifiers().parameters_set():
            raise ValueError(f"CKKS 参数未通过 SEAL 校验: {params}")
        self._encoder = sealapi.CKKSEncoder(self._ctx)
        self._galois = sealapi.GaloisKeys()
        kg = sealapi.KeyGenerator(self._ctx)
        if public_only and keys_dir is None:
            # 纯公钥上下文：仅生成公钥（不落盘私钥）
            self._pk = sealapi.PublicKey()
            kg.create_public_key(self._pk)
        elif keys_dir is not None:
            self._load_keys(keys_dir, public_only)
            kg = None
        else:
            self._pk = sealapi.PublicKey()
            kg.create_public_key(self._pk)
            self._sk = kg.secret_key()
        if kg is not None and galois:
            kg.create_galois_keys(self._galois)
        if kg is not None:
            self._relin = sealapi.RelinKeys()
            kg.create_relin_keys(self._relin)
        self._encryptor = sealapi.Encryptor(self._ctx, self._pk)
        self._decryptor = None if public_only else sealapi.Decryptor(self._ctx, self._sk)
        self._evaluator = sealapi.Evaluator(self._ctx)
        self.scale = float(2 ** params.scale_log2)

    # ---- 密钥序列化（任务书 1：公钥给 P2、私钥仅 P0）----
    def save_keys(self, keys_dir: str, with_secret: Optional[bool] = None) -> dict:
        """密钥落盘：公钥+伽罗瓦+重线性化必存；私钥仅 P0 侧（with_secret=True）。

        D5 修订口径：P2 只能拿到 with_secret=False 的目录。"""
        if self.public_only:
            raise OpNotAllowedError("public_only 上下文不含可导出的私钥")
        with_secret = (with_secret is True) if with_secret is not None else True
        d = Path(keys_dir)
        d.mkdir(parents=True, exist_ok=True)
        self._pk.save(str(d / "public.seal"))
        self._galois.save(str(d / "galois.seal"))
        self._relin.save(str(d / "relin.seal"))
        saved = ["public", "galois", "relin"]
        if with_secret:
            self._sk.save(str(d / "secret.seal"))
            saved.append("secret")
        return {"dir": str(d), "saved": saved}

    def _load_keys(self, keys_dir: str, public_only: bool) -> None:
        d = Path(keys_dir)
        self._pk = sealapi.PublicKey()
        self._pk.load(self._ctx, str(d / "public.seal"))
        self._galois = sealapi.GaloisKeys()
        self._galois.load(self._ctx, str(d / "galois.seal"))
        self._relin = sealapi.RelinKeys()
        self._relin.load(self._ctx, str(d / "relin.seal"))
        if not public_only:
            self._sk = sealapi.SecretKey()
            self._sk.load(self._ctx, str(d / "secret.seal"))

    # ---- 基础编解码 ----
    @property
    def slot_count(self) -> int:
        return self._encoder.slot_count()

    def _encode(self, values: List[float], scale_exp: Optional[int],
                parms_id=None) -> sealapi.Plaintext:
        vals = list(values) + [0.0] * (self.slot_count - len(values))
        if len(vals) > self.slot_count:
            raise ValueError(f"槽溢出: {len(values)} > {self.slot_count}")
        pt = sealapi.Plaintext()
        scale = float(2 ** (scale_exp if scale_exp is not None
                            else self.params.scale_log2))
        if parms_id is None:
            self._encoder.encode(vals, scale, pt)
        else:
            # 深层密文的 add_plain/multiply_plain 必须在同 level 编码
            self._encoder.encode(vals, list(parms_id), scale, pt)
        return pt

    # ---- 加密 / 解密 ----
    def encrypt_vector(self, values: List[float],
                       scale_exp: Optional[int] = None) -> "CKKSCiphertext":
        ct = sealapi.Ciphertext()
        self._encryptor.encrypt(self._encode(values, scale_exp), ct)
        return CKKSCiphertext(ct, level=0,
                              scale_exp=scale_exp or self.params.scale_log2)

    def encrypt_vector_at_scale(self, values: List[float],
                                scale: float) -> "CKKSCiphertext":
        """以**任意精确 float scale** 加密（P4 转换协议用：rescale 素数非
        精确 2^k，深层密文 scale 有 ~1e-6 相对漂移——掩码密文必须按目标
        密文的精确 scale 编码，否则 _align_scale 的 relabeling 在 OTP 大值
        域引入绝对误差数千（P4 实测，见 conversion.py）。"""
        vals = list(values) + [0.0] * (self.slot_count - len(values))
        if len(vals) > self.slot_count:
            raise ValueError(f"槽溢出: {len(values)} > {self.slot_count}")
        pt = sealapi.Plaintext()
        self._encoder.encode(vals, scale, pt)
        ct = sealapi.Ciphertext()
        self._encryptor.encrypt(pt, ct)
        return CKKSCiphertext(ct, level=0, scale_exp=None)

    def decrypt(self, ct: "CKKSCiphertext") -> List[float]:
        if self.public_only or self._decryptor is None:
            raise OpNotAllowedError("public_only 上下文禁止解密（t=1 防线，D5）")
        pt = sealapi.Plaintext()
        self._decryptor.decrypt(ct.raw, pt)
        return self._encoder.decode_double(pt)

    def noise_budget_bits(self, ct: "CKKSCiphertext") -> int:
        """噪声预算（仅 BFV/BGV 支持）——CKKS 下 SEAL 抛 unsupported scheme。

        P2 实测结论：CKKS 的噪声度量用 :meth:`decrypt_error_rmse`（已知明文
        对照的相对误差 RMSE），见 tests/unit/test_ckks_ops.py 精度回归。"""
        if self.public_only or self._decryptor is None:
            raise OpNotAllowedError("public_only 上下文无法观测噪声预算")
        return self._decryptor.invariant_noise_budget(ct.raw)

    def decrypt_error_rmse(self, ct: "CKKSCiphertext",
                           expected: List[float]) -> float:
        """CKKS 噪声代理指标：解密值与期望明文的相对误差 RMSE（P2 实测口径）。

        相对误差按 |dec−ref|/max(|ref|,1e-9) 定义（docs/01 §5.4 合成误差的
        ε_ckks 观测口径）；public_only 不可用。"""
        got = self.decrypt(ct)
        n = min(len(got), len(expected))
        err = np.array(got[:n]) - np.asarray(expected[:n])
        scale = np.maximum(np.abs(np.asarray(expected[:n])), 1e-9)
        return float(np.sqrt(np.mean((err / scale) ** 2)))

    # ---- 同态原语（每乘必 rescale：深度记账 = 1/乘，docs/01 §5.2）----
    def multiply_plain(self, ct: "CKKSCiphertext", plain_values: List[float],
                       scale_exp: Optional[int] = None,
                       rescale: bool = True) -> "CKKSCiphertext":
        """密文 × 明文（槽乘）。

        rescale=True（默认）：乘后立即 rescale，深度 +1；
        rescale=False：保留放大后的 scale（供 BSGS 同 level 累加/掩码乘），
        调用方负责后续统一 ``rescale_next``（P1 §5.2 深度记账的前提）。
        明文按密文当前 level 编码（深层可加偏置/掩码）。"""
        pt = self._encode(plain_values, scale_exp, ct.raw.parms_id())
        out = sealapi.Ciphertext()
        self._evaluator.multiply_plain(ct.raw, pt, out)
        if rescale:
            self._evaluator.rescale_to_next(out, out)
            return CKKSCiphertext(out, level=ct.level + 1)
        return CKKSCiphertext(out, level=ct.level)

    def rescale_next(self, ct: "CKKSCiphertext") -> "CKKSCiphertext":
        """显式 rescale（深度 +1；供乘法累加后统一调用）。"""
        out = sealapi.Ciphertext()
        self._evaluator.rescale_to_next(ct.raw, out)
        return CKKSCiphertext(out, level=ct.level + 1)

    @staticmethod
    def _align_scale(a: "CKKSCiphertext", b: "CKKSCiphertext") -> None:
        """SEAL 加法/密文乘要求双方 scale 精确相等；rescale 除素数引入 ulp 级
        漂移，按 CKKS 惯例直接赋值对齐（偏差 ≪ 噪声预算，P2 实测口径）。"""
        if a.raw.scale != b.raw.scale:
            b.raw.scale = a.raw.scale

    def multiply_ct(self, a: "CKKSCiphertext", b: "CKKSCiphertext") -> "CKKSCiphertext":
        """密文 × 密文（relinearize + rescale），深度 = max(a,b) + 1。

        调用方需保证两者同 level（不同则先 ``mod_switch_to`` 对齐——残差
        对齐的文档口径），scale 由本方法自动对齐。"""
        self._align_scale(a, b)
        out = sealapi.Ciphertext()
        self._evaluator.multiply(a.raw, b.raw, out)
        self._evaluator.relinearize_inplace(out, self._relin)
        self._evaluator.rescale_to_next(out, out)
        return CKKSCiphertext(out, level=max(a.level, b.level) + 1)

    def add(self, a: "CKKSCiphertext", b: "CKKSCiphertext") -> "CKKSCiphertext":
        self._align_scale(a, b)
        out = sealapi.Ciphertext()
        self._evaluator.add(a.raw, b.raw, out)
        return CKKSCiphertext(out, level=max(a.level, b.level))

    def add_plain(self, ct: "CKKSCiphertext", plain_values: List[float],
                  scale_exp: Optional[int] = None) -> "CKKSCiphertext":
        out = sealapi.Ciphertext()
        pt = self._encode(plain_values, scale_exp, ct.raw.parms_id())
        pt.scale = ct.raw.scale                          # 加法要求精确相等
        self._evaluator.add_plain(ct.raw, pt, out)
        return CKKSCiphertext(out, level=ct.level)

    def negate(self, ct: "CKKSCiphertext") -> "CKKSCiphertext":
        out = sealapi.Ciphertext()
        self._evaluator.negate(ct.raw, out)
        return CKKSCiphertext(out, level=ct.level)

    def sub(self, a: "CKKSCiphertext", b: "CKKSCiphertext") -> "CKKSCiphertext":
        """同态减法（转换出口的掩码消去 Enc(a₁+s) − Enc(s) 用，P3-R1 G 项）。"""
        self._align_scale(a, b)
        out = sealapi.Ciphertext()
        self._evaluator.sub(a.raw, b.raw, out)
        return CKKSCiphertext(out, level=max(a.level, b.level))

    def rotate(self, ct: "CKKSCiphertext", steps: int) -> "CKKSCiphertext":
        """槽旋转：**左移**语义 res[s] = v[s+steps]（P2 探针实测）。"""
        out = sealapi.Ciphertext()
        self._evaluator.rotate_vector(ct.raw, steps, self._galois, out)
        return CKKSCiphertext(out, level=ct.level)

    def mod_switch_to(self, ct: "CKKSCiphertext", level: int) -> "CKKSCiphertext":
        """模切换（降 level，不加噪声预算消耗，用于残差对齐）。

        pybind 无 Ciphertext 拷贝构造——克隆经文件往返（调用频率低：仅残差
        对齐等场景，P3 管线约 2 次/层）。"""
        steps = level - ct.level
        if steps < 0:
            raise ValueError("不能升 level（mod_switch 只能向链底移动）")
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".seal")
        os.close(fd)
        try:
            ct.raw.save(path)
            out = sealapi.Ciphertext()
            out.load(self._ctx, path)
        finally:
            os.unlink(path)
        for _ in range(steps):
            self._evaluator.mod_switch_to_next_inplace(out)
        return CKKSCiphertext(out, level=level)

    # ---- 密文序列化 ----
    def serialize_ct(self, ct: "CKKSCiphertext", path: str) -> None:
        ct.raw.save(path)

    def load_ct(self, path: str, level: int = 0) -> "CKKSCiphertext":
        raw = sealapi.Ciphertext()
        raw.load(self._ctx, path)
        return CKKSCiphertext(raw, level=level)

    def serialize_ct_bytes(self, ct: "CKKSCiphertext") -> bytes:
        """密文 → 字节（通道传输用；P4 转换协议 CONVERT_MASKED_CT/
        RECRYPT_SHARES 载荷）。经临时文件中转（pybind 无内存 save 接口，
        与 mod_switch_to 同一约束）。"""
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".seal")
        os.close(fd)
        try:
            ct.raw.save(path)
            with open(path, "rb") as f:
                return f.read()
        finally:
            os.unlink(path)

    def load_ct_bytes(self, data: bytes, level: int = 0) -> "CKKSCiphertext":
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".seal")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(data)
            raw = sealapi.Ciphertext()
            raw.load(self._ctx, path)
        finally:
            os.unlink(path)
        return CKKSCiphertext(raw, level=level)


class CKKSCiphertext:
    """密文包装：携带 level（链深记账）与所属上下文弱引用。"""

    def __init__(self, raw: sealapi.Ciphertext, level: int = 0,
                 scale_exp: Optional[int] = None) -> None:
        self.raw = raw
        self.level = level
        self.scale_exp = scale_exp

    @property
    def size_bytes(self) -> int:
        import tempfile
        fd, p = tempfile.mkstemp(suffix=".seal")
        os.close(fd)
        try:
            self.raw.save(p)
            return os.path.getsize(p)
        finally:
            os.unlink(p)
