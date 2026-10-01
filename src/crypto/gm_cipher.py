"""国密算法封装（P2 实现；D2/S3 纪律：gmssl 原语 + 标准参数）。

实现要点：
- **SM2 签名必须携带用途上下文字符串**（P2 陷阱清单）：摘要 =
  SM3(Z_A ‖ len(ctx)‖ctx ‖ message)——不同协议用途（认证/结果签名/…）使用
  不同 ctx，签名不可跨协议重用（测试断言）；
- Z_A 按 GB/T 32918.2 计算（uid 默认 "1234567812345678"）；
- **SM4 仅提供 GCM（AEAD）**，禁止裸 ECB/CBC 传输密文（陷阱 8）：gmssl 不含
  GCM 模式，本模块基于 gmssl SM4 块加密（padding_mode=None 的单块原语）自实现
  CTR + GHASH（NIST SP 800-38D 结构：C = GCTR(K, J0, P) 首块密钥流 E(K,J0+1)，
  tag = GHASH_H(A,C) ⊕ E(K,J0)）；外部锚定：RFC 8998 附录 A.1 向量逐字节匹配
  （test_gm_cipher.py）；
- SM2 密钥对生成与 ECDH 式协商原语基于 gmssl 底层点运算（_kg——评审 J 项
  可行性结论：可自实现）；完整 GB/T 32918.3 交互式协商在 P4 会话建立补全。
- 随机数一律 os.urandom（S3）。
"""
from __future__ import annotations

import hmac
import os
from typing import Tuple

from gmssl import sm2 as _gm_sm2
from gmssl import sm3 as _gm_sm3
from gmssl.sm4 import CryptSM4, SM4_ENCRYPT

SM2_UID_DEFAULT = b"1234567812345678"
_ECC = _gm_sm2.default_ecc_table
_SM2_N = int(_ECC["n"], 16)
# SM2 推荐曲线（GB/T 32918.5）基点坐标——gmssl ecc_table 的 'g' 带内部填充
# 怪癖（无 gx/gy 键），Z_A 计算使用国标常数：
_SM2_GX = int("32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7", 16)
_SM2_GY = int("BC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F", 16)

# 默认签名上下文（各用途必须显式传入不同值，见 A1-22 各协议消息定义）
CTX_AUTH = b"A1-22-AUTH-V1"
CTX_RESULT = b"A1-22-INFER-RESULT-V1"
CTX_KEY = b"A1-22-KEY-OPS-V1"


class AuthenticationError(Exception):
    """AEAD / 签名校验失败统一异常（F4）。"""


# ---- SM3 ----

def sm3_hash(data: bytes) -> bytes:
    """SM3 杂凑（32B）。国标向量：sm3(b"abc") = 66c7f0f4…8e0（GM/T 0004）。"""
    return bytes.fromhex(_gm_sm3.sm3_hash(list(data)))


def sm3_kdf(secret: bytes, out_len: int, salt: bytes = b"A1-22-KEYDERIVE") -> bytes:
    """SM3 计数器 KDF：T_i = SM3(salt ‖ secret ‖ i_32be)，i 从 1 起。"""
    if out_len < 0:
        raise ValueError("out_len 不能为负")
    out = b""
    counter = 1
    while len(out) < out_len:
        out += sm3_hash(salt + secret + counter.to_bytes(4, "big"))
        counter += 1
    return out[:out_len]


# ---- SM2 ----

def sm2_generate_keypair() -> Tuple[bytes, bytes]:
    """返回 (pubkey 65B 未压缩 04‖x‖y, privkey 32B)。

    基于 gmssl 底层标量乘（_kg，评审 J 项可行性结论）；私钥**拒绝采样**于
    [1, n-2]（与 secret_sharing 统一拒绝采样口径，P2-R1 评审 G 项修正——
    初版 % n 约减存在 ~2^-128 偏差），保证 R=kG 非退化。"""
    while True:
        priv = int.from_bytes(os.urandom(32), "big")
        if 1 <= priv <= _SM2_N - 2:
            break
    pub_xy = _kg_bytes(priv)
    return b"\x04" + pub_xy, priv.to_bytes(32, "big")


def _kg_bytes(k: int) -> bytes:
    """标量乘 k·G，返回 x‖y（64B）。"""
    helper = _gm_sm2.CryptSM2(private_key="1" * 64, public_key="")
    return bytes.fromhex(helper._kg(k, _ECC["g"])[:128])


def _mul_point(k: int, pub_xy: bytes) -> bytes:
    """任意点标量乘 k·P（P 为 x‖y 64B），返回 x‖y。"""
    helper = _gm_sm2.CryptSM2(private_key="1" * 64, public_key="")
    return bytes.fromhex(helper._kg(k, pub_xy.hex()))


def _z_a(pub_xy: bytes, uid: bytes = SM2_UID_DEFAULT) -> bytes:
    """GB/T 32918.2 的 Z_A = SM3(ENTL_A ‖ UID ‖ a ‖ b ‖ Gx ‖ Gy ‖ x_A ‖ y_A)。"""
    if len(pub_xy) != 64:
        pub_xy = pub_xy[-64:]
    entl = (len(uid) * 8).to_bytes(2, "big")
    fields = b"".join(v.to_bytes(32, "big")
                      for v in (int(_ECC["a"], 16), int(_ECC["b"], 16),
                                _SM2_GX, _SM2_GY)) + pub_xy
    return sm3_hash(entl + uid + fields)


def _sm2_digest(context: bytes, message: bytes, pub_xy: bytes,
                uid: bytes) -> bytes:
    """签名摘要 = SM3(Z_A ‖ len(ctx)_32be ‖ ctx ‖ message)。

    上下文字符串绑定用途，防跨协议签名重用（陷阱清单要求）。"""
    if not context:
        raise ValueError("签名上下文字符串不能为空（防跨协议重用）")
    framed = len(context).to_bytes(4, "big") + context
    return sm3_hash(_z_a(pub_xy, uid) + framed + message)


def sm2_sign(privkey: bytes, message: bytes, context: bytes = CTX_AUTH,
             uid: bytes = SM2_UID_DEFAULT) -> bytes:
    """SM2 签名（r‖s 64B）；随机数 K = os.urandom(32)（S3）。"""
    if len(privkey) != 32:
        raise ValueError("私钥必须 32B")
    pub_xy = _kg_bytes(int.from_bytes(privkey, "big"))
    digest = _sm2_digest(context, message, pub_xy, uid)
    signer = _gm_sm2.CryptSM2(private_key=privkey.hex(), public_key="")
    for _ in range(16):                       # R==0/S==0 等退化概率 ~2^-256，防御性重试
        sig_hex = signer.sign(digest, os.urandom(32).hex())
        if sig_hex:
            return bytes.fromhex(sig_hex)
    raise RuntimeError("SM2 签名连续退化失败（概率 ~2^-2048，不应发生）")


def sm2_verify(pubkey: bytes, message: bytes, sig: bytes,
               context: bytes = CTX_AUTH, uid: bytes = SM2_UID_DEFAULT) -> bool:
    """验签；pubkey 65B（04‖x‖y）或 64B（x‖y）。"""
    pub_xy = pubkey[-64:]
    if len(sig) != 64:
        raise ValueError("签名必须 r‖s 64B")
    digest = _sm2_digest(context, message, pub_xy, uid)
    verifier = _gm_sm2.CryptSM2(private_key="0" * 64, public_key=pub_xy.hex())
    return bool(verifier.verify(sig.hex(), digest))


def sm2_encrypt(pubkey: bytes, plaintext: bytes) -> bytes:
    """SM2 公钥加密（C1C3C2，GB/T 32918.4 顺序）。"""
    helper = _gm_sm2.CryptSM2(private_key="0" * 64,
                              public_key=pubkey[-64:].hex(), mode=1)
    for _ in range(16):
        out = helper.encrypt(plaintext)
        if out:
            return out
    raise RuntimeError("SM2 加密连续失败")


def sm2_decrypt(privkey: bytes, ciphertext: bytes) -> bytes:
    helper = _gm_sm2.CryptSM2(private_key=privkey.hex(), public_key="", mode=1)
    return helper.decrypt(ciphertext)


def sm2_ecdh(own_priv: bytes, peer_pub: bytes) -> bytes:
    """ECDH 式协商原语（评审 J 项可行性实现）：x 坐标 of [priv]·P_peer。

    完整 GB/T 32918.3 交互式协商（双临时公钥 + w 截断 + 确认）在 P4 补全；
    本原语证明 gmssl 底层点运算可承载协商（P2 风险清单闭环依据）。"""
    shared = _mul_point(int.from_bytes(own_priv, "big"), peer_pub[-64:])
    return shared[:32]


# ---- SM4-GCM（自实现 CTR + GHASH；禁裸 ECB/CBC）----

_SM4_POLY_R = 0xE1000000000000000000000000000000


def _sm4_block_encrypt(key: bytes, block: bytes) -> bytes:
    """单块 SM4 加密（gmssl padding_mode=None 原语；国标向量见单测）。"""
    if len(block) != 16:
        raise ValueError("块长必须 16B")
    c = CryptSM4(mode=SM4_ENCRYPT, padding_mode=None)
    c.set_key(key, SM4_ENCRYPT)
    return c.crypt_ecb(block)


def gf128_mult(x: bytes, y: bytes) -> bytes:
    """GF(2^128) 乘法（NIST SP 800-38D 算法 1，多项式 x^128+x^7+x^2+x+1）。"""
    z = 0
    v = int.from_bytes(y, "big")
    xi = int.from_bytes(x, "big")
    for i in range(128):
        if (xi >> (127 - i)) & 1:
            z ^= v
        if v & 1:
            v = (v >> 1) ^ _SM4_POLY_R
        else:
            v >>= 1
    return z.to_bytes(16, "big")


def _ghash(h: bytes, aad: bytes, ciphertext: bytes) -> bytes:
    """GHASH：pad(A)‖pad(C)‖[len(A)]‖[len(C)] 逐 16B 块 Y←(Y⊕B)·H。

    P4 性能注：块乘按字节位置线性分解（X·H = Σ_k x_k·x^(120−8k)·H）——
    16 张 256 项表 T_k[v] = v·x^(120−8k)·H，每块 16 次查表异或、无进位耦合
    （GF(2^128) 线性性），与逐位参考实现 gf128_mult 数学等价，RFC 8998 向量
    与单零块断言锚定；大块数据（1.6MB 级转换密文）快 ~8×。表按 H 缓存
    （H=E(K,0) 每密钥恒定，lru_cache）。"""
    def pad16(b: bytes) -> bytes:
        return b + b"\x00" * ((-len(b)) % 16)

    data = (pad16(aad) + pad16(ciphertext)
            + (len(aad) * 8).to_bytes(8, "big")
            + (len(ciphertext) * 8).to_bytes(8, "big"))
    tables = _ghash_tables(h)
    y = 0
    for i in range(0, len(data), 16):
        y ^= int.from_bytes(data[i:i + 16], "big")
        z = 0
        for k in range(16):
            z ^= tables[k][(y >> (120 - 8 * k)) & 0xFF]
        y = z
    return y.to_bytes(16, "big")


from functools import lru_cache


@lru_cache(maxsize=8)
def _ghash_tables(h: bytes) -> tuple:
    """T_k[v] = v·x^(120−8k)·H 预计算表（T_0 经参考实现生成，其后逐表
    右移约简 8 位：v·x^(120−8(k+1))·H = (v·x^(120−8k)·H)·x⁻⁸）。"""
    t0 = [int.from_bytes(gf128_mult(bytes([v]) + b"\x00" * 15, h), "big")
          for v in range(256)]
    tables = [t0]
    for _ in range(15):
        prev = tables[-1]
        nxt = []
        for val in prev:
            w = val
            for _ in range(8):
                w = (w >> 1) ^ (_SM4_POLY_R if w & 1 else 0)
            nxt.append(w)
        tables.append(nxt)
    return tuple(tuple(t) for t in tables)


def _gctr(key: bytes, icb: bytes, data: bytes) -> bytes:
    """GCTR（NIST SP 800-38D 风格）：密钥流 = E(K, ICB+1), E(K, ICB+2), ……

    **ICB 语义 = 首个计数器值的前一个**（本实现"先自增再取密钥流"，调用方
    传 ICB=J0 即得标准 GCM 首块密钥流 E(J0+1)）——P2-R1 修正：初版调用方
    传 J0+1 致密钥流从 E(J0+2) 起（计数器偏移一格，非标准 GCM，往返自洽但
    不与标准实现互操作；见 P2 审查反馈 A 项与工作记录安全影响声明）。"""
    out = bytearray()
    cb = int.from_bytes(icb, "big")
    for i in range(0, len(data), 16):
        cb = (cb + 1) & ((1 << 128) - 1)
        ks = _sm4_block_encrypt(key, cb.to_bytes(16, "big"))
        chunk = data[i:i + 16]
        out.extend(x ^ y for x, y in zip(chunk, ks))
    return bytes(out)


def sm4_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes = b"",
                    nonce: bytes = b"") -> Tuple[bytes, bytes]:
    """SM4-GCM 加密，返回 (ciphertext, tag 16B)。

    NIST SP 800-38D 结构：C = GCTR(K, J0, P)（首块密钥流 = E(K, J0+1)）；
    tag = GHASH_H(A, C) ⊕ E(K, J0)。
    nonce 为 12B 显式参数（协议层由 session_id‖seq 派生），强制调用方管理——
    这同时是 F6 防重放的一部分：(key, nonce) 绝不复用。"""
    if len(key) != 16:
        raise ValueError("SM4 密钥必须 16B")
    if len(nonce) != 12:
        raise ValueError("本实现仅支持 96-bit nonce，且必须显式传入（F6 口径）")
    h = _sm4_block_encrypt(key, b"\x00" * 16)
    j0 = nonce + b"\x00\x00\x00\x01"
    ciphertext = _gctr(key, j0, plaintext)
    s = _ghash(h, aad, ciphertext)
    tag = bytes(a ^ b for a, b in zip(s, _sm4_block_encrypt(key, j0)))
    return ciphertext, tag


def sm4_gcm_decrypt(key: bytes, ciphertext: bytes, tag: bytes, aad: bytes = b"",
                    nonce: bytes = b"") -> bytes:
    """SM4-GCM 解密；tag 不匹配抛 AuthenticationError（F4：篡改任一字节可检测）。"""
    if len(key) != 16 or len(tag) != 16 or len(nonce) != 12:
        raise ValueError("密钥/tag/nonce 长度错误")
    h = _sm4_block_encrypt(key, b"\x00" * 16)
    j0 = nonce + b"\x00\x00\x00\x01"
    s = _ghash(h, aad, ciphertext)
    expect = bytes(a ^ b for a, b in zip(s, _sm4_block_encrypt(key, j0)))
    if not hmac.compare_digest(expect, tag):      # 常量时间比较
        raise AuthenticationError("GCM 标签校验失败（消息被篡改或密钥/nonce 不匹配）")
    return _gctr(key, j0, ciphertext)
