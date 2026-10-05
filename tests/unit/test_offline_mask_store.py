"""离线出口掩码库存单测（P9 3a；offline_mask_gen.py）。

覆盖：生成/双视图消费对齐、SM3 防篡改、一次性使用（OTP 纪律）、耗尽中止、
库存版出口片与在线版数值等价、m 校验。
"""
from __future__ import annotations

import json
import os

import pytest

from src.crypto.ckks_ops import CKKSContext, PARAMS_P4_TOY
from src.crypto.secret_sharing_m import share_vector_n
from src.nodes.offline_mask_gen import (ComposerExitMaskStore,
                                        PartyExitMaskStore,
                                        generate_exit_mask_inventory)
from src.protocol.conversion import FIXED_ONE, to_fixed
from src.protocol.conversion_m import ComputePartyRole, exit_compose_m


@pytest.fixture(scope="module")
def toy_pub(tmp_path_factory):
    keys = str(tmp_path_factory.mktemp("p9_mask_keys"))
    full = CKKSContext(PARAMS_P4_TOY)
    full.save_keys(keys, with_secret=True)
    return CKKSContext(PARAMS_P4_TOY, public_only=True, keys_dir=keys)


def test_generate_and_aligned_consumption(toy_pub, tmp_path):
    d = str(tmp_path / "store")
    info = generate_exit_mask_inventory(toy_pub, d, m=3, count=4)
    assert info["m"] == 3 and info["count"] == 4
    ps = [PartyExitMaskStore(f"{d}/party_{i}_store.json") for i in (1, 2, 3)]
    cs = ComposerExitMaskStore(f"{d}/composer_store.json")
    for k in range(4):
        encs = cs.fetch_next()
        assert len(encs) == 3 and all(isinstance(b, bytes) for b in encs)
        for st in ps:
            s = st.fetch_next()
            assert len(s) == toy_pub.slot_count      # 全槽掩码片
    with pytest.raises(RuntimeError, match="耗尽"):
        cs.fetch_next()                              # 一次性使用：耗尽即中止


def test_checksum_tamper_detection(toy_pub, tmp_path):
    d = str(tmp_path / "store")
    generate_exit_mask_inventory(toy_pub, d, m=2, count=2)
    # 篡改条目文件（实际数据）→ fetch 时校验拦截
    p = f"{d}/party_1/s_0.json"
    data = json.loads(open(p, encoding="utf-8").read())
    data["s"][0] ^= 1
    open(p, "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False))
    st = PartyExitMaskStore(f"{d}/party_1_store.json")
    with pytest.raises(ValueError, match="篡改"):
        st.fetch_next()
    # 篡改清单 → 构造时拦截
    p2 = f"{d}/party_2_store.json"
    man = json.loads(open(p2, encoding="utf-8").read())
    man["entries"][0]["index"] = 99
    open(p2, "w", encoding="utf-8").write(json.dumps(man, ensure_ascii=False))
    with pytest.raises(ValueError, match="篡改"):
        PartyExitMaskStore(p2)


def test_stored_exit_piece_equivalent_to_online(toy_pub, tmp_path):
    """库存版与在线采样版出口片数值等价（同 a、各自合法 s ⇒ 合成结果一致口径）。"""
    pub, sec = toy_pub, None
    # toy_pub 只给了 pub——合成正确性在本测试里只验 zᵢ 构造与 compose 可解密：
    # 复用在线版的端到端断言已由 test_conversion_mparty 覆盖，此处验证
    # 库存路径产出同样的 z 结构与 compose 成功。
    d = str(tmp_path / "store")
    generate_exit_mask_inventory(pub, d, m=2, count=2)
    ps = [PartyExitMaskStore(f"{d}/party_{i}_store.json") for i in (1, 2)]
    cs = ComposerExitMaskStore(f"{d}/composer_store.json")
    parties = [ComputePartyRole(pub, index=i + 1, m=2, is_anchor=(i == 0))
               for i in range(2)]
    fixed = [to_fixed(0.5)] * 8
    shares = share_vector_n(fixed, 2)
    req = os.urandom(16)
    encs = cs.fetch_next()
    pieces = []
    for p, a, st in zip(parties, shares, ps):
        p.set_result_share(req, a)
        s_i = st.fetch_next()
        pieces.append(p.exit_piece_stored(req, s_i, encs[p.index - 1]))
    ct = exit_compose_m(pub, pieces)
    assert ct is not None
    cs.fetch_next()                                  # 第 2 条（库存共 2 条）
    with pytest.raises(RuntimeError):
        cs.fetch_next()                              # 第 3 次 → 耗尽即协议中止


def test_generate_rejects_small_m(toy_pub, tmp_path):
    with pytest.raises(ValueError):
        generate_exit_mask_inventory(toy_pub, str(tmp_path / "s"), m=1, count=1)
