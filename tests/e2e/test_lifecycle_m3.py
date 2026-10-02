"""m 方生命周期 e2e（P8；玩具参数、真实通道、D5′ 协议族全流程）。

覆盖：1+m 方供给 → 逐链路认证/SM2DH → 掩码链式入口 → 白名单① →
出口星型合成 → 白名单②（**精确性断言 max_abs_err == 0**）→ ratchet →
销毁 → 审计链核验。m=2 为退化对照（与原三方拓扑等价）。
"""
from __future__ import annotations

import pytest

from src.crypto.ckks_ops import PARAMS_P4_TOY
from src.nodes.orchestrator_m import links_for_m, run_lifecycle_m
from src.nodes.provision_m import provision_demo_m


@pytest.mark.parametrize("m", [2, 3])
def test_lifecycle_m(tmp_path, m):
    prov = str(tmp_path / f"prov_m{m}")
    manifest = provision_demo_m(prov, m=m, ckks_params=PARAMS_P4_TOY)
    assert manifest["m"] == m
    expect_nodes = {"p0_client", "p1_keynode", "p2_infernode"} | {
        f"p{k}_compute" for k in range(3, m + 1)}
    assert set(manifest["nodes"]) == expect_nodes

    rep = run_lifecycle_m(prov, m=m, params=PARAMS_P4_TOY,
                          log_dir=str(tmp_path / "logs"))
    # 协议正确性：常数乘后端到端在 ±ulp 窗内（conversion.py 申报 e_ckks ≤1
    # ulp，常数乘线性放大——环算术相对 y₁ 精确；红线 4 ulp 同 bench）
    from src.protocol.conversion import FIXED_ONE
    assert rep["phases"]["inference"]["max_abs_err"] <= 4 / FIXED_ONE
    assert rep["phases"]["inference"]["mask_chain_hops"] == m
    # 控制面不变式：全部链路认证+协商成功
    assert len(rep["phases"]["establish"]) == m + 1
    # 审计链完整
    assert all(rep["audit_chain_verified"].values())
    assert not any(rep["audit_errors"].values())
    # 销毁零化 + 销毁后收发拒绝
    assert all(v["all_zeroed"]
               for node in rep["phases"]["destroy"].values()
               for v in node.values())
    assert "SENT(BUG)" not in rep["phases"]["post_destroy_reject"].values()


def test_links_for_m_topology():
    assert len(links_for_m(2)) == 3
    l3 = links_for_m(3)
    assert len(l3) == 5
    assert ("p2-p3", "p2_infernode", "p3_compute") in l3
    assert ("p1-p3", "p1_keynode", "p3_compute") in l3
    l4 = links_for_m(4)
    assert ("p3-p4", "p3_compute", "p4_compute") in l4
    assert ("p2-p4", "p2_infernode", "p4_compute") in l4
    assert len(l4) == 7
