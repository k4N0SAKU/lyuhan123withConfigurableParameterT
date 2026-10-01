"""P4 全生命周期 e2e（任务书 5：会话建立 → 推理 → 销毁的完整生命周期）。

进程内全流程（玩具 CKKS 参数）：三条链路认证协商 → 真实通道数据面
（INFER_REQUEST → 转换真实路径 → INFER_RESULT）→ ratchet → 销毁 →
审计链核验 + 状态机轨迹断言。多进程编排为 slow 用例（三真进程 spawn）。
"""
from __future__ import annotations

import pytest

from src.crypto.ckks_ops import PARAMS_P4_TOY

orchestrator = pytest.importorskip("src.nodes.orchestrator")
from src.nodes.orchestrator import run_lifecycle, run_multiprocess_smoke  # noqa: E402
from src.nodes.provision import provision_demo  # noqa: E402


@pytest.fixture(scope="module")
def prov(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("p4_prov"))
    provision_demo(d, ckks_params=PARAMS_P4_TOY)
    return d


class TestFullLifecycle:
    def test_inprocess_lifecycle(self, prov, tmp_path):
        report = run_lifecycle(prov, params=PARAMS_P4_TOY,
                               log_dir=str(tmp_path / "logs"),
                               ratchet_threshold=2)
        ph = report["phases"]
        # F3/F4/F5：三链路 ACTIVE、纪元 0、密钥材料在册
        for nid, chans in ph["establish"].items():
            for name, st in chans["channels"].items():
                assert st["state"] == "ACTIVE", f"{nid}/{name}"
        # 数据面：真实通道 + 转换真实路径，2·x 往返数值一致
        assert ph["inference"]["max_abs_err"] < 1e-6, ph["inference"]
        assert ph["inference"]["const_mult"] == 2
        # ratchet：轮换已发生且双端纪元一致（触发帧自身计数 ⇒ 阈值2/4条 ⇒ 2 轮）
        assert ph["ratchet"]["p0_send_epoch"] == ph["ratchet"]["p1_recv_epoch"]
        assert ph["ratchet"]["p0_send_epoch"] >= 1
        # F6：销毁后所有通道拒绝收发
        assert ph["destroy"]["p0_client"]["p0-p2"]["all_zeroed"]
        for k, v in ph["post_destroy_reject"].items():
            assert v.startswith("rejected:"), f"{k}: {v}"
        # F5：三节点审计链核验通过
        assert all(report["audit_chain_verified"].values()), report["audit_errors"]
        # 状态机实测轨迹（出口条件 3）：每链 IDLE→NEGOTIATED→ACTIVE→DESTROYED
        for nid, chans in report["trajectories"].items():
            for link, hist in chans.items():
                names = [h["from"] for h in hist] + [hist[-1]["to"]]
                assert names == ["IDLE", "NEGOTIATED", "ACTIVE", "DESTROYED"], \
                    f"{nid}/{link}: {names}"

    def test_lifecycle_audit_files_written(self, prov, tmp_path):
        log_dir = tmp_path / "logs2"
        run_lifecycle(prov, params=PARAMS_P4_TOY, log_dir=str(log_dir),
                      ratchet_threshold=2)
        for nid in ("p0_client", "p1_keynode", "p2_infernode"):
            f = log_dir / f"{nid}.jsonl"
            assert f.exists() and f.stat().st_size > 0
            text = f.read_text(encoding="utf-8")
            assert "SESSION_ACTIVE" in text and "KEYS_DESTROYED" in text
            assert "DECRYPT_WHITELIST" not in text   # 正常生命周期无拒绝事件


@pytest.mark.slow
class TestMultiprocessOrchestration:
    def test_three_process_smoke(self, prov, tmp_path):
        out = run_multiprocess_smoke(prov, out_path=str(tmp_path / "mp_report.json"))
        assert out["all_processes_ended"]
        assert len(out["reports"]) == 3
        for nid, r in out["reports"].items():
            cmds = [c["cmd"] for c in r["commands"]]
            # 每节点 2 条链路各一次 GCM 往返
            assert cmds.count("ping") == 2, nid
            assert cmds[0] == "establish" and cmds[-1] == "destroy", nid
            assert r["destroy"] and all(r["destroy"].values())
        # GCM 消息真实跨进程往返（ping 回显——对端的 node_id+链路名可验证）
        for r in out["reports"].values():
            for c in [c for c in r["commands"] if c["cmd"] == "ping"]:
                assert "hello-from-" in str(c["recv"]["data"]), c
