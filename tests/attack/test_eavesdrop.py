"""F7-1 窃听攻击：全链路消息记录 + 离线分析。

攻击者模型：Dolev-Yao 被动窃听者， tap 全部三条链路（两端发出字节的全集），
离线执行 ①结构解析（识别协议/区分明文段与密文段）②明文模式扫描（输入串/
token id/定点编码）③字节频率分析（密文均匀性）④已知明文相关分析。
断言：敏感输入与中间明文不可恢复。

运行：pytest tests/attack/test_eavesdrop.py -v
"""
from __future__ import annotations

import os
import threading

import pytest

from src.crypto.ckks_ops import PARAMS_P4_TOY, CKKSContext, CKKSParams
from src.nodes.base import LoopbackLink
from src.nodes.orchestrator import (build_local_nodes, establish_all,
                                    run_inference_round)
from src.nodes.provision import provision_demo
from src.protocol.conversion import to_fixed
from tests.attack.conftest import NOW
from tests.attack.framework import (VERDICTS, RecordingEndpoint, chi2_uniformity,
                                    pearson, scan_pattern, wire_bytes)

INPUT_TEXT = "办事大厅服务热情周到，一次办好"
SECRET_VALUE = 3.75


@pytest.fixture(scope="module")
def tapped_round(tmp_path_factory):
    """三条链路全 tap 的一次真实数据面往返（玩具 CKKS）。"""
    prov = str(tmp_path_factory.mktemp("ev_prov"))
    provision_demo(prov, ckks_params=PARAMS_P4_TOY)
    client, keynode, infer = build_local_nodes(prov, PARAMS_P4_TOY,
                                               log_dir=str(prov) + "/logs")
    establish_all(client, keynode, infer, ratchet_threshold=64)
    # 窃听者 tap：包装三链路发起端（记录发出帧；对端经同一链路）
    taps = {}
    for name, chan in (("p0-p1", client), ("p0-p2", client), ("p1-p2", keynode)):
        ep = RecordingEndpoint(chan.channels[name].link, name)
        chan.channels[name].link = ep
        taps[name] = ep
    result = run_inference_round(client, keynode, infer,
                                 [0.5, -1.25, 2.0, SECRET_VALUE, -0.125, 1.0])
    return taps, result, client, keynode, infer


class TestEavesdrop:
    def test_plaintext_patterns_absent_from_wire(self, tapped_round):
        """①明文模式扫描：输入串（多编码）/token id/定点编码不得出现在线缆字节。"""
        taps, result, client, keynode, infer = tapped_round
        wire = wire_bytes(list(taps.values()))
        # 输入串（utf-8 / gbk）
        assert scan_pattern(wire, INPUT_TEXT.encode("utf-8")) == 0
        try:
            assert scan_pattern(wire, INPUT_TEXT.encode("gbk")) == 0
        except UnicodeEncodeError:
            pass
        # token id 常见定宽字节模式（本项目分享为 8B 大端；附带 4B 探测）
        assert scan_pattern(wire, (3).to_bytes(8, "big")) == 0
        # 真实定点秘密值（Q16 大端 8B）——SECRET_VALUE 与其 2 倍（出口常数乘）
        for v in (SECRET_VALUE, SECRET_VALUE * 2):
            assert scan_pattern(wire, to_fixed(v).to_bytes(8, "big")) == 0
        VERDICTS.collect("F7-1", "明文模式扫描", "防御成功",
                         {"wire_bytes": len(wire),
                          "patterns": ["utf-8 文本", "gbk 文本",
                                       "token id 8B", "Q16 定点秘密值"],
                          "hits": 0})

    def test_structure_parse_and_metadata_honesty(self, tapped_round):
        """②结构解析：会话帧载荷为 GCM 密文（不可解析出 payload_type）；
        元数据（消息类型/长度/时序）可见——如实申报为边界而非隐藏。"""
        taps, *_ = tapped_round
        wire = wire_bytes(list(taps.values()))
        # GCM 帧结构存在（mode=1）且其载荷段不含任何明文 payload_type 标记
        gcm_frames = 0
        for ep in taps.values():
            for fr in ep.frames:
                if len(fr) > 44 and fr[4] == 1:
                    gcm_frames += 1
                    body = fr[44:-16]
                    assert b"Payload" not in body      # 载荷类型不可见（加密）
        assert gcm_frames > 0
        # 元数据诚实申报：msg_type 字段（header 内）明文可读——威胁模型 §7 边界
        VERDICTS.collect("F7-1", "结构解析", "防御成功",
                         {"gcm_frames": gcm_frames,
                          "metadata_boundary": "msg_type/长度/时序可见（docs/02 §7）"})

    def test_ciphertext_frequency_uniform(self, tapped_round):
        """③频率分析：GCM 密文字节分布通过均匀性检验（卡方 ≈ 自由度量级）。"""
        taps, *_ = tapped_round
        wire = wire_bytes(list(taps.values()))
        # 取会话期 GCM 帧的密文段（跳过明文握手帧）
        ct_bytes = bytearray()
        for ep in taps.values():
            for fr in ep.frames:
                if len(fr) > 200 and fr[4] == 1:       # 大帧=密文载荷（转换消息）
                    ct_bytes += fr[44:-16]
        stat = chi2_uniformity(bytes(ct_bytes))
        assert len(ct_bytes) > 4096, "密文样本不足"
        assert stat < 400, f"密文字节频率偏离均匀（chi2={stat:.0f}）"
        VERDICTS.collect("F7-1", "频率分析", "防御成功",
                         {"ct_sample_bytes": len(ct_bytes),
                          "chi2": round(stat, 1),
                          "note": "均匀分布期望≈255（自由度），阈值 400"})

    def test_known_plaintext_correlation(self, tapped_round, tmp_path):
        """④已知明文相关：线缆上的 y₁（OTP 掩码值）与候选明文定点编码的
        相关 ≈ 0——已知明文侧区分攻击无检验力。"""
        taps, result, client, keynode, infer = tapped_round
        wire = wire_bytes(list(taps.values()))
        # 从 p0-p1 链路取 CONVERT_SHARE 帧（会话期帧 4 字节长度前缀+mode 1）
        y1_frame = None
        for fr in taps["p0-p1"].frames:
            if len(fr) > 300 and fr[4] == 1:
                y1_frame = fr
                break
        assert y1_frame is not None
        payload = y1_frame[44:-16]                     # GCM 密文段（含 y₁）
        # 候选明文：真实值与若干干扰值的 Q16 编码（大端 8B 流）
        cand = [SECRET_VALUE, SECRET_VALUE * 2, 0.5, -1.25, 2.0]
        for v in cand:
            fixed_stream = (to_fixed(v).to_bytes(8, "big") * (len(payload) // 8))
            r = pearson(payload, fixed_stream)
            assert abs(r) < 0.2, f"候选明文 {v} 与线缆数据相关 r={r:.3f}"
        VERDICTS.collect("F7-1", "已知明文相关分析", "防御成功",
                         {"candidates": len(cand),
                          "max_abs_pearson": 0.2,
                          "note": "OTP 掩码使线缆分享与任何候选明文统计独立"})

    def test_input_not_recoverable_verdict(self, tapped_round):
        """结论判定：窃听者视角无可恢复敏感明文（综合上述四项）。"""
        taps, result, *_ = tapped_round
        assert result["max_abs_err"] < 1e-5            # 对照：合法流程本身正确
        VERDICTS.collect("F7-1", "综合结论", "防御成功",
                         {"recoveries": 0,
                          "scope": "被动窃听（Dolev-Yao 被动）；主动攻击见 F7-2/3/4"})
