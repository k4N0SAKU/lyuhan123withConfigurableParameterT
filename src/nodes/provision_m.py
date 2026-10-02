"""m 方身份供给（P8；provision.py 的 m 参数化，零改动原文件）。

与 provision_demo 的差异仅一处：节点清单从固定三方扩为 1 + m 方——
    p0_client(role 0) + p1_keynode(1) + p2_infernode(2) + p{k}_compute(2)
CKKS 密钥仪式不变：一次生成，公钥材料全计算方可得，私钥仅 P0（D5 延续）。
CA 根私钥离线保管不入节点；纪律 S3/S4 与原版一致。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.crypto.ckks_ops import CKKSContext, CKKSParams, PARAMS_MODE_B
from src.crypto.gm_cipher import sm2_generate_keypair
from src.protocol.auth import ca_issue_cert


def provision_demo_m(dir_path: str, m: int = 3,
                     validity_ms: int = 365 * 24 * 3600 * 1000,
                     ckks_params: CKKSParams = PARAMS_MODE_B) -> dict:
    """离线生成 CA 根、1+m 方节点证书与 CKKS 密钥材料，返回清单。"""
    if m < 2:
        raise ValueError("m 必须 ≥ 2（门限语义）")
    root = Path(dir_path)
    ca_dir = root / "ca_root"
    cert_dir = root / "certs"
    ca_dir.mkdir(parents=True, exist_ok=True)
    cert_dir.mkdir(parents=True, exist_ok=True)

    ca_pub, ca_priv = sm2_generate_keypair()   # 库返回顺序：(pub, priv)
    (ca_dir / "ca_priv.hex").write_text(ca_priv.hex(), encoding="utf-8")
    (ca_dir / "ca_pub.hex").write_text(ca_pub.hex(), encoding="utf-8")

    nodes = [("p0_client", 0), ("p1_keynode", 1), ("p2_infernode", 2)]
    nodes += [(f"p{k}_compute", 2) for k in range(3, m + 1)]
    certs = {}
    for node_id, role in nodes:
        pub, priv = sm2_generate_keypair()
        cert = ca_issue_cert(ca_priv, node_id, pub, validity_ms=validity_ms)
        doc = {
            "node_id": node_id, "role": role,
            "static_pub": pub.hex(), "static_priv": priv.hex(),
            "cert": {
                "subject_id": cert.subject_id,
                "subject_pub": cert.subject_pub.hex(),
                "not_before_ms": cert.not_before_ms,
                "not_after_ms": cert.not_after_ms,
                "ca_sig": cert.ca_sig.hex(),
            },
        }
        (cert_dir / f"{node_id}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        certs[node_id] = doc

    # CKKS：一次密钥生成，公钥材料给全部计算方，私钥仅 P0（D5 延续）
    ctx = CKKSContext(ckks_params)
    keys_pub = str(root / "ckks")
    keys_sec = str(root / "ckks_secret")
    ctx.save_keys(keys_pub, with_secret=False)
    ctx.save_keys(keys_sec, with_secret=True)

    manifest = {
        "m": m,
        "ca_pub": ca_pub.hex(),
        "nodes": {k: {"role": v["role"], "cert_file": f"certs/{k}.json"}
                  for k, v in certs.items()},
        "ckks_public": keys_pub, "ckks_secret": keys_sec,
        "ckks_params": {"name": ckks_params.name,
                        "poly": ckks_params.poly_modulus_degree,
                        "chain": list(ckks_params.coeff_mod_bit_sizes)},
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list | None = None) -> int:
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="供给产物目录")
    parser.add_argument("--m", type=int, default=3,
                        help="计算方数量（t = m−1；mode-b 上限 15）")
    parser.add_argument("--toy", action="store_true",
                        help="玩具 CKKS 参数（测试/演示提速：2^12，链 [30,30,40]）")
    args = parser.parse_args(argv)
    params = PARAMS_MODE_B
    if args.toy:
        from src.crypto.ckks_ops import CKKSParams
        params = CKKSParams(name="p4-toy", poly_modulus_degree=1 << 12,
                            coeff_mod_bit_sizes=(30, 30, 40), scale_log2=30,
                            slots=1 << 11,
                            note="P4 协议演示/测试玩具参数（非推理配置）")
    manifest = provision_demo_m(args.dir, m=args.m, ckks_params=params)
    print(f"provisioned (m={args.m}, t={args.m - 1}) -> {args.dir}")
    print(f"  nodes: {list(manifest['nodes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
