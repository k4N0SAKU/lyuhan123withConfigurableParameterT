"""离线身份供给工具（CA 根密钥 + 节点证书 + CKKS 密钥分发；docs/01 §8.7）。

CA 根密钥**离线生成**（本工具在部署前运行一次；根私钥不进入任何在线节点）：
    python -m src.nodes.provision --dir data/p4_demo [--slots 512]

产物布局：
    <dir>/ca_root/ca_priv.hex      CA 根私钥（离线保管，不入节点）
    <dir>/ca_root/ca_pub.hex       CA 根公钥（三节点信任锚）
    <dir>/certs/<node>.json        节点证书 + 静态密钥对（本节点私文件）
    <dir>/ckks/public.seal|galois.seal|relin.seal   CKKS 公钥材料（P1/P2 可得）
    <dir>/ckks_secret/secret.seal  CKKS 私钥（仅 P0；D5 修订口径）

纪律：随机数 os.urandom（S3）；私钥文件 hex 明文仅因演示级部署（S4 声明：
商用需 GM X.509 证书链 + 密钥保护硬件/软件）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.crypto.ckks_ops import CKKSContext, CKKSParams, PARAMS_MODE_B
from src.crypto.gm_cipher import sm2_generate_keypair
from src.protocol.auth import NodeIdentity, ca_issue_cert
from src.protocol.messages import CertExchangePayload

ROLE_CLIENT, ROLE_KEYNODE, ROLE_INFER = 0, 1, 2


def provision_demo(dir_path: str, validity_ms: int = 365 * 24 * 3600 * 1000,
                   ckks_params: CKKSParams = PARAMS_MODE_B) -> dict:
    """离线生成 CA 根、三节点身份证书与 CKKS 密钥材料，返回清单。"""
    root = Path(dir_path)
    ca_dir = root / "ca_root"
    cert_dir = root / "certs"
    ca_dir.mkdir(parents=True, exist_ok=True)
    cert_dir.mkdir(parents=True, exist_ok=True)

    ca_pub, ca_priv = sm2_generate_keypair()   # 库返回顺序：(pub, priv)
    (ca_dir / "ca_priv.hex").write_text(ca_priv.hex(), encoding="utf-8")
    (ca_dir / "ca_pub.hex").write_text(ca_pub.hex(), encoding="utf-8")

    certs = {}
    for node_id, role in (("p0_client", 0), ("p1_keynode", 1),
                          ("p2_infernode", 2)):
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

    # CKKS：一次密钥生成，公钥材料给 P1/P2，私钥仅 P0（D5 修订）
    ctx = CKKSContext(ckks_params)
    keys_pub = str(root / "ckks")
    keys_sec = str(root / "ckks_secret")
    ctx.save_keys(keys_pub, with_secret=False)
    ctx.save_keys(keys_sec, with_secret=True)

    manifest = {
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


def load_identity(dir_path: str, node_id: str) -> NodeIdentity:
    """从供给目录加载节点身份（本节点静态私钥 + CA 签发的证书）。"""
    doc = json.loads((Path(dir_path) / "certs" / f"{node_id}.json")
                     .read_text(encoding="utf-8"))
    c = doc["cert"]
    cert = CertExchangePayload(
        subject_id=c["subject_id"], subject_pub=bytes.fromhex(c["subject_pub"]),
        not_before_ms=c["not_before_ms"], not_after_ms=c["not_after_ms"],
        ca_sig=bytes.fromhex(c["ca_sig"]))
    return NodeIdentity(node_id=doc["node_id"], role=doc["role"],
                        static_pub=bytes.fromhex(doc["static_pub"]),
                        static_priv=bytes.fromhex(doc["static_priv"]),
                        cert=cert)


def load_ca_pub(dir_path: str) -> bytes:
    return bytes.fromhex((Path(dir_path) / "ca_root" / "ca_pub.hex")
                         .read_text(encoding="utf-8").strip())


def main(argv: list | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="供给产物目录")
    parser.add_argument("--toy", action="store_true",
                        help="玩具 CKKS 参数（测试/演示提速：2^12，链 [30,30,40]）")
    args = parser.parse_args(argv)
    params = PARAMS_MODE_B
    if args.toy:
        params = CKKSParams(name="p4-toy", poly_modulus_degree=1 << 12,
                            coeff_mod_bit_sizes=(30, 30, 40), scale_log2=30,
                            slots=1 << 11,
                            note="P4 协议演示/测试玩具参数（非推理配置）")
    manifest = provision_demo(args.dir, ckks_params=params)
    print(f"provisioned -> {args.dir}")
    print(f"  nodes: {list(manifest['nodes'])}")
    print(f"  ckks: {manifest['ckks_params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
