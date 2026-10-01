"""阶段交付打包器（每阶段质量门后的固定动作，见 docs/phases/ 工作记录约定）。

产出（默认写到仓库上级目录）：
    p<阶段>.zip          —— 仓库在该阶段最终提交处的完整快照（git archive，
                            自动排除 .venv/、data/models/ 等未入库内容）
    p<阶段>.zip.sha256   —— 压缩包校验和
压缩包内额外注入两份文件：
    MANIFEST.txt         —— 快照提交哈希、生成时间、文件数、复现说明
    <阶段>-工作记录.md    —— 当前工作区的阶段工作记录（docs/phases/）

用法（仓库根目录）：
    python -m benchmarks.package_phase --phase P0 --commit ea28405
    python -m benchmarks.package_phase --phase P3 --commit HEAD
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(args: list) -> str:
    out = subprocess.run(["git"] + args, cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {args} 失败: {out.stderr}")
    return out.stdout.strip()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, help="阶段名，如 P0 / P1")
    parser.add_argument("--commit", default="HEAD", help="阶段最终提交（哈希/分支/HEAD）")
    parser.add_argument("--out-dir", default=str(REPO_ROOT.parent),
                        help="输出目录，默认仓库上级目录")
    args = parser.parse_args()

    commit = _git(["rev-parse", args.commit])
    short = commit[:7]
    log_line = _git(["log", "-1", "--format=%h %ad %s", "--date=short", commit])
    phase = args.phase.upper()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{phase.lower()}.zip"

    # 1) git archive 生成快照基础包
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base.zip"
        proc = subprocess.run(
            ["git", "archive", "--format=zip", "-o", str(base), commit],
            cwd=REPO_ROOT, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="replace"))
        # 2) 注入 MANIFEST 与阶段工作记录
        worklog = REPO_ROOT / "docs" / "phases" / f"{phase}-工作记录.md"
        with zipfile.ZipFile(base) as src, \
                zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as dst:
            names = src.namelist()
            for name in names:
                dst.writestr(name, src.read(name))
            manifest = [
                f"A1-22 阶段交付包: {phase}",
                "仓库: a122-llm-privacy",
                f"快照提交: {commit}",
                f"提交说明: {log_line}",
                f"快照文件数: {len(names)}",
                f"打包时间(UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                "未包含: .venv/、data/models/（模型权重，gitignore，"
                "经 python -m benchmarks.download_models 复现）、.git/、基准结果由 results/ 快照携带",
                f"工作记录: {phase}-工作记录.md（压缩包根目录 + 仓库 docs/phases/）",
                "复现: 见包内 README『快速开始』；测试: python -m benchmarks.run_ci",
                f"打包器: python -m benchmarks.package_phase --phase {phase} --commit {short}",
            ]
            dst.writestr("MANIFEST.txt", "\n".join(manifest) + "\n")
            if worklog.exists():
                dst.writestr(f"{phase}-工作记录.md", worklog.read_bytes())
            else:
                print(f"WARN: 未找到 {worklog}（工作记录未写入压缩包）")

    # 3) 校验 + 侧车哈希
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"zip 损坏: {bad}")
        n_files = len(z.namelist())
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (out_dir / f"{phase.lower()}.zip.sha256").write_text(f"{digest}  {zip_path.name}\n",
                                                         encoding="utf-8")
    size_mb = zip_path.stat().st_size / 2**20
    print(f"OK {zip_path}")
    print(f"   commit={short} files={n_files} size={size_mb:.2f} MiB")
    print(f"   sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
