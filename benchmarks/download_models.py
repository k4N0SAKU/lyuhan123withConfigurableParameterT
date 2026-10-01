"""模型下载脚本：将 GPT-2 与 BERT-base-chinese 下载到 data/models/（可离线复跑）。

网络说明：默认 huggingface.co；若不可达自动切换 hf-mirror.com（国内镜像）。
下载结果不进 git（.gitignore），任何人可经本脚本一键复现。

用法（仓库根目录）：
    python -m benchmarks.download_models                 # 下载全部
    python -m benchmarks.download_models --models gpt2   # 只下载 GPT-2
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "gpt2": ("gpt2", ["config.json", "*.safetensors", "vocab.json", "merges.txt",
                      "tokenizer.json", "tokenizer_config.json"]),
    "bert-base-chinese": ("bert-base-chinese", ["config.json", "*.safetensors",
                                                "vocab.txt", "tokenizer.json",
                                                "tokenizer_config.json"]),
}
ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]


def endpoint_reachable(endpoint: str, timeout_s: float = 8.0) -> bool:
    # 探测站点根路径即可；探测 API 子路径会因重定向/鉴权差异产生误报
    try:
        urllib.request.urlopen(endpoint, timeout=timeout_s)
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(MODELS),
                        choices=list(MODELS), help="要下载的模型，默认全部")
    parser.add_argument("--endpoint", default=None,
                        help="强制指定 HF 端点；默认自动探测可达性")
    args = parser.parse_args()

    if args.endpoint:
        endpoints = [args.endpoint]
    else:
        endpoints = [ep for ep in ENDPOINTS if endpoint_reachable(ep)]
        if not endpoints:
            print("ERROR: no reachable HF endpoint (tried %s)" % ", ".join(ENDPOINTS))
            return 1

    # 端点必须在导入 huggingface_hub 之前确定（该库在导入时读取环境变量）
    os.environ["HF_ENDPOINT"] = endpoints[0]
    # hf-mirror 不支持 xet 下载后端，禁用以强制走 HTTP 回退
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    print(f"using endpoint: {endpoints[0]}")
    from huggingface_hub import snapshot_download

    ok = True
    for name in args.models:
        repo_id, patterns = MODELS[name]
        target = REPO_ROOT / "data" / "models" / name
        print(f"downloading {repo_id} -> {target}")
        try:
            snapshot_download(repo_id=repo_id, local_dir=str(target),
                              allow_patterns=patterns, max_workers=4)
            print(f"OK: {name}")
        except Exception as exc:
            ok = False
            print(f"FAILED: {name}: {exc!r}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
