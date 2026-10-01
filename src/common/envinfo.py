"""环境信息采集（对应需求 F9：性能数据必须随环境信息输出）。"""
from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Optional

import psutil

# 性能报告必须携带的包版本（D6 技术栈）
_TRACKED_PACKAGES = [
    "torch", "transformers", "tokenizers", "tenseal", "gmssl",
    "numpy", "psutil", "fastapi", "uvicorn", "pytest",
]


def _cpu_model_name() -> str:
    """Windows 下从注册表读取友好的 CPU 型号名，失败则退回 platform.processor()。"""
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return str(name).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _package_versions() -> dict:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def _gpu_info(probe: bool) -> Optional[dict]:
    """probe=False 时不导入 torch（保持单元测试轻量）；基准脚本应传 True。"""
    if not probe:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"available": True, "name": props.name,
                    "vram_gb": round(props.total_memory / 2**30, 2)}
        return {"available": False, "note": "torch 报告无 CUDA 设备，CPU 推理"}
    except Exception as exc:  # torch 未安装 / 初始化失败，如实记录
        return {"available": False, "probe_error": repr(exc)}


def collect_environment(network_topology: str = "loopback-same-machine",
                        probe_gpu: bool = False) -> dict:
    """采集并返回运行环境快照。默认口径为同机回环，多机部署时须改写 topology。

    P1-R1 补记：新增 ``cpu_load_percent_at_collect``（采集时 CPU 占用率）与
    ``torch_threads``（torch 已加载时的线程数）——支撑 P6 基线稳定性协议
    （docs/phases/阶段任务登记.md）；torch 未加载时不导入、记 None。
    """
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": {
            "model": _cpu_model_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "ram_total_gb": round(vm.total / 2**30, 2),
        "gpu": _gpu_info(probe_gpu),
        "packages": _package_versions(),
        "network_topology": network_topology,
        "cpu_load_percent_at_collect": psutil.cpu_percent(interval=0.1),
        "torch_threads": (sys.modules["torch"].get_num_threads()
                          if "torch" in sys.modules else None),
    }


def main() -> int:  # pragma: no cover - 自检脚本入口
    """环境自检：打印各库版本并对关键密码/推理依赖做冒烟验证。

    用法：``python -m src.common.envinfo``
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("== A1-22 环境自检 ==")
    env = collect_environment(probe_gpu=True)
    print(f"platform : {env['platform']}")
    print(f"python   : {env['python_version']}")
    print(f"cpu      : {env['cpu']['model']} ({env['cpu']['physical_cores']}C/"
          f"{env['cpu']['logical_cores']}t)")
    print(f"ram      : {env['ram_total_gb']} GB")
    print(f"gpu      : {env['gpu']}")
    print("-- packages --")
    for pkg, ver in env["packages"].items():
        print(f"  {pkg:12s}: {ver or '未安装'}")

    failures = []
    print("-- smoke checks --")
    try:
        import tenseal as ts
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
        ctx.global_scale = 2 ** 40
        v = ts.ckks_vector(ctx, [1.5, 2.5])
        got = (v * v + v).decrypt()[0]
        ok = abs(got - 3.75) < 1e-3
        print(f"  tenseal CKKS x^2+x   : {'OK' if ok else 'FAIL'} (got {got:.4f}, want 3.75)")
        if not ok:
            failures.append("tenseal")
    except Exception as exc:
        print(f"  tenseal CKKS         : FAIL ({exc!r})")
        failures.append("tenseal")
    try:
        from gmssl import sm3
        digest = sm3.sm3_hash(list(b"abc"))
        ok = digest == "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"
        print(f"  gmssl SM3('abc')     : {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append("gmssl")
    except Exception as exc:
        print(f"  gmssl SM3            : FAIL ({exc!r})")
        failures.append("gmssl")
    try:
        import torch
        print(f"  torch threads        : {torch.get_num_threads()} "
              f"(cuda={torch.cuda.is_available()})")
    except Exception as exc:
        print(f"  torch                : FAIL ({exc!r})")
        failures.append("torch")

    missing = [p for p, v in env["packages"].items() if v is None]
    if missing:
        failures.extend(missing)
        print(f"-- 缺失依赖: {', '.join(missing)}")
    if failures:
        print(f"SELF-CHECK FAILED: {', '.join(sorted(set(failures)))}")
        return 1
    print("SELF-CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
