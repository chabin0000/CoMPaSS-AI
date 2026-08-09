"""설치 여부와 관계없이 동작하는 Windows/Linux 시스템 점검."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import ctypes
from pathlib import Path


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5).strip() or None
    except Exception:
        return None


def system_check(path: Path = Path(".")) -> dict:
    disk = shutil.disk_usage(path)
    data = {"os": platform.platform(), "architecture": platform.machine(), "cpu": platform.processor() or None, "python": sys.version.split()[0], "ram_total": _ram_total(), "disk": {"total": disk.total, "free": disk.free}, "git": _command_output(["git", "--version"]), "cuda_available": False, "pytorch": "not installed", "gpu": None, "nvidia_driver": _command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]), "transformers": "installed" if importlib.util.find_spec("transformers") else "not installed", "vllm": "installed" if importlib.util.find_spec("vllm") else "not installed"}
    try:
        import torch  # type: ignore
        data["pytorch"] = torch.__version__
        data["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            data["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        meminfo = Path("/proc/meminfo")
        data["proc_meminfo_available"] = meminfo.exists()
        if meminfo.exists():
            data["meminfo"] = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in meminfo.read_text().splitlines()[:8]}
    else:
        data["proc_meminfo_available"] = False
    return data


def _ram_total() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except Exception:
            return None
    if sys.platform.startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(status)
        return int(status.ullTotalPhys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
    return None


def dgx_readiness(check: dict) -> dict:
    """엔진을 추천하지 않고 DGX 판단에 필요한 환경 정보만 요약한다."""
    return {"linux": "linux" in check["os"].lower(), "architecture": check["architecture"], "cuda_available": check["cuda_available"], "gpu": check["gpu"], "nvidia_driver": check["nvidia_driver"], "ram_total": check["ram_total"], "disk_free": check["disk"]["free"], "python": check["python"], "pytorch": check["pytorch"], "transformers": check["transformers"], "vllm": check["vllm"], "note": "inference engine is intentionally not selected automatically"}


def save_system_check(path: Path) -> dict:
    data = system_check(path)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data
