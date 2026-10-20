"""Reproducible run metadata; collection failures remain explicit."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError, distributions
from pathlib import Path
import platform
import subprocess
import sys
import os
import shutil
import psutil
from .annotations import file_sha256


@lru_cache(maxsize=1)
def environment():
    packages = {}
    for name in ("numpy", "opencv-python", "onnx", "onnxruntime", "torch", "ultralytics",
                 "tensorrt", "tensorrt-cu12", "cuda-python", "scikit-learn", "poseguard"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = dict(os=platform.platform(), machine=platform.machine(), python=sys.version,
                  executable=sys.executable, packages=packages, cpu=platform.processor(),
                  logical_cpus=os.cpu_count(), ram_bytes=psutil.virtual_memory().total,
                  installed_packages={d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]})
    cuda_root = Path(os.environ.get("CUDA_PATH", str(Path.home() / ".local/poseguard-sdk/cuda-12.8")))
    commands = {"cuda_toolkit": [shutil.which("nvcc") or str(cuda_root / "bin/nvcc.exe"), "--version"],
                "cmake": ["cmake", "--version"], "cxx_compiler": ["cl"] if os.name == "nt" else ["c++", "--version"]}
    result["toolchain"] = {}
    for name, command in commands.items():
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=4)
            result["toolchain"][name] = (run.stdout + run.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["toolchain"][name] = dict(unavailable=str(exc))
    try:
        run = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,uuid",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        result["gpu"] = run.stdout.strip() if run.returncode == 0 else None
        result["gpu_query_error"] = run.stderr.strip() if run.returncode else None
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.update(gpu=None, gpu_query_error=str(exc))
    return result


def artifact(path):
    if not path:
        return None
    source = Path(path)
    return dict(path=str(source.resolve()), exists=source.is_file(),
                sha256=file_sha256(source) if source.is_file() else None)


def run_provenance(config_path, cfg, demo=False):
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), environment=environment(),
                config=artifact(config_path), model=artifact(cfg.model) if not demo else None,
                classifier=artifact(cfg.classifier), library=artifact(cfg.library),
                source=artifact(cfg.source) if Path(str(cfg.source)).is_file() else None,
                code={str(p.name): file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
