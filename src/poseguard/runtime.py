"""Discover user-local Windows SDK DLLs without requiring a global PATH change."""
from __future__ import annotations

import os
from pathlib import Path

_dll_handles = []
_configured = False


def configure_gpu_runtime() -> None:
    global _configured
    if _configured or os.name != "nt":
        return
    root = Path(os.environ.get("POSEGUARD_SDK_ROOT", str(Path.home() / ".local" / "poseguard-sdk")))
    cuda = Path(os.environ.get("CUDA_PATH", str(root / "cuda-12.8")))
    tensorrt = Path(os.environ.get("TENSORRT_ROOT", str(root / "TensorRT-10.8.0.43")))
    paths = [cuda / "bin", tensorrt / "lib", tensorrt / "bin"]
    existing = os.environ.get("PATH", "").split(os.pathsep)
    for path in paths:
        if path.is_dir():
            _dll_handles.append(os.add_dll_directory(str(path)))
            if str(path) not in existing:
                existing.insert(0, str(path))
    os.environ["PATH"] = os.pathsep.join(existing)
    _configured = True

