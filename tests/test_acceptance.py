import importlib.util
import json
from pathlib import Path
import sys


def load_acceptance():
    source = Path(__file__).resolve().parents[1] / "tools" / "acceptance.py"
    spec = importlib.util.spec_from_file_location("acceptance_test_module", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_preflight_runs_without_video_or_tensorrt(tmp_path, monkeypatch):
    module = load_acceptance()
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module, "virtualization_preflight", lambda: {"guest": True, "signals": {"vendor": "VMware"}})
    monkeypatch.setattr(module, "environment", lambda: {"os": "Linux VMware test"})
    output = tmp_path / "preflight"
    monkeypatch.setattr(sys, "argv", ["acceptance.py", "--output", str(output),
                                    "--profile", "linux-cpu", "--preflight-only"])
    assert module.main() == 0
    report = json.loads((output / "acceptance.json").read_text())
    assert report["target_linux"] is True
    assert report["profile"] == "linux-cpu"
    assert report["virtualization"]["guest"] is True
    assert report["capabilities"]["gpu_acceptance"] is False
    assert report["final_acceptance_complete"] is False
    assert report["status"] == "preflight_only"
    assert report["engine_build_modes"]["fp16"] == "fp16-strict"


def test_probe_decodes_windows_utf16_without_corrupting_evidence(monkeypatch):
    module = load_acceptance()
    monkeypatch.setattr(module.shutil, "which", lambda name: "C:/Windows/System32/wsl.exe")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: type("Probe", (), {
        "returncode": 50, "stdout": "WSL missing".encode("utf-16-le"), "stderr": b""})())
    report = module._probe_command(["wsl.exe", "--status"])
    assert report["available"] is True
    assert report["returncode"] == 50
    assert report["output"] == "WSL missing"
