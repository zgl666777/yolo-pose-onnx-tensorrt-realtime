"""Collect honest USB/RTSP fault-test evidence for the current machine.

The probe deliberately separates software recovery evidence from physical device
evidence.  A missing camera or RTSP URL is recorded as ``unavailable``/``skipped``
and can never be reported as a passed physical disconnect test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import platform
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import cv2

from poseguard.provenance import environment


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_url(value: str) -> str:
    """Remove RTSP credentials while retaining a useful endpoint."""
    parsed = urlsplit(value)
    if not parsed.scheme:
        return value
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def probe_tcp(url: str, timeout: float = 2.0) -> dict:
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
        return {"status": "invalid", "error": "URL must use rtsp:// or rtsps://"}
    if not parsed.hostname:
        return {"status": "invalid", "error": "RTSP URL has no hostname"}
    try:
        port = parsed.port or (322 if parsed.scheme.lower() == "rtsps" else 554)
    except ValueError as exc:
        return {"status": "invalid", "error": str(exc)}
    started = time.perf_counter()
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return {"status": "tcp_reachable", "host": parsed.hostname, "port": port,
                    "latency_ms": (time.perf_counter() - started) * 1000}
    except (OSError, ValueError) as exc:
        return {"status": "unreachable", "host": parsed.hostname, "port": port,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.perf_counter() - started) * 1000}


def _probe_capture(source, timeout_ms: int, frames: int) -> dict:
    cap = cv2.VideoCapture()
    started = time.perf_counter()
    result = {"source": redact_url(source) if isinstance(source, str) else source,
              "started_utc": utc_now(), "frames_requested": frames}
    try:
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms]
        if isinstance(source, str):
            opened = cap.open(source, cv2.CAP_FFMPEG, params)
        else:
            opened = cap.open(source)
        result["opened"] = bool(opened)
        result["backend"] = cap.getBackendName() if opened else None
        if not opened:
            result.update(status="unavailable", error="VideoCapture.open returned false")
            return result
        shape = None
        read = 0
        for _ in range(max(1, frames)):
            ok, image = cap.read()
            if not ok or image is None:
                result.update(status="read_failed", read_frames=read,
                              error="VideoCapture.read returned false")
                return result
            read += 1
            shape = list(image.shape)
        result.update(status="passed", read_frames=read, shape=shape,
                      fps=cap.get(cv2.CAP_PROP_FPS))
        return result
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return result
    finally:
        cap.release()
        result["elapsed_seconds"] = time.perf_counter() - started
        result["finished_utc"] = utc_now()


def _probe_process(connection, source, timeout_ms: int, frames: int):
    try:
        connection.send(_probe_capture(source, timeout_ms, frames))
    finally:
        connection.close()


def probe_capture(source, timeout_ms: int, frames: int) -> dict:
    """Isolate native driver calls so unsupported timeouts cannot hang evidence collection."""
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_probe_process, args=(send, source, timeout_ms, frames), daemon=True)
    started = time.perf_counter()
    deadline_seconds = 10 + (timeout_ms / 1000) * (frames + 1)
    child.start()
    send.close()
    result = None
    try:
        if receive.poll(deadline_seconds):
            try:
                result = receive.recv()
            except EOFError:
                pass
        if result is None:
            result = {"source": redact_url(source) if isinstance(source, str) else source,
                      "status": "probe_timeout" if child.is_alive() else "worker_failed",
                      "opened": False, "error": "No probe result before deadline",
                      "deadline_seconds": deadline_seconds}
    finally:
        receive.close()
        child.join(timeout=.2)
        if child.is_alive():
            child.terminate()
            child.join(timeout=1)
        if child.is_alive():
            child.kill()
            child.join(timeout=1)
    result["probe_elapsed_seconds"] = time.perf_counter() - started
    result["probe_worker_alive"] = child.is_alive()
    if child.is_alive():
        result["status"] = "worker_cleanup_failed"
    return result


def run_watchdog_regression(output: Path, timeout: float) -> dict:
    """Run the real child-process blocking tests as software-only evidence."""
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_capture_process.py"]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, cwd=Path(__file__).resolve().parents[1])
        log = output / "capture-watchdog-pytest.log"
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        return {"status": "passed" if completed.returncode == 0 else "failed",
                "exit_code": completed.returncode, "command": command,
                "elapsed_seconds": time.perf_counter() - started, "log": str(log.resolve()),
                "scope": "software child-process watchdog; no physical hotplug"}
    except subprocess.TimeoutExpired as exc:
        return {"status": "failed", "error": f"timeout after {timeout}s", "command": command,
                "elapsed_seconds": time.perf_counter() - started,
                "stdout": (exc.stdout or "")[-2000:], "stderr": (exc.stderr or "")[-2000:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new evidence directory")
    parser.add_argument("--camera-index", type=int, default=None,
                        help="OpenCV camera index; omit to enumerate indices 0..3")
    parser.add_argument("--rtsp", help="RTSP URL to probe (credentials are redacted in output)")
    parser.add_argument("--probe-frames", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--watchdog-timeout", type=float, default=30)
    args = parser.parse_args()
    if args.probe_frames < 1 or args.timeout_ms < 1 or args.watchdog_timeout <= 0:
        parser.error("probe frames, timeout and watchdog timeout must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be new and empty")
    output.mkdir(parents=True, exist_ok=True)

    report = {"created_utc": utc_now(), "host": environment(),
              "platform": platform.platform(), "scope": {
                  "physical_usb_disconnect": "not executable without an external USB camera and operator unplug/replug",
                  "real_rtsp_disconnect": "requires --rtsp and a reachable upstream; loopback software tests are separate",
              }, "camera": [], "rtsp": None, "software_watchdog": None}
    indices = [args.camera_index] if args.camera_index is not None else list(range(4))
    for index in indices:
        result = probe_capture(index, args.timeout_ms, args.probe_frames)
        result["index"] = index
        # OpenCV cannot tell whether an index is an integrated, USB, virtual or
        # VMware-provided device.  Keep the classification unknown until an
        # operator records the hardware identity and performs hotplug actions.
        result["device_kind"] = "unknown_until_operator_identifies"
        result["hotplug_fault"] = "not_run"
        report["camera"].append(result)
    if args.rtsp:
        safe_url = redact_url(args.rtsp)
        tcp = probe_tcp(args.rtsp)
        capture = (probe_capture(args.rtsp, args.timeout_ms, args.probe_frames)
                   if tcp["status"] == "tcp_reachable" else
                   {"source": safe_url, "status": "skipped", "reason": "RTSP TCP endpoint is unavailable or invalid"})
        report["rtsp"] = {"url": safe_url, "tcp": tcp,
                          "capture": capture,
                          "disconnect_fault": "not_run (operator/network action required)"}
    report["software_watchdog"] = run_watchdog_regression(output, args.watchdog_timeout)
    usable_camera = any(row.get("status") == "passed" for row in report["camera"])
    rtsp_available = bool(report["rtsp"] and report["rtsp"]["capture"].get("status") == "passed")
    report["status"] = "evidence_collected" if usable_camera or rtsp_available else "partial"
    report["physical_faults_executed"] = False
    report["interpretation"] = (
        "Software watchdog evidence collected. Physical USB unplug/replug and RTSP upstream/network fault timestamps remain required."
    )
    path = output / "device-fault-report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(path.resolve()),
                      "software_watchdog": report["software_watchdog"]["status"],
                      "physical_faults_executed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
