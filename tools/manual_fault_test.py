"""Capture operator-attested external USB or remote RTSP disconnect/recovery evidence.

Run this in a terminal beside the physical test setup.  The tool starts the real
pipeline, records UTC action timestamps and verifies resumed frame delivery.  It
never labels an integrated camera release or a local RTSP process restart as a
physical USB unplug or a remote network disconnect.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from poseguard.config import Config
from poseguard.provenance import artifact, environment


def summarize_recovery(summary: dict, health: list, frames: list, actions: list,
                       client_offset: float = 0) -> dict:
    generations = sorted({int(row["generation"]) for row in frames})
    disconnected = next((row for row in actions if row["action"] == "disconnect_completed"), None)
    restored = next((row for row in actions if row["action"] == "reconnect_completed"), None)
    failures = [row for row in health if row.get("event") in
                {"source_read_failed", "capture_watchdog_timeout"} and
                (disconnected is None or client_offset + float(row.get("elapsed_seconds", 0)) >=
                 float(disconnected["elapsed_seconds"]))]
    # Verify opens after restoration using the client and operator clocks' shared
    # launch offset.  Frame generations link processing back to the successful open.
    restored_generations = {int(row["generation"]) for row in health if row.get("event") == "source_opened" and
                            (restored is None or client_offset + float(row.get("elapsed_seconds", 0)) >=
                             float(restored["elapsed_seconds"]))}
    action_names = {row["action"] for row in actions}
    checks = {"disconnect_attested": "disconnect_completed" in action_names,
              "reconnect_attested": "reconnect_completed" in action_names,
              "read_failure_observed": bool(failures),
              "resumed_frame_delivery": len(generations) >= 2 and bool(restored_generations & set(generations)),
              "run_completed": summary.get("status") == "completed",
              "capture_stopped": not summary.get("capture_thread_alive") and not summary.get("capture_worker_alive"),
              "no_capture_error": not summary.get("capture_error")}
    return {"status": "passed_operator_attested" if all(checks.values()) else "failed",
            "checks": checks, "delivered_generations": generations,
            "fault_health_events": failures,
            "physical_action_evidence": "operator attestation; the software measures resulting failure and recovery"}


def wait_for_initial_frames(directory: Path, process, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Pipeline stopped before initial frames (exit {process.returncode})")
        csv_path = directory / "frames.csv"
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as stream:
                if next(csv.DictReader(stream), None) is not None:
                    return
        time.sleep(.2)
    raise TimeoutError("No initial frames before readiness deadline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--device-kind", choices=["external-usb", "remote-rtsp"], required=True)
    parser.add_argument("--device-id", required=True, help="physical serial/model or camera asset ID; avoid passwords")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--duration", type=float, default=180)
    parser.add_argument("--disconnect-seconds", type=float, default=5)
    parser.add_argument("--readiness-timeout", type=float, default=45)
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("Physical action attestation requires an interactive terminal")
    if args.duration < 60 or args.disconnect_seconds < 1 or args.readiness_timeout <= 0:
        parser.error("duration must be at least 60s; disconnect/readiness timeouts must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be new and empty")
    cfg = Config.load(args.config)
    if args.device_kind == "external-usb" and not str(cfg.source).isdigit():
        parser.error("external-usb requires a camera-index source")
    if args.device_kind == "remote-rtsp" and not str(cfg.source).lower().startswith(("rtsp://", "rtsps://")):
        parser.error("remote-rtsp requires an RTSP source")
    output.mkdir(parents=True, exist_ok=True)
    cfg.show = False
    cfg.duration = args.duration
    # Preserve enough retries for a human action without making recovery infinite.
    cfg.max_reconnects = max(cfg.max_reconnects, 100)
    cfg_path = output / "client.yaml"
    cfg.save(cfg_path)
    actions = []
    started = time.perf_counter()
    report = {"status": "running", "environment": environment(), "device_kind": args.device_kind,
              "device_id": args.device_id, "operator": args.operator, "source_config": artifact(args.config),
              "credential_note": "client.yaml preserves connection credentials; keep this evidence directory private",
              "action_scope": ("operator physically unplugged/replugged an external USB camera" if args.device_kind == "external-usb"
                               else "operator disconnected/reconnected the remote RTSP camera or its network link"),
              "actions": actions}
    process = None
    log = None
    client_offset = 0

    def mark(action):
        row = {"action": action, "utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.perf_counter() - started, "operator": args.operator,
               "device_id": args.device_id}
        actions.append(row)
        with (output / "faults.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    try:
        log = (output / "client.log").open("w", encoding="utf-8")
        client_offset = time.perf_counter() - started
        process = subprocess.Popen([sys.executable, "-m", "poseguard", "run", "--config", str(cfg_path),
                                    "--output", str(output / "client")], stdout=log, stderr=subprocess.STDOUT)
        wait_for_initial_frames(output / "client", process, args.readiness_timeout)
        mark("initial_frames_ready")
        prompt = ("Physically unplug the external USB camera, then press Enter. " if args.device_kind == "external-usb"
                  else "Disconnect the remote RTSP camera/network link, then press Enter. ")
        input(prompt)
        mark("disconnect_completed")
        time.sleep(args.disconnect_seconds)
        input("Restore the same device/network link, then press Enter. ")
        mark("reconnect_completed")
        print("Collecting resumed frames until the configured test duration completes.", flush=True)
        process.wait(timeout=args.duration + 30)
        summary = json.loads((output / "client/summary.json").read_text(encoding="utf-8"))
        health = [json.loads(line) for line in (output / "client/health.jsonl").read_text(encoding="utf-8").splitlines()]
        with (output / "client/frames.csv").open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        report.update(summarize_recovery(summary, health, frames, actions, client_offset))
        report.update(summary=summary, client_exit_code=process.returncode)
        if process.returncode:
            report["status"] = "failed"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if log is not None:
            log.close()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["logs"] = {path.name: artifact(path) for path in output.rglob("*.log")}
        (output / "fault-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str((output / "fault-report.json").resolve())}))
    return 0 if report["status"] == "passed_operator_attested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
