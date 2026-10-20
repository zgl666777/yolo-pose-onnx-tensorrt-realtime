"""Real socket RTSP fault injection against a local server, explicitly not a remote camera test."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen, Request
import zipfile
from poseguard.config import Config
from poseguard.provenance import artifact, environment

URL = "https://github.com/bluenviron/mediamtx/releases/download/v1.21.2/mediamtx_v1.21.2_windows_amd64.zip"
SHA256 = "a84224f5a490855684e62ba6fb71e80ea2a5e16aed61883bb58380290e93e467"
ARCHIVE_BYTES = 28583662


def analyze_recovery(summary, health, faults, client_exit_code, frames=(), client_offset=0):
    """Require frame delivery after each induced restart, not just successful open calls."""
    opened = [row for row in health if row["event"] == "source_opened"]
    generations = sorted({int(row["generation"]) for row in frames})
    first_delivery = {generation: min(float(row["timestamp"]) for row in frames
                                     if int(row["generation"]) == generation)
                      for generation in generations}
    fault_names = {row["event"] for row in faults}
    required_faults = {"publisher_terminated", "publisher_restarted", "server_terminated",
                       "server_and_publisher_restarted"}
    fault_phases = []
    for stopped_name, restored_name in (("publisher_terminated", "publisher_restarted"),
                                       ("server_terminated", "server_and_publisher_restarted")):
        stopped = next((row for row in faults if row["event"] == stopped_name), None)
        restored = next((row for row in faults if row["event"] == restored_name), None)
        opened_before = {int(row["generation"]) for row in opened if stopped and "elapsed_seconds" in stopped and
                         client_offset + float(row.get("elapsed_seconds", 0)) < float(stopped["elapsed_seconds"])}
        opened_after = {int(row["generation"]) for row in opened if restored and "elapsed_seconds" in restored and
                        client_offset + float(row.get("elapsed_seconds", 0)) >= float(restored["elapsed_seconds"])}
        fault_phases.append({"fault": stopped_name, "restore": restored_name,
                             "delivered_before_fault": bool(opened_before & set(generations)),
                             "delivered_after_restore": bool(opened_after & set(generations))})
    recovery_delivered = all(row["delivered_before_fault"] and row["delivered_after_restore"] for row in fault_phases)
    healthy = (client_exit_code == 0 and summary.get("status") == "completed" and
               summary.get("processed", 0) > 0 and summary.get("reconnects", 0) >= 2 and
               len(opened) >= 3 and len(generations) >= 3 and
               required_faults <= fault_names and
               recovery_delivered and
               not summary.get("capture_thread_alive") and not summary.get("capture_worker_alive") and
               not summary.get("capture_error"))
    return dict(client_exit_code=client_exit_code, summary=summary, source_open_events=opened,
                delivered_generations=generations, first_delivery_timestamp_by_generation=first_delivery,
                fault_phase_delivery=fault_phases, client_launch_offset_seconds=client_offset,
                real_faults_executed=bool(faults), physical_camera_faults_executed=False,
                acceptance_checks={"clean_exit": client_exit_code == 0,
                                   "completed_status": summary.get("status") == "completed",
                                   "at_least_two_reconnects": summary.get("reconnects", 0) >= 2,
                                   "at_least_three_opened_generations": len(opened) >= 3,
                                   "frames_after_both_restarts": len(generations) >= 3 and recovery_delivered,
                                   "both_fault_injections_executed": required_faults <= fault_names,
                                   "capture_stopped": not summary.get("capture_thread_alive") and not summary.get("capture_worker_alive")},
                status="passed" if healthy else "failed")


def _stop_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _wait_port(port, process, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"RTSP server exited before listening ({process.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.2):
                return
        except OSError:
            time.sleep(.1)
    raise RuntimeError("RTSP server did not start listening before deadline")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--video", default="data/urfall/fall-01-cam0.mp4")
    p.add_argument("--server", help="Existing MediaMTX executable, required outside Windows")
    p.add_argument("--model", default="models/pose-fp32.engine")
    p.add_argument("--backend", choices=["onnx", "trt", "cpp"], default="trt")
    p.add_argument("--library", help="Native library for --backend cpp")
    p.add_argument("--archive", help="Previously downloaded official Windows release ZIP")
    p.add_argument("--download-timeout", type=float, default=1200,
                   help="Maximum seconds allowed for the official Windows archive download")
    args = p.parse_args()
    if args.download_timeout <= 0:
        p.error("--download-timeout must be positive")
    if not Path(args.video).is_file():
        p.error("--video does not exist")
    if not Path(args.model).is_file():
        p.error("--model does not exist")
    if args.backend == "cpp" and not args.library:
        p.error("--backend cpp requires --library")
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        p.error("Output must be empty")
    root.mkdir(parents=True, exist_ok=True)
    server_path = Path(args.server) if args.server else root / "mediamtx.exe"
    if not args.server:
        if sys.platform != "win32":
            p.error("Supply target-platform MediaMTX with --server")
        archive = Path(args.archive) if args.archive else root / "mediamtx.zip"
        if not archive.exists():
            request = Request("https://api.github.com/repos/bluenviron/mediamtx/releases/assets/623087500",
                              headers={"User-Agent": "poseguard", "Accept": "application/octet-stream"})
            with urlopen(request, timeout=45) as response, archive.open("wb") as destination:
                download_start = time.monotonic()
                while payload := response.read(256 * 1024):
                    destination.write(payload)
                    if time.monotonic() - download_start > args.download_timeout:
                        raise TimeoutError("MediaMTX archive download exceeded --download-timeout")
        payload = archive.read_bytes()
        if len(payload) != ARCHIVE_BYTES:
            raise ValueError(f"MediaMTX archive is incomplete: {len(payload)}/{ARCHIVE_BYTES} bytes")
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError("MediaMTX archive checksum does not match official release digest")
        with zipfile.ZipFile(archive) as bundle:
            server_path.write_bytes(bundle.read("mediamtx.exe"))
            (root / "mediamtx-LICENSE").write_bytes(bundle.read("LICENSE"))
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"rtsp://127.0.0.1:{port}/test"
    server_config = root / "mediamtx.yml"
    server_config.write_text(f"logLevel: debug\nrtspAddress: 127.0.0.1:{port}\nrtspTransports: [tcp]\nrtmp: no\nhls: no\nwebrtc: no\nsrt: no\nmoq: no\napi: no\nmetrics: no\npprof: no\npaths:\n  test:\n    source: publisher\n", encoding="utf-8")
    Config(backend=args.backend, model=args.model, library=args.library, source=url, video_id="loopback-rtsp-real-video",
           show=False, max_reconnects=30, reconnect_delay=.5, duration=45).save(root / "client.yaml")
    handles, processes, faults = [], [], []
    started = time.perf_counter()
    def launch(command, name):
        stream = (root / f"{name}.log").open("w", encoding="utf-8")
        handles.append(stream)
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        processes.append(process)
        return process
    def mark(event):
        row = dict(event=event, elapsed_seconds=time.perf_counter()-started,
                   utc=datetime.now(timezone.utc).isoformat())
        faults.append(row)
        with (root / "faults.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row)+"\n")
    def publisher(name):
        return launch([ffmpeg, "-nostdin", "-re", "-stream_loop", "-1", "-i", args.video,
                       "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                       "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
                       "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp", url], name)
    report = dict(scope="Loopback RTSP real socket failure; recorded URFD RGB video, not physical camera/network test",
                  environment=environment(), server=artifact(server_path), publisher=artifact(ffmpeg),
                  video=artifact(args.video), server_download=dict(url=URL, sha256=SHA256,
                  bytes=ARCHIVE_BYTES, archive=artifact(archive)) if not args.server else None,
                  status="running")
    try:
        server = launch([str(server_path.resolve()), str(server_config.resolve())], "server-1")
        _wait_port(port, server)
        pub = publisher("publisher-1")
        time.sleep(2)
        client_offset = time.perf_counter() - started
        client = launch([sys.executable, "-m", "poseguard", "run", "--config", str(root / "client.yaml"),
                         "--output", str(root / "client")], "client")
        mark("client_started")
        time.sleep(9)
        _stop_process(pub)
        mark("publisher_terminated")
        time.sleep(4)
        pub = publisher("publisher-2")
        mark("publisher_restarted")
        time.sleep(10)
        _stop_process(server)
        mark("server_terminated")
        if pub.poll() is None:
            _stop_process(pub)
        time.sleep(4)
        server = launch([str(server_path.resolve()), str(server_config.resolve())], "server-2")
        _wait_port(port, server)
        pub = publisher("publisher-3")
        mark("server_and_publisher_restarted")
        client.wait(timeout=25)
        summary = json.loads((root / "client/summary.json").read_text(encoding="utf-8"))
        health = [json.loads(line) for line in (root / "client/health.jsonl").read_text(encoding="utf-8").splitlines()]
        with (root / "client/frames.csv").open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        report.update(analyze_recovery(summary, health, faults, client.returncode, frames, client_offset))
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        for process in reversed(processes):
            _stop_process(process)
        for stream in handles:
            stream.close()
        report["faults"] = faults
        report["logs"] = {p.name: artifact(p) for p in root.rglob("*.log")}
        (root / "fault-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], reconnects=report.get("summary", {}).get("reconnects"))))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output")
        known, _ = parser.parse_known_args()
        if known.output:
            destination = Path(known.output) / "fault-report.json"
            if destination.parent.exists() and not destination.exists():
                destination.write_text(json.dumps(dict(status="blocked_before_fault_injection",
                                                       error=f"{type(exc).__name__}: {exc}",
                                                       real_faults_executed=False), indent=2), encoding="utf-8")
        raise
