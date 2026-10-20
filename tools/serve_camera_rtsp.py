"""Publish a real Windows camera on a specified host-only interface for Linux VM testing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import ipaddress
from pathlib import Path
import shutil
import socket
import subprocess
import time

from poseguard.provenance import artifact, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="already verified MediaMTX executable")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bind", required=True, help="Windows VM host-only interface IP; no wildcard address")
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--camera", default="HP True Vision FHD Camera")
    parser.add_argument("--duration", type=float, default=3900)
    parser.add_argument("--fault-after", type=float, default=0,
                        help="Terminate publisher once after this many serving seconds, then restart it")
    parser.add_argument("--fault-seconds", type=float, default=4)
    args = parser.parse_args()
    try:
        bind_address = ipaddress.ip_address(args.bind)
    except ValueError:
        parser.error("--bind must be an IP address from the host-only interface")
    if bind_address.is_unspecified:
        parser.error("Use a specific host-only interface, not a wildcard bind")
    if bind_address.version != 4:
        parser.error("This Windows host-only publisher currently requires an IPv4 bind address")
    if args.duration <= 0 or args.fault_after < 0 or args.fault_seconds <= 0 or not 1 <= args.port <= 65535:
        parser.error("duration and port must be valid")
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        parser.error("output must be a new empty directory")
    root.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not Path(args.server).is_file():
        parser.error("verified MediaMTX and ffmpeg are required")
    url = f"rtsp://{args.bind}:{args.port}/soak"
    configuration = root / "mediamtx.yml"
    configuration.write_text(f"logLevel: info\nrtspAddress: {args.bind}:{args.port}\nrtspTransports: [tcp]\nrtmp: no\nhls: no\nwebrtc: no\nsrt: no\nmoq: no\napi: no\nmetrics: no\npprof: no\npaths:\n  soak:\n    source: publisher\n", encoding="utf-8")
    processes, handles = [], []
    report = {"status": "starting", "created_utc": datetime.now(timezone.utc).isoformat(),
              "environment": environment(), "camera": args.camera, "url": url,
              "scope": "real integrated Windows camera encoded over RTSP to a Linux VM; not an external RTSP camera",
              "server": artifact(args.server), "ffmpeg": artifact(ffmpeg), "bind": args.bind,
              "firewall_modified": False}
    started = time.perf_counter()

    def save():
        (root / "server-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def launch(command, name):
        handle = (root / f"{name}.log").open("w", encoding="utf-8")
        handles.append(handle)
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def publish(name):
        return launch([ffmpeg, "-nostdin", "-f", "dshow", "-framerate", "30", "-video_size", "640x480",
                       "-i", f"video={args.camera}", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                       "-tune", "zerolatency", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
                       "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp", url], name)

    def mark(event):
        row = {"event": event, "utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.perf_counter() - started}
        report.setdefault("faults", []).append(row)
        with (root / "faults.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        save()
        print(json.dumps(row), flush=True)

    try:
        server = launch([str(Path(args.server).resolve()), str(configuration.resolve())], "mediamtx")
        for _ in range(50):
            if server.poll() is not None:
                raise RuntimeError(f"MediaMTX exited {server.returncode}")
            try:
                with socket.create_connection((args.bind, args.port), timeout=.2):
                    break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError("MediaMTX did not listen before deadline")
        publisher = publish("publisher-1")
        time.sleep(3)
        if publisher.poll() is not None:
            raise RuntimeError(f"Camera publisher exited {publisher.returncode}")
        report.update(status="serving", server_pid=server.pid, publisher_pid=publisher.pid,
                      available_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps({"status": "serving", "url": url, "server_pid": server.pid,
                          "publisher_pid": publisher.pid}), flush=True)
        serving_started, fault_done = time.perf_counter(), False
        while time.perf_counter() - started < args.duration:
            if args.fault_after > 0 and not fault_done and time.perf_counter() - serving_started >= args.fault_after:
                publisher.terminate()
                publisher.wait(timeout=5)
                mark("camera_publisher_terminated")
                time.sleep(args.fault_seconds)
                publisher = publish("publisher-2")
                mark("camera_publisher_restarted")
                time.sleep(3)
                if publisher.poll() is not None:
                    raise RuntimeError("Restarted camera publisher exited")
                report["publisher_pid"] = publisher.pid
                mark("camera_publisher_stable")
                fault_done = True
            if server.poll() is not None or publisher.poll() is not None:
                raise RuntimeError("RTSP server or camera publisher stopped unexpectedly")
            time.sleep(.5)
        report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "stopped_by_operator"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for handle in handles:
            handle.close()
        report.update(elapsed_seconds=time.perf_counter() - started,
                      logs={path.name: artifact(path) for path in root.glob("*.log")})
        save()


if __name__ == "__main__":
    main()
