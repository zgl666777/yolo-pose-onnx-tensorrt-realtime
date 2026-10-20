"""Run reproducible Linux/GPU acceptance, or explicitly limited Windows exploration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import signal
import shutil
import importlib.util
import cv2
from poseguard.annotations import file_sha256
from poseguard.provenance import environment, artifact
from poseguard.image_io import write_image


def _probe_command(command, timeout=4):
    """Return a small, JSON-safe command probe for the acceptance record."""
    try:
        resolved = str(Path(command[0]).resolve()) if Path(command[0]).is_file() else shutil.which(command[0])
        if not resolved:
            return {"available": False}
        result = subprocess.run(command, capture_output=True, timeout=timeout)
        raw = (result.stdout or b"") + (result.stderr or b"")
        encoding = "utf-16-le" if b"\x00" in raw[:128] else "utf-8"
        output = raw.decode(encoding, errors="replace").strip()
        return {"available": True, "path": resolved, "returncode": result.returncode,
                "output": output[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}


def virtualization_preflight():
    """Collect evidence useful when the Linux target is a VMware guest.

    This is deliberately informational: a guest cannot prove that its host
    has a passed-through NVIDIA device.  The CPU profile therefore records
    GPU/TensorRT as an explicit skipped capability instead of guessing.
    """
    result = {"guest": False, "signals": {}, "commands": {}}
    if platform.system() == "Linux":
        for path in ("/sys/class/dmi/id/product_name", "/sys/devices/virtual/dmi/id/product_name",
                     "/sys/class/dmi/id/sys_vendor", "/sys/class/dmi/id/board_vendor"):
            try:
                result["signals"][path] = Path(path).read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                pass
        result["commands"]["systemd-detect-virt"] = _probe_command(["systemd-detect-virt", "--vm"])
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            result["signals"]["cpu_hypervisor_flag"] = "hypervisor" in cpuinfo
        except OSError:
            pass
        product = " ".join(str(v).lower() for v in result["signals"].values())
        result["guest"] = bool(result["signals"].get("cpu_hypervisor_flag") or
                                any(token in product for token in ("vmware", "virtualbox", "kvm", "microsoft")))
    else:
        # Windows host evidence helps the operator discover that VMware is
        # installed/running before moving the workflow into the guest.
        result["commands"]["wsl"] = _probe_command(["wsl.exe", "--status"])
        result["commands"]["nvidia-smi"] = _probe_command(["nvidia-smi", "--query-gpu=name,driver_version",
                                                               "--format=csv,noheader"])
        vmrun = shutil.which("vmrun")
        if not vmrun:
            candidate = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "VMware/VMware Workstation/vmrun.exe"
            if candidate.is_file():
                vmrun = str(candidate)
        result["commands"]["vmrun"] = _probe_command([vmrun or "vmrun", "-T", "ws", "list"])
        try:
            result["signals"]["vmware_processes"] = [p.info.get("name") for p in
                                                       __import__("psutil").process_iter(["name"])
                                                       if str(p.info.get("name") or "").lower().startswith("vmware")]
        except Exception as exc:  # discovery must never block acceptance
            result["signals"]["vmware_processes_error"] = str(exc)
    result["interpretation"] = ("VMware/virtualized guest detected; CPU/ONNX evidence is valid, "
                                 "but TensorRT/GPU requires PCIe passthrough or a native Linux host."
                                 if result["guest"] else
                                 "No VMware guest signal detected; inspect the recorded signals before claiming VM coverage.")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--onnx", default="models/yolo11n-pose.onnx")
    p.add_argument("--weights", default="models/yolo11n-pose.pt")
    p.add_argument("--fp32", default="models/pose-fp32.engine")
    p.add_argument("--fp16", default="models/pose-mixed-strict.engine")
    p.add_argument("--fp16-build-mode", choices=["fp16-strict", "fp16"], default="fp16-strict",
                   help="strict is constrained mixed precision; use fp16 only for separately reported exploratory comparison")
    p.add_argument("--library", default="build/cpp/libpose_trt.so")
    p.add_argument("--video", action="append", default=[])
    p.add_argument("--source-roi",type=int,nargs=4)
    p.add_argument("--classifier")
    p.add_argument("--ground-truth")
    p.add_argument("--truth-events")
    p.add_argument("--ground-truth-frames")
    p.add_argument("--split-manifest")
    p.add_argument("--training-report")
    p.add_argument("--stage-timeout", type=float, default=1200)
    p.add_argument("--allow-windows", action="store_true")
    p.add_argument("--profile", choices=["auto", "linux-cpu", "linux-gpu"], default="auto",
                   help="auto selects GPU only when TensorRT and engines are available; linux-cpu is VMware-safe")
    p.add_argument("--preflight-only", action="store_true",
                   help="write environment/virtualization evidence and exit without running inference")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--camera-config", action="append", default=[])
    p.add_argument("--soak-seconds", type=float, default=28800)
    args = p.parse_args()
    if not args.video and not args.preflight_only:
        p.error("--video is required unless --preflight-only is selected")
    if args.stage_timeout <= 0 or args.soak_seconds <= 0:
        p.error("Timeout and soak duration must be positive")
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        p.error("Output must be a new directory")
    root.mkdir(parents=True, exist_ok=True)
    target_linux = platform.system() == "Linux"
    preflight = virtualization_preflight()
    has_tensorrt = importlib.util.find_spec("tensorrt") is not None
    has_engines = Path(args.fp32).is_file() and Path(args.fp16).is_file()
    if args.profile == "auto":
        profile = "linux-gpu" if target_linux and has_tensorrt and has_engines else "linux-cpu"
    else:
        profile = args.profile
    if profile == "linux-gpu" and not target_linux:
        p.error("--profile linux-gpu requires a Linux host (VMware CPU guests must use --profile linux-cpu)")
    report = dict(environment=environment(), target_linux=target_linux, profile=profile,
                  engine_build_modes={"fp32": "fp32", "fp16": args.fp16_build_mode},
                  virtualization=preflight, capabilities={
                      "onnx_cpu": True,
                      "tensorrt_importable": has_tensorrt,
                      "fp32_engine_inputs": has_engines,
                      "gpu_acceptance": profile == "linux-gpu",
                      "physical_usb": False,
                      "real_rtsp_fault_injection": False,
                  }, stages=[],
                  scope="Exploratory real-video replay; final acceptance also requires frozen independent manual test data and real fault logs",
                  models={k: artifact(getattr(args, k)) for k in ("onnx", "weights", "fp32", "fp16", "library", "classifier")},
                  videos=[artifact(v) for v in args.video], status="running",
                  manual_review_gate={"status": "blocked", "reason": "No verified manual person-independent test export supplied"})
    def save():
        (root / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    def run(name, command, timeout=None):
        start = time.perf_counter()
        log = root / f"{name}.log"
        with log.open("w", encoding="utf-8") as stream:
            options = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else dict(start_new_session=True)
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, **options)
            timed_out = False
            try:
                exit_code = process.wait(timeout=timeout or args.stage_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.name == "nt":
                    subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],stdout=stream,stderr=stream,timeout=10)
                else:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=10)
                exit_code = 124
        report["stages"].append(dict(name=name, command=command, exit_code=exit_code,timed_out=timed_out,
                                      elapsed_seconds=time.perf_counter()-start, log=log.name, log_sha256=file_sha256(log)))
        save()
        print(f"{name}: {'OK' if exit_code == 0 else 'FAILED'}", flush=True)
        return exit_code == 0
    save()
    if args.preflight_only:
        report.update(status="preflight_only", final_acceptance_complete=False,
                      remaining=["Run the selected profile with --video on the Linux target",
                                 "Physical USB and real RTSP fault evidence",
                                 "Long live soak on target Linux/GPU with fault timestamps"])
        save()
        print(json.dumps({"status": report["status"], "profile": profile,
                          "virtualization": preflight}, indent=2))
        return 0
    if not report["target_linux"] and not args.allow_windows:
        report.update(status="blocked", reason="Target Linux not available on this host")
        save()
        return 2
    python = [sys.executable, "-m", "poseguard"]
    video_rois = {Path(v).stem:args.source_roi for v in args.video}
    if args.ground_truth:
        if not args.ground_truth_frames or not args.split_manifest or not args.training_report or not args.classifier:
            raise ValueError("Final real scoring requires frame coverage, frozen split manifest, training report and learned classifier")
        manifest_path = Path(args.split_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        training = json.loads(Path(args.training_report).read_text(encoding="utf-8"))
        test = manifest["splits"]["test"]
        if (not manifest.get("identity_verified") or training.get("synthetic") or
                not training.get("four_class_training_coverage") or
                training.get("model_sha256") != file_sha256(args.classifier) or
                training.get("split_manifest_sha256") != file_sha256(manifest_path) or
                test["sha256"] != file_sha256(args.ground_truth) or
                test.get("frames_sha256") != file_sha256(args.ground_truth_frames) or
                (args.truth_events and test["events_sha256"] != file_sha256(args.truth_events))):
            raise ValueError("Model/training/test hashes or real four-class coverage do not agree with frozen experiment")
        from poseguard.evaluation import load_frame_manifest
        ledger = load_frame_manifest(args.ground_truth_frames,"person_count")
        if {v for v,_ in ledger} != {Path(v).stem for v in args.video}:
            raise ValueError("Replay videos must exactly cover frozen test video IDs")
        for video in video_rois:
            values = {json.dumps(r.get("source_roi")) for (v,_),r in ledger.items() if v == video}
            if len(values) != 1:
                raise ValueError("Frame truth has inconsistent source crops for a video")
            video_rois[video] = json.loads(values.pop())
        report["frozen_test"] = dict(manifest=artifact(manifest_path),annotations=artifact(args.ground_truth),
                                     frames=artifact(args.ground_truth_frames),training=artifact(args.training_report))
        report["manual_review_gate"] = {"status": "verified_frozen_test",
                                        "reason": "Identity-verified manual test, frame ledger and real training hashes agree"}
    if args.rebuild and profile != "linux-gpu":
        report["stages"].append(dict(name="rebuild", status="skipped",
                                      reason="--rebuild requires the linux-gpu profile; VMware Ubuntu CPU acceptance uses ONNX Runtime"))
        save()
    if args.rebuild and profile == "linux-gpu":
        if not run("cmake-configure", ["cmake", "-S", ".", "-B", "build-acceptance",
                                      f"-DTENSORRT_ROOT={os.environ.get('TENSORRT_ROOT', '')}"]):
            report["status"] = "failed"
            save()
            return 1
        if not run("cmake-build", ["cmake", "--build", "build-acceptance", "-j"]):
            report["status"] = "failed"
            save()
            return 1
        args.library = str(Path("build-acceptance/cpp/libpose_trt.so").resolve())
        for precision in ("fp32", "fp16"):
            path = str(root / f"{precision}.engine")
            mode = args.fp16_build_mode if precision == "fp16" else "fp32"
            if not run(f"build-{precision}", python + ["build-engine", "--onnx", args.onnx,
                                                        "--output", path, "--precision", mode]):
                report["status"] = "failed"
                save()
                return 1
            setattr(args, precision, path)
        report["models"].update({k: artifact(getattr(args, k)) for k in ("fp32", "fp16", "library")})
    images = root / "precision-images"
    images.mkdir()
    for index, video in enumerate(args.video):
        cap = cv2.VideoCapture(video)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or count < 1:
            raise ValueError(f"Unreadable input video: {video}")
        for fraction in (0.0, .25, .5, .75, .95):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((count - 1) * fraction))
            ok, image = cap.read()
            if ok:
                roi = video_rois[Path(video).stem]
                if roi:
                    x1,y1,x2,y2=roi
                    image=image[y1:y2,x1:x2].copy()
                write_image(images / f"v{index}-{fraction}.png", image)
        cap.release()
    verify = python + ["verify", "--onnx", args.onnx, "--images", str(images),
                       "--output", str(root / "precision.json")]
    if profile == "linux-gpu":
        verify += ["--engine", args.fp32, "--engine", args.fp16]
    weights_comparable = (Path(args.weights).is_file() and importlib.util.find_spec("torch") is not None and
                          importlib.util.find_spec("ultralytics") is not None)
    if weights_comparable:
        verify += ["--weights", args.weights]
    if profile == "linux-gpu" or weights_comparable:
        run("precision", verify)
    else:
        report["stages"].append(dict(name="precision", status="skipped",
                                      reason="CPU profile has no independent PyTorch reference; install .[export] and supply --weights to compare"))
        save()
    variants = [("onnx", "onnx", args.onnx)]
    if profile == "linux-gpu":
        variants += [("trt-fp32", "trt", args.fp32), ("trt-fp16", "trt", args.fp16),
                     ("cpp-fp32", "cpp", args.fp32), ("cpp-fp16", "cpp", args.fp16)]
    else:
        report["stages"].extend([
            dict(name="trt-fp32", status="skipped", reason="linux-cpu/VMware profile; no TensorRT GPU claim"),
            dict(name="trt-fp16", status="skipped", reason="linux-cpu/VMware profile; no TensorRT GPU claim"),
            dict(name="cpp-fp32", status="skipped", reason="linux-cpu/VMware profile; C++ TensorRT backend not required"),
            dict(name="cpp-fp16", status="skipped", reason="linux-cpu/VMware profile; C++ TensorRT backend not required"),
        ])
    for name, backend, model in variants:
        extra = ["--library", args.library] if backend == "cpp" else []
        run(f"benchmark-{name}", python + ["benchmark", "--backend", backend, "--model", model,
                                           "--image", str(next(images.glob("*.png"))), "--iterations", "200",
                                           "--output", str(root / f"benchmark-{name}.json")] + extra)
        for index, video in enumerate(args.video):
            video_id = Path(video).stem
            output = root / f"{name}-v{index}"
            extra_run = extra + (["--classifier", args.classifier] if args.classifier else [])
            if video_rois[video_id]:
                extra_run += ["--source-roi"] + [str(v) for v in video_rois[video_id]]
            if run(f"replay-{name}-v{index}", python + ["run", "--backend", backend, "--model", model,
                                                        "--source", video, "--video-id", video_id, "--headless",
                                                        "--output", str(output)] + extra_run):
                run(f"latency-{name}-v{index}", [sys.executable, "tools/soak_report.py", "--run", str(output),
                                                "--output", str(output / "performance.json")])
        if args.ground_truth:
            combined = root / f"{name}-predictions.jsonl"
            combined_events = root / f"{name}-events.csv"
            combined_frames = root / f"{name}-frames.csv"
            combined_delivery = root / f"{name}-event-delivery.jsonl"
            import csv
            with combined.open("w", encoding="utf-8") as stream, combined_events.open("w", newline="", encoding="utf-8") as events, \
                    combined_frames.open("w",newline="",encoding="utf-8") as frames, combined_delivery.open("w",encoding="utf-8") as delivery:
                writer = frame_writer = None
                for index in range(len(args.video)):
                    directory = root / f"{name}-v{index}"
                    if (directory / "predictions.jsonl").exists():
                        stream.write((directory / "predictions.jsonl").read_text(encoding="utf-8"))
                    if (directory / "events.csv").exists():
                        with (directory / "events.csv").open(newline="", encoding="utf-8") as source:
                            reader = csv.DictReader(source)
                            if writer is None:
                                writer = csv.DictWriter(events, fieldnames=reader.fieldnames)
                                writer.writeheader()
                            for row in reader:
                                row["event_id"] = f'{index}-{row["event_id"]}'
                                writer.writerow(row)
                    with (directory / "frames.csv").open(newline="",encoding="utf-8") as source:
                        reader=csv.DictReader(source)
                        if frame_writer is None:
                            frame_writer=csv.DictWriter(frames,fieldnames=reader.fieldnames)
                            frame_writer.writeheader()
                        frame_writer.writerows(reader)
                    for line in (directory / "event-delivery.jsonl").read_text(encoding="utf-8").splitlines():
                        row=json.loads(line)
                        row["event_id"]=f'{index}-{row["event_id"]}'
                        delivery.write(json.dumps(row)+"\n")
            command = python + ["evaluate", "--ground-truth", args.ground_truth, "--predictions", str(combined),
                                 "--ground-truth-frames",args.ground_truth_frames,"--prediction-frames",str(combined_frames),
                                 "--output", str(root / f"evaluation-{name}.json")]
            if args.truth_events:
                command += ["--events", str(combined_events), "--truth-events", args.truth_events,
                            "--deadline-scope","emitted","--event-delivery",str(combined_delivery)]
            run(f"evaluation-{name}", command)
    for index, config in enumerate(args.camera_config):
        camera_dir = root / f"camera-{index}"
        cpu_options = (["--backend", "onnx", "--model", args.onnx, "--ort-provider", "CPUExecutionProvider"]
                       if profile == "linux-cpu" else [])
        if run(f"camera-{index}", python + ["run", "--config", config, "--headless", "--duration", str(args.soak_seconds),
                                            "--output", str(camera_dir)] + cpu_options,timeout=args.soak_seconds+120):
            run(f"camera-summary-{index}", [sys.executable, "tools/soak_report.py", "--run", str(camera_dir),
                                            "--output", str(camera_dir / "soak.json")])
    failed_stages = [s for s in report["stages"] if s.get("exit_code", 0) != 0]
    report["status"] = "completed_with_failures" if failed_stages else "completed"
    report["final_acceptance_complete"] = False
    report["remaining"] = ["Physical USB and real RTSP disconnect/reconnect fault evidence"]
    if report["manual_review_gate"]["status"] != "verified_frozen_test":
        report["remaining"].insert(0, "Cross-video person identities and independent manual test-set approval")
    if profile == "linux-gpu":
        report["remaining"].append("Long live soak on target Linux/GPU with fault timestamps")
    else:
        report["remaining"].append("Long live soak on target Linux CPU (run --camera-config with a real Ubuntu input)")
    save()
    return 1 if report["status"] == "completed_with_failures" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output")
        known, _ = parser.parse_known_args()
        if known.output:
            destination = Path(known.output) / "acceptance.json"
            if destination.exists():
                report = json.loads(destination.read_text(encoding="utf-8"))
                report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
