"""Sequential full-frame URFD exploration with source timestamp alignment.

Official coarse depth labels and sequence events are review hints. This run is
exploration/tuning evidence and never establishes manual final-test truth.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2

from poseguard.annotations import file_sha256
from poseguard.backends import make_backend
from poseguard.config import Config
from poseguard.features import RuleClassifier
from poseguard.geometry import preprocess, decode
from poseguard.provenance import artifact, environment
from poseguard.tracking import Tracker, TemporalDetector


def replay_video(video, output, config, backend):
    output.mkdir(parents=True)
    config.save(output / "config.yaml")
    timestamp_path = video.with_name(video.name.replace("-cam0.mp4", "-data.csv"))
    with timestamp_path.open(newline="", encoding="utf-8") as stream:
        times = {int(row[0]): float(row[1]) / 1000 for row in csv.reader(stream)}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Unreadable input video: {video}")
    expected_frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    tracker, temporal, classifier = Tracker(config.track_ttl), TemporalDetector(config), RuleClassifier()
    processed = alerts = event_id = prediction_instances = missing_sync = 0
    previous_timestamp = None
    start = time.perf_counter()
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions, \
                (output / "events.csv").open("w", newline="", encoding="utf-8") as event_stream, \
                (output / "frames.csv").open("w", newline="", encoding="utf-8") as frame_stream:
            events = csv.DictWriter(event_stream, fieldnames=["video_id", "track_id", "timestamp", "event",
                                     "event_id", "capture_to_emit_ms", "emitted_unix_ns", "capture_to_emit_scope"])
            events.writeheader()
            frames = csv.DictWriter(frame_stream, fieldnames=["video_id", "frame_id", "timestamp", "playback_timestamp",
                                     "people", "processing_ms", "backend_ms"])
            frames.writeheader()
            while True:
                captured_ns = time.perf_counter_ns()
                ok, image = cap.read()
                if not ok:
                    break
                if image.shape[:2] != (240, 640):
                    raise ValueError("URFD composite preview must be 640x240 before RGB crop")
                playback_timestamp = processed / fps
                timestamp = times.get(processed + 1)
                if timestamp is None:
                    missing_sync += 1
                    timestamp = playback_timestamp if previous_timestamp is None else previous_timestamp + 1 / fps
                previous_timestamp = timestamp
                image = image[:, 320:].copy()
                tensor, meta = preprocess(image, config.imgsz)
                result = backend.infer(tensor)
                detections = decode(result.output, meta, config.confidence, config.iou)
                tracks = tracker.update(detections, timestamp)
                for track in tracks:
                    probabilities = classifier.predict(track.detection, config.keypoint_confidence)
                    event = temporal.update(track, probabilities, timestamp)
                    if event:
                        event_id += 1
                        events.writerow(dict(video_id=config.video_id, track_id=track.id, timestamp=timestamp,
                                             event=event, event_id=event_id,
                                             capture_to_emit_ms=(time.perf_counter_ns() - captured_ns) / 1e6,
                                             emitted_unix_ns=time.time_ns(),
                                             capture_to_emit_scope="offline_frame_read_to_event_write_call"))
                        alerts += int(event == "fall")
                    row = dict(video_id=config.video_id, frame_id=processed, source_frame=processed + 1,
                               timestamp=timestamp, playback_timestamp=playback_timestamp,
                               track_id=track.id, bbox=track.detection.bbox.tolist(),
                               keypoints=track.detection.keypoints.tolist(), label=track.label,
                               confidence=track.confidence, probabilities=probabilities,
                               motion_features=track.motion_features, alert=track.alert_active)
                    predictions.write(json.dumps(row) + "\n")
                    prediction_instances += 1
                frames.writerow(dict(video_id=config.video_id, frame_id=processed, timestamp=timestamp,
                                     playback_timestamp=playback_timestamp, people=len(tracks),
                                     processing_ms=(time.perf_counter_ns() - captured_ns) / 1e6,
                                     backend_ms=result.timings.get("backend_ms")))
                processed += 1
    finally:
        cap.release()
    if processed != expected_frames:
        raise ValueError(f"Decoded {processed}/{expected_frames} frames for {video}")
    summary = {"status": "completed", "scope": "exploratory_tuning_not_final_test", "final_acceptance_complete": False,
               "video_id": config.video_id, "frames": processed, "expected_frames": expected_frames,
               "prediction_instances": prediction_instances, "alerts": alerts, "dropped_frames": 0,
               "missing_source_timestamps": missing_sync, "elapsed_seconds": time.perf_counter() - start,
               "timestamp_scope": "official source sync CSV seconds; absent entries extrapolated at source FPS and counted; playback_timestamp retained separately",
               "latency_scope": "offline per-frame processing only, not real-time end-to-end camera latency",
               "inputs": {"video": artifact(video), "timestamps": artifact(timestamp_path)},
               "files": {name: file_sha256(output / name) for name in
                         ("config.yaml", "predictions.jsonl", "events.csv", "frames.csv")}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/urfall")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/pose-fp32.engine")
    parser.add_argument("--backend", choices=["trt", "onnx"], default="trt")
    parser.add_argument("--config", help="Optional frozen detector settings; no fitting is performed")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        parser.error("Output must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__).resolve(), *sorted((source_root / "src/poseguard").glob("*.py"))]
    code = {str(path.relative_to(source_root)): file_sha256(path) for path in code_paths}
    config = Config.load(args.config, dict(backend=args.backend, model=args.model, source_roi=[320, 0, 640, 240], show=False))
    videos = sorted(Path(args.data).glob("*-cam0.mp4"))
    if not videos:
        parser.error("No URFD camera 0 videos found")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "scope": "exploratory_tuning_not_final_test", "final_acceptance_complete": False,
              "environment": environment(), "model": artifact(args.model), "code": code,
              "configuration": artifact(args.config), "videos": [],
              "limitations": ["No verified cross-video identity, RGB manual boxes or four-class manual ground truth",
                              "Official depth timing/labels require human RGB alignment review",
                              "Offline sequential inference removes real-time dropping; throughput is not camera latency",
                              "All videos are exploratory/tuning data; cannot serve as independent final test"]}
    destination = root / "replay-manifest.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    backend = make_backend(config)
    try:
        for index, video in enumerate(videos):
            config.video_id, config.source = video.stem, str(video)
            summary = replay_video(video, root / f"fp32-v{index:02d}", config, backend)
            report["videos"].append(summary)
            print(f'{index + 1}/{len(videos)} {video.stem}: frames={summary["frames"]} alerts={summary["alerts"]}', flush=True)
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["status"] = "completed"
        report["total_frames"] = sum(v["frames"] for v in report["videos"])
        report["total_alerts"] = sum(v["alerts"] for v in report["videos"])
        report["code_unchanged_during_run"] = all(file_sha256(path) == code[str(path.relative_to(source_root))]
                                                    for path in code_paths)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        backend.close()
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
