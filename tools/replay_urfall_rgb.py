"""Compare original RGB and preview-crop diagnostics with identical settings."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time

from poseguard.annotations import file_sha256
from poseguard.backends import make_backend
from poseguard.config import Config
from poseguard.features import RuleClassifier
from poseguard.geometry import preprocess, decode
from poseguard.image_io import read_image
from poseguard.provenance import artifact, environment
from poseguard.tracking import Tracker, TemporalDetector


def replay_tasks(tasks, output, config, backend, media_scope):
    output.mkdir(parents=True)
    config.save(output / "config.yaml")
    temporal, tracker, classifier = TemporalDetector(config), Tracker(config.track_ttl), RuleClassifier()
    alerts = event_id = prediction_count = no_detection = 0
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions, \
            (output / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        events = csv.DictWriter(stream, fieldnames=["video_id", "track_id", "timestamp", "event", "event_id",
                                                   "capture_to_emit_ms", "emitted_unix_ns", "capture_to_emit_scope"])
        events.writeheader()
        for task in tasks:
            start = time.perf_counter_ns()
            image = read_image(task["image"])
            if image is None:
                raise ValueError(f'Unreadable source image: {task["image"]}')
            tensor, meta = preprocess(image, config.imgsz)
            result = backend.infer(tensor)
            detections = decode(result.output, meta, config.confidence, config.iou)
            tracks = tracker.update(detections, float(task["source_timestamp"]))
            no_detection += not bool(tracks)
            for track in tracks:
                probs = classifier.predict(track.detection, config.keypoint_confidence)
                event = temporal.update(track, probs, float(task["source_timestamp"]))
                if event:
                    event_id += 1
                    events.writerow(dict(video_id=config.video_id, track_id=track.id, timestamp=task["source_timestamp"],
                                         event=event, event_id=event_id,
                                         capture_to_emit_ms=(time.perf_counter_ns() - start) / 1e6,
                                         emitted_unix_ns=time.time_ns(), capture_to_emit_scope="offline_read_to_write_call"))
                    alerts += int(event == "fall")
                predictions.write(json.dumps(dict(video_id=config.video_id, frame_id=task["frame_id"],
                                                   timestamp=task["source_timestamp"], track_id=track.id,
                                                   bbox=track.detection.bbox.tolist(), keypoints=track.detection.keypoints.tolist(),
                                                   label=track.label, confidence=track.confidence, probabilities=probs,
                                                   motion_features=track.motion_features, alert=track.alert_active)) + "\n")
                prediction_count += 1
    report = dict(status="completed", scope="exploratory_resolution_comparison_not_final_truth",
                  media_scope=media_scope, frames=len(tasks), video_id=config.video_id, alerts=alerts,
                  no_detection_frames=no_detection, prediction_instances=prediction_count,
                  final_acceptance_complete=False,
                  files={name: file_sha256(output / name) for name in ("config.yaml", "predictions.jsonl", "events.csv")})
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-data", default="data/urfall-original-rgb")
    parser.add_argument("--preview-tasks", default="data/urfall-review-dense-rgb/review-tasks.jsonl")
    parser.add_argument("--sequence", action="append", choices=[f"fall-{i:02d}" for i in range(1, 31)], required=True)
    parser.add_argument("--model", default="models/pose-fp32.engine")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be new or empty")
    output.mkdir(parents=True)
    preview_path = Path(args.preview_tasks).resolve()
    preview = [json.loads(line) for line in preview_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    config = Config.load(args.config, dict(backend="trt", model=args.model, show=False, source_roi=None))
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *sorted((root / "src/poseguard").glob("*.py"))]
    code = {str(p.relative_to(root)): file_sha256(p) for p in paths}
    report = dict(status="running", scope="exploratory_resolution_comparison_not_manual_truth",
                  final_acceptance_complete=False, environment=environment(), model=artifact(args.model),
                  config=artifact(args.config), code=code, sequences=[])
    backend = make_backend(config)
    try:
        for index, sequence in enumerate(args.sequence):
            config.video_id = sequence + "-cam0"
            original_path = Path(args.original_data) / sequence / "review-tasks.jsonl"
            original = [json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            preview_rows = {int(r["source_frame"]): r for r in preview if r["video_id"] == config.video_id}
            paired_preview = []
            for task in original:
                row = dict(preview_rows[task["source_frame"]])
                row["image"] = str((preview_path.parent / row["image"]).resolve())
                if row["source_timestamp"] != task["source_timestamp"]:
                    raise ValueError("Original/preview source timestamp mismatch")
                paired_preview.append(row)
            summaries = []
            for name, tasks, scope in (("original-fp32", original, "original_full_resolution_RGB"),
                                       ("preview-fp32", paired_preview, "320x240_RGB_preview_crop")):
                summaries.append(replay_tasks(tasks, output / f"{name}-v{index:02d}", config, backend, scope))
            report["sequences"].append(dict(sequence=sequence, source_tasks=artifact(original_path),
                                           preview_tasks=artifact(preview_path), results=summaries))
            print(json.dumps(dict(sequence=sequence, results=summaries), indent=2), flush=True)
        report["status"] = "completed"
        report["code_unchanged_during_run"] = all(file_sha256(p) == code[str(p.relative_to(root))] for p in paths)
    finally:
        backend.close()
        (output / "comparison-manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
