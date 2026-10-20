"""Inspect exploratory replay failures without changing detector parameters."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from poseguard.annotations import file_sha256
from poseguard.config import Config
from poseguard.image_io import write_image
from poseguard.tracking import Track, TemporalDetector
from poseguard.types import Detection


def inspect(directory, result, output):
    config = Config.load(str(directory / "config.yaml"))
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines() if line.strip()]
    temporal = TemporalDetector(config)
    tracks, trace = {}, []
    for row in rows:
        detection = Detection(np.asarray(row["bbox"]), float(row["confidence"]), np.asarray(row["keypoints"]))
        track = tracks.setdefault(row["track_id"], Track(row["track_id"], detection, row["timestamp"]))
        track.detection, track.last_seen = detection, row["timestamp"]
        event = temporal.update(track, row["probabilities"], row["timestamp"])
        width, height = detection.bbox[2:] - detection.bbox[:2]
        trace.append(dict(frame_id=row["frame_id"], timestamp=row["timestamp"], track_id=track.id,
                          label=track.label, probabilities=row["probabilities"],
                          reliable_torso=track.motion_features.get("reliable", False),
                          visible_ratio=float((detection.keypoints[:, 2] >= config.keypoint_confidence).mean()),
                          width_height_ratio=float(width / max(height, 1)),
                          motion=track.motion_features, motion_strength=track.motion_strength,
                          last_upright=track.last_upright, landing_evidence_at=track.landing_evidence_at,
                          evidence_valid_until=track.motion_evidence_valid_until,
                          candidate_since=track.candidate_since, event=event))
    with (directory / "frames.csv").open(newline="") as stream:
        frames = list(csv.DictReader(stream))
    center = result["alarms"][0] if result["fp"] else result["onset"]
    offsets = [-1.2, -.6, -.2, 0, .6, 1.2] if result["fp"] else [-.3, 0, .35, .7, 1.1, 1.5]
    cap = cv2.VideoCapture(config.source)
    panel_w, panel_h = 480, 450
    canvas = np.full((2 * panel_h, 3 * panel_w, 3), 24, dtype=np.uint8)
    chosen = []
    for index, offset in enumerate(offsets):
        target = max(0, center + offset)
        frame = min(frames, key=lambda f: abs(float(f["timestamp"]) - target))
        frame_id = int(frame["frame_id"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = cap.read()
        if not ok:
            continue
        image = image[:, 320:].copy()
        selected = [r for r in rows if r["frame_id"] == frame_id]
        chosen.extend(t for t in trace if t["frame_id"] == frame_id)
        for row in selected:
            box = [int(v) for v in row["bbox"]]
            cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (70, 230, 180), 1)
        resized = cv2.resize(image, (480, 360))
        y, x = (index // 3) * panel_h, (index % 3) * panel_w
        canvas[y:y + 360, x:x + 480] = resized
        descriptions = [f'{config.video_id} f{frame_id} t={float(frame["timestamp"]):.3f}',
                        "; ".join(f'id{r["track_id"]} {r["label"]}' for r in selected) or "no detection"]
        if selected:
            item = max((t for t in trace if t["frame_id"] == frame_id),
                       key=lambda t: t["visible_ratio"])
            descriptions.append(f'visible={item["visible_ratio"]:.2f} W/H={item["width_height_ratio"]:.2f} motion={item["motion_strength"]:.2f}')
        for line_index, description in enumerate(descriptions):
            cv2.putText(canvas, description[:58], (x + 8, y + 381 + line_index * 23), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, (240, 240, 240), 1, cv2.LINE_AA)
    cap.release()
    write_image(output / f"{config.video_id}.jpg", canvas)
    relevant = [t for t in trace if max(0, center - 1.5) <= t["timestamp"] <= center + 2]
    return dict(**result, diagnostic_window=relevant, selected_frames=chosen,
                prediction_sha256=file_sha256(directory / "predictions.jsonl"),
                maximum_visible_ratio=max((t["visible_ratio"] for t in relevant), default=0),
                reliable_torso_samples=sum(t["reliable_torso"] for t in relevant),
                samples=len(relevant),
                max_motion_strength=max((t["motion_strength"] for t in relevant), default=0),
                max_vertical_speed=max((t["motion"].get("hip_down_body_heights_per_second", 0) for t in relevant), default=0),
                max_tilt_change=max((t["motion"].get("torso_tilt_change", 0) for t in relevant), default=0),
                candidate_samples=sum(t["candidate_since"] is not None for t in relevant))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.run_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((root / "official-coarse-report.json").read_text())
    results = report["reports"]["fp32"]["video_results"]
    directories = {Config.load(str(d / "config.yaml")).video_id: d for d in root.glob("fp32-v*")}
    inspections = [inspect(directories[r["video_id"]], r, output) for r in results if r["fp"] or r["fn"]]
    value = {"scope": "exploratory_error_analysis_not_manual_truth", "videos": inspections,
             "report_sha256": file_sha256(root / "official-coarse-report.json"),
             "detector_code_sha256": file_sha256(Path(__file__).resolve().parents[1] / "src/poseguard/tracking.py")}
    (output / "diagnostics.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    for row in inspections:
        print(row["video_id"], "FP" if row["fp"] else "FN", "samples", row["samples"],
              "torso", row["reliable_torso_samples"], "motion", round(row["max_motion_strength"], 3),
              "speed", round(row["max_vertical_speed"], 3), "tilt", round(row["max_tilt_change"], 3),
              "candidate", row["candidate_samples"])


if __name__ == "__main__":
    main()
