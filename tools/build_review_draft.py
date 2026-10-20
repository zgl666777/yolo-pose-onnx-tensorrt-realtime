"""Build a clearly non-authoritative review queue from URFD coarse hints.

The output deliberately leaves identity and four-class posture blank. It is a
review aid for a human annotator, never accepted by ``load_annotations``.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from poseguard.annotations import file_sha256
from poseguard.config import Config
from poseguard.features import RuleClassifier
from poseguard.geometry import preprocess, decode
from poseguard.image_io import read_image
from poseguard.tracking import Tracker


def suggested_people(task, detections, tracker, classifier):
    """Model suggestions keep local tracking separate from real identity."""
    people = []
    for track in tracker.update(detections, float(task["timestamp"])):
        probabilities = classifier.predict(track.detection)
        hint = max(probabilities, key=probabilities.get)
        people.append({"bbox": track.detection.bbox.tolist(),
                       "confidence": track.detection.confidence,
                       "source": "model_suggestion",
                       "posture_hint": hint, "posture_hint_source": "untrained_pose_rule",
                       "posture_hint_probabilities": probabilities,
                       "candidate_local_actor_id": f'{task["video_id"]}:local_actor_{track.id}',
                       "identity_scope": "video_local_tracker_only",
                       "identity_verified": False, "person_id": "", "label": None,
                       "review_status": "pending", "human_review_pending": True, "authoritative": False})
    return people


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="data/urfall-review-dense-rgb/review-tasks.jsonl")
    parser.add_argument("--output", default="data/urfall-review-draft")
    parser.add_argument("--suggest-backend", choices=["onnx", "trt"])
    parser.add_argument("--model", help="YOLO pose ONNX/engine used only to suggest boxes")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=.35)
    parser.add_argument("--iou", type=float, default=.45)
    parser.add_argument("--ort-provider", default="CPUExecutionProvider")
    args = parser.parse_args()
    if args.suggest_backend and not args.model:
        parser.error("--suggest-backend requires --model")
    tasks_path, output = Path(args.tasks), Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("output must be empty or new")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    identities = {}
    counts = Counter()
    suggestion_counts = Counter()
    backend = None
    trackers = {}
    classifier = RuleClassifier()
    if args.suggest_backend:
        from poseguard.backends import make_backend
        config = Config(backend=args.suggest_backend, model=args.model, imgsz=args.imgsz,
                        confidence=args.confidence, iou=args.iou, ort_provider=args.ort_provider).validate()
        backend = make_backend(config)
    with tasks_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            task = json.loads(line)
            hint = task.get("official_label_hint")
            suggestion = {None: "unknown", -1: "standing_or_sitting", 0: "transition_candidate",
                          1: "lying_candidate"}.get(hint, "unknown")
            video = task["video_id"]
            identities.setdefault(video, {"video_id": video, "candidate_person_id": "",
                                          "identity_verified": False, "identity_evidence": "",
                                          "status": "unresolved"})
            image_path = (tasks_path.resolve().parent / task["image"]).resolve()
            people = []
            if backend:
                image = read_image(image_path)
                if image is None:
                    raise ValueError(f"Unreadable review image: {image_path}")
                tensor, meta = preprocess(image, args.imgsz)
                detections = decode(backend.infer(tensor).output, meta, args.confidence, args.iou)
                people = suggested_people(task, detections, trackers.setdefault(video, Tracker()), classifier)
                suggestion_counts["processed_tasks"] += 1
                suggestion_counts["tasks_with_boxes"] += bool(people)
                suggestion_counts["no_detection_tasks"] += not bool(people)
                suggestion_counts["box_suggestions"] += len(people)
                if suggestion_counts["processed_tasks"] % 500 == 0:
                    print(f'Generated suggestions for {suggestion_counts["processed_tasks"]} frames', flush=True)
            rows.append({"image": str(image_path), "video_id": video,
                         "source_frame": task.get("source_frame"),
                         "frame_id": task.get("frame_id"), "timestamp": task.get("timestamp"),
                         "source_timestamp": task.get("source_timestamp"),
                         "source_roi": task.get("source_roi"), "bbox": None, "person_id": "",
                         "label": None, "identity_verified": False, "annotation_source": "draft",
                         "review_status": "pending", "suggested_posture": suggestion,
                         "suggested_posture_source": "official_depth_hint",
                         "suggested_people": people,
                         "suggestion_status": "generated" if backend else "not_requested",
                         "authoritative": False, "human_review_pending": True,
                         "review_notes": "Confirm RGB bbox, four-class posture, identity and event membership manually."})
            counts[suggestion] += 1
    with (output / "draft-review.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "identity-registry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "candidate_person_id", "identity_verified",
                                                    "identity_evidence", "status"])
        writer.writeheader(); writer.writerows(identities.values())
    source_hash = file_sha256(tasks_path)
    if backend:
        backend.close()
    manifest = {"schema_version": 3, "status": "draft_requires_human_review", "authoritative": False,
                "human_review_pending": True,
                "tasks": len(rows), "videos": len(identities), "suggestions": dict(counts),
                "source_tasks": str(tasks_path.resolve()),
                "source_tasks_sha256": source_hash,
                "suggestion_generation": {"requested": bool(args.suggest_backend),
                                          "backend": args.suggest_backend,
                                          "model": str(Path(args.model).resolve()) if args.model else None,
                                          "model_sha256": file_sha256(args.model) if args.model else None,
                                          "confidence": args.confidence, "iou": args.iou, "imgsz": args.imgsz,
                                          "coverage": dict(suggestion_counts),
                                          "scope": "Model bbox and rule posture hints only; local tracks are not real identity",
                                          "authoritative": False},
                "completion": {"reviewed_tasks": 0, "remaining_tasks": len(rows),
                                "identity_verified_tasks": 0, "manual_bbox_tasks": 0,
                                "manual_posture_tasks": 0},
                "acceptance": {"final_acceptance_complete": False,
                               "status": "blocked",
                               "blocking_requirements": [
                                   "Human RGB bounding boxes for every person and empty-scene frame",
                                   "Four-class posture labels (standing/sitting/falling/lying)",
                                   "Cross-video identity registry with evidence and reviewer sign-off",
                                   "Manually reviewed event intervals and frozen person-independent test split"]},
                "files": {name: file_sha256(output / name) for name in
                          ("draft-review.jsonl", "identity-registry.csv")},
                "limitations": ["All suggested boxes require human correction and confirmation; missed people must be added",
                                "Rule posture hints are not four-class ground truth", "No real person identity was inferred",
                                "Official depth labels are only review hints and do not define standing/sitting",
                                "draft rows are rejected as evaluation annotations until exported by ReviewStore"]}
    (output / "draft-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
