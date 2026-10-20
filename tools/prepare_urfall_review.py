"""Import official coarse labels and prepare review tasks without inventing identities or boxes."""
import argparse
import csv
import json
from pathlib import Path
import cv2
from poseguard.annotations import file_sha256
from poseguard.image_io import write_image


def source_labels(root):
    labels = {}
    for filename in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        with (root / filename).open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                sequence, frame, label = row[0], int(row[1]), int(row[2])
                if label not in {-1, 0, 1}:
                    raise ValueError("Unknown official posture code")
                labels.setdefault(sequence, {})[frame] = label
    return labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/urfall")
    p.add_argument("--output", default="data/urfall-review")
    p.add_argument("--dense", action="store_true", help="Export every RGB frame for complete evaluation coverage")
    p.add_argument("--rgb-right-half", action="store_true", help="Use RGB half of official 640x240 depth/RGB preview")
    args = p.parse_args()
    root, out = Path(args.data), Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error("Choose an empty output directory")
    (out / "frames").mkdir(parents=True, exist_ok=True)
    labels = source_labels(root)
    tasks, coarse, events = [], [], []
    for sequence, frame_labels in sorted(labels.items()):
        video = root / f"{sequence}-cam0.mp4"
        if not video.exists():
            continue
        cap = cv2.VideoCapture(str(video))
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with (root / f"{sequence}-data.csv").open(newline="", encoding="utf-8") as stream:
            times = {int(r[0]): float(r[1]) / 1000 for r in csv.reader(stream)}
        for frame, label in sorted(frame_labels.items()):
            coarse.append(dict(video_id=video.stem, source_frame=frame, timestamp=times.get(frame),
                               official_label=label, label_scope="not_lying/transition/lying from depth; transition is not accidental-fall truth"))
        # Review every tenth frame plus both sides of each posture transition.
        selected = set(range(1, count + 1, 1 if args.dense else 10))
        previous = None
        for frame, label in sorted(frame_labels.items()):
            if previous != label:
                selected.update(range(max(1, frame - 2), min(count, frame + 2) + 1))
            previous = label
        for frame in sorted(selected):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
            ok, image = cap.read()
            if not ok:
                continue
            roi = None
            if args.rgb_right_half:
                if image.shape[:2] != (240,640):
                    raise ValueError("Unexpected URFD preview dimensions; inspect RGB layout before cropping")
                roi = [320,0,640,240]
                image = image[:,320:].copy()
            path = out / "frames" / f"{sequence}-{frame:05d}.jpg"
            write_image(path, image)
            tasks.append(dict(image=str(path.relative_to(out)), video_id=video.stem, source_frame=frame,
                              frame_id=frame-1, timestamp=(frame-1)/fps, source_timestamp=times.get(frame),
                              source_roi=roi,
                              person_id=None, bbox=None, label=None,
                              official_label_hint=frame_labels.get(frame), identity_verified=False,
                              annotation_source="pending", review_status="pending"))
        falling = [frame for frame, label in frame_labels.items()
                   if label == 0 and frame in times and sequence.startswith("fall-")]
        lying = [frame for frame, label in frame_labels.items() if label == 1 and frame in times]
        if falling:
            events.append(dict(video_id=video.stem, person_id="", start=times[min(falling)],
                               end=times[max(lying or falling)], source="official_depth_label_transition",
                               review_status="pending_RGB_and_identity_review"))
        cap.release()
    for filename, rows in (("review-tasks.jsonl", tasks), ("official-coarse-labels.jsonl", coarse)):
        (out / filename).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with (out / "events.provisional.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "person_id", "start", "end", "source", "review_status"])
        writer.writeheader()
        writer.writerows(events)
    manifest = dict(status="requires_human_review", tasks=len(tasks), coarse_labels=len(coarse),
                    dense_rgb_coverage=args.dense, timestamp_scope="RGB playback time; source_timestamp retains depth synchronization",
                    source_roi=[320,0,640,240] if args.rgb_right_half else None,
                    media_scope="RGB preview crop (320x240), not original full-resolution RGB PNG sequence" if args.rgb_right_half else "Depth/RGB composite preview",
                    source_hashes={f: file_sha256(root / f) for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")},
                    limitations=["Pending tasks cannot enter train/evaluate",
                                 "RGB/depth alignment and four-class labels require review",
                                 "Independent split requires verified cross-video identities",
                                 "Task completion requires human review even when dense frames have been exported"])
    (out / "review-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
