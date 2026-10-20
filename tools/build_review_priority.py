"""Prioritize exploratory error windows without turning them into truth."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from poseguard.annotations import file_sha256, write_jsonl


def build_queue(tasks_path, diagnostics_path, output, before=1.2, after=1.5, stride=3):
    if (not math.isfinite(before) or not math.isfinite(after) or before < 0 or after < 0 or
            not isinstance(stride, int) or isinstance(stride, bool) or stride < 1):
        raise ValueError("Window seconds must be non-negative and stride positive")
    tasks_path, diagnostics_path, output = Path(tasks_path).resolve(), Path(diagnostics_path).resolve(), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new priority directory")
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    tasks_hash, diagnostics_hash = file_sha256(tasks_path), file_sha256(diagnostics_path)
    by_video = {}
    for row in tasks:
        by_video.setdefault(row["video_id"], []).append(row)
    queue, counts = [], Counter()
    for error in diagnostics["videos"]:
        if not error.get("fp") and not error.get("fn"):
            continue
        video = error["video_id"]
        center = error["alarms"][0] if error.get("fp") else error["onset"]
        if center is None:
            raise ValueError(f"Error window needs a center timestamp: {video}")
        available = by_video.get(video, [])
        if not available:
            raise ValueError(f"No source draft tasks for {video}")
        # RGB playback and depth sync are retained separately in the source
        # draft. Use source timestamps for matching the diagnostic window.
        window = [r for r in available if r.get("source_timestamp") is not None and
                  max(0, center - before) <= float(r["source_timestamp"]) <= center + after]
        anchors = {int(r["frame_id"]) for r in error.get("selected_frames", [])}
        chosen_by_frame = {int(r["frame_id"]): r for index, r in enumerate(window) if index % stride == 0}
        chosen_by_frame.update({int(r["frame_id"]): r for r in available if int(r["frame_id"]) in anchors})
        chosen = list(chosen_by_frame.values())
        if not chosen:
            raise ValueError(f"No source timestamps cover the error window: {video}")
        kind = "false_alarm" if error.get("fp") else "missed_fall_sequence"
        for source in sorted(chosen, key=lambda r: int(r["frame_id"])):
            row = dict(source)
            row["image"] = str((tasks_path.parent / source["image"]).resolve())
            row.update(bbox=None, person_id="", label=None, identity_verified=False,
                       annotation_source="draft", review_status="pending", authoritative=False,
                       human_review_pending=True,
                       priority_reason=kind,
                       priority_context={"center_source_timestamp": center, "before_seconds": before,
                                         "after_seconds": after,
                                         "source_task_id": {"video_id": video, "frame_id": source["frame_id"]},
                                         "source_tasks_sha256": tasks_hash, "diagnostics_sha256": diagnostics_hash,
                                         "scope": "exploratory triage; event label and identity require human review"})
            queue.append(row)
            counts[video] += 1
    output.mkdir(parents=True)
    queue_path = output / "review-priority.jsonl"
    write_jsonl(queue_path, queue)
    manifest = {"schema_version": 1, "status": "priority_draft_requires_human_review",
                "authoritative": False, "human_review_pending": True, "final_acceptance_complete": False,
                "tasks": len(queue), "videos": len(counts), "tasks_per_video": dict(counts),
                "source_tasks": str(tasks_path), "source_tasks_sha256": tasks_hash,
                "diagnostics": str(diagnostics_path), "diagnostics_sha256": diagnostics_hash,
                "queue_sha256": file_sha256(queue_path), "code_sha256": file_sha256(__file__),
                "sampling": {"before_seconds": before, "after_seconds": after, "frame_stride": stride,
                             "contact_sheet_anchor_frames_always_included": True},
                "limitations": ["Sparse error windows are not complete video truth or independent evaluation data",
                                "Priority reasons are exploratory diagnostics, not human posture or event labels",
                                "Model box/posture suggestions remain suggestions; missing people must be added",
                                "Video-local track candidates cannot verify cross-video human identity"]}
    (output / "priority-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="data/urfall-review-draft-suggestions/draft-review.jsonl")
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--before", type=float, default=1.2)
    parser.add_argument("--after", type=float, default=1.5)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    manifest = build_queue(args.tasks, args.diagnostics, args.output, args.before, args.after, args.stride)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
