"""Transactional human review storage; pending tasks never become evaluation truth."""
from __future__ import annotations
import csv
import json
import sqlite3
from pathlib import Path
import numpy as np
from .features import LABELS
from .annotations import write_jsonl, file_sha256, load_annotations, load_truth_events
from .image_io import read_image


def _task_image_path(task, image_root=None):
    path = Path(task["image"])
    if not path.is_absolute() and image_root is not None:
        path = Path(image_root) / path
    return path.resolve()


def _task_image_hash(task, image_root=None):
    path = _task_image_path(task, image_root)
    if not path.is_file():
        raise ValueError(f"Review image does not exist: {path}")
    if read_image(path) is None:
        raise ValueError(f"Review image is unreadable: {path}")
    return file_sha256(path)


class ReviewStore:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS frames(video TEXT, frame INTEGER, payload TEXT, PRIMARY KEY(video,frame));
            CREATE TABLE IF NOT EXISTS events(video TEXT, person TEXT, start REAL, end REAL, reviewer TEXT,
                                             PRIMARY KEY(video,person,start));
        """)

    def get(self, task):
        frame_id = task["frame_id"] if "frame_id" in task else task["source_frame"]-1
        row = self.connection.execute("SELECT payload FROM frames WHERE video=? AND frame=?",
                                      (task["video_id"], frame_id)).fetchone()
        return json.loads(row[0]) if row else None

    def audit(self, tasks_path):
        """Return a machine-readable completion gate for a review queue.

        A queue is ready only when every task has been saved, every annotated
        person has a valid box/posture and verified cross-video identity
        evidence, and every falling annotation is covered by a reviewed event.
        This audit never promotes draft rows to ground truth.
        """
        task_path = Path(tasks_path).resolve()
        tasks = [json.loads(s) for s in task_path.read_text(encoding="utf-8").splitlines() if s.strip()]
        if not tasks:
            raise ValueError("Review tasks are empty")
        missing, invalid, falling_without_event = [], [], []
        changed = []
        seen_tasks = set()
        reviewed = 0
        people = 0
        verified = 0
        known_people = set()
        events = [dict(zip(("video_id", "person_id", "start", "end", "reviewer"), r)) for r in
                  self.connection.execute("SELECT video,person,start,end,reviewer FROM events")]
        for task in tasks:
            task_id = (task["video_id"], task["frame_id"] if "frame_id" in task else task["source_frame"] - 1)
            if task_id in seen_tasks:
                raise ValueError("Duplicate video/frame in review tasks")
            seen_tasks.add(task_id)
            frame = self.get(task)
            if frame is None:
                missing.append({"video_id": task.get("video_id"),
                                "frame_id": task.get("frame_id", task.get("source_frame"))})
                continue
            reviewed += 1
            try:
                image_hash = _task_image_hash(task, task_path.parent)
            except (ValueError, OSError):
                image_hash = None
            if (float(frame["timestamp"]) != float(task["timestamp"]) or frame["image"] != task["image"] or
                    frame.get("source_roi") != task.get("source_roi") or
                    image_hash is None or not frame.get("image_sha256") or
                    frame.get("image_sha256") != image_hash):
                changed.append({"video_id": task_id[0], "frame_id": task_id[1],
                                "reason": "saved frame no longer matches review task"})
            if not str(frame.get("reviewer", "")).strip():
                invalid.append({"video_id": task_id[0], "frame_id": task_id[1], "reason": "reviewer is required"})
            seen_people = set()
            for person in frame.get("people", []):
                people += 1
                try:
                    box = np.asarray(person.get("bbox"), dtype=float)
                    valid = (box.shape == (4,) and np.isfinite(box).all() and
                             np.all(box[2:] > box[:2]) and str(person.get("person_id", "")).strip() and
                             person.get("label") in LABELS and
                             person.get("person_id") not in seen_people and
                             person.get("identity_verified") is True and
                             str(person.get("identity_evidence", "")).strip())
                except (TypeError, ValueError):
                    valid = False
                seen_people.add(str(person.get("person_id", "")))
                known_people.add((frame.get("video_id"), str(person.get("person_id", ""))))
                if person.get("identity_verified") is True and str(person.get("identity_evidence", "")).strip():
                    verified += 1
                if not valid:
                    invalid.append({"video_id": frame.get("video_id"), "frame_id": frame.get("frame_id"),
                                    "person_id": person.get("person_id", ""),
                                    "reason": "requires bbox, four-class posture, verified identity and evidence"})
                if person.get("label") == "falling":
                    pid = str(person.get("person_id", ""))
                    ts = float(frame.get("timestamp", -1))
                    if not any(e["video_id"] == frame.get("video_id") and e["person_id"] == pid and
                               float(e["start"]) <= ts <= float(e["end"]) for e in events):
                        falling_without_event.append({"video_id": frame.get("video_id"),
                                                      "frame_id": frame.get("frame_id"), "person_id": pid})
        blockers = []
        if missing:
            blockers.append("Every task, including empty scenes, must be manually completed")
        if invalid:
            blockers.append("Every annotated person needs a valid RGB bbox, four-class posture, verified identity and evidence")
        if falling_without_event:
            blockers.append("Every falling frame needs a manually reviewed event interval")
        if changed:
            blockers.append("Saved reviews must match the current task image, timestamp and crop")
        orphan_events = [e for e in events if (e["video_id"], e["person_id"]) not in known_people]
        if orphan_events:
            blockers.append("Every reviewed event identity must appear in completed annotations")
        return {"status": "ready" if not blockers else "blocked", "ready_for_export": not blockers,
                "tasks": len(tasks), "reviewed_tasks": reviewed, "remaining_tasks": len(missing),
                "people_instances": people, "identity_verified_people": verified,
                "missing_tasks": missing[:200], "missing_tasks_truncated": len(missing) > 200,
                "invalid_people": invalid[:200], "invalid_people_truncated": len(invalid) > 200,
                "falling_without_event": falling_without_event[:200],
                "falling_without_event_truncated": len(falling_without_event) > 200,
                "changed_tasks": changed[:200], "changed_tasks_truncated": len(changed) > 200,
                "orphan_events": orphan_events[:200], "orphan_events_truncated": len(orphan_events) > 200,
                "blockers": blockers,
                "tasks_sha256": file_sha256(task_path)}

    def save(self, task, people, reviewer, *, image_root=None, expected_image_sha256=None):
        if not reviewer.strip():
            raise ValueError("Reviewer is required")
        seen = set()
        for person in people:
            box = np.asarray(person["bbox"], dtype=float)
            identity = str(person["person_id"]).strip()
            if (not identity or identity in seen or person["label"] not in LABELS or box.shape != (4,) or
                    not np.isfinite(box).all() or np.any(box[2:] <= box[:2])):
                raise ValueError("Each person needs a unique identity, valid box and posture")
            if person.get("identity_verified") is not True or not str(person.get("identity_evidence", "")).strip():
                raise ValueError("Verify cross-video identity and record evidence before completing a frame")
            seen.add(identity)
        frame_id = task["frame_id"] if "frame_id" in task else task["source_frame"]-1
        image_hash = _task_image_hash(task, image_root)
        if expected_image_sha256 is not None and image_hash != expected_image_sha256:
            raise ValueError("Review image changed since display; reload and review the image again")
        payload = dict(video_id=task["video_id"], frame_id=frame_id, timestamp=task["timestamp"],
                       source_roi=task.get("source_roi"),
                       image=task["image"], image_sha256=image_hash,
                       people=people, reviewer=reviewer, annotation_source="manual")
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO frames VALUES (?,?,?)",
                                     (task["video_id"], frame_id, json.dumps(payload)))

    def save_event(self, video, person, start, end, reviewer):
        if not video or not person or not reviewer.strip() or not np.isfinite([start,end]).all() or start < 0 or end < start:
            raise ValueError("Invalid manually reviewed event")
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?)", (video,person,start,end,reviewer))

    def export(self, tasks_path, output):
        destination = Path(output)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("Use a new export directory to preserve previous reviewed truth")
        task_path = Path(tasks_path).resolve()
        audit = self.audit(task_path)
        if not audit["ready_for_export"]:
            if audit["missing_tasks"]:
                raise ValueError("Every task, including empty scenes, must be manually completed before export")
            if audit["invalid_people"]:
                raise ValueError("Every person needs a unique identity, valid box, posture and identity evidence")
            if audit["changed_tasks"]:
                raise ValueError("Saved reviews do not match the current tasks; review changed images, timestamps or crops again")
            if audit["orphan_events"]:
                raise ValueError("Event identity is absent from completed annotations")
            raise ValueError("Every reviewed falling frame needs a manually reviewed event interval")
        tasks = [json.loads(s) for s in task_path.read_text(encoding="utf-8").splitlines() if s.strip()]
        rows, frames = [], []
        for task in tasks:
            frame = self.get(task)
            if frame is None:
                raise ValueError("Every task, including empty scenes, must be manually completed before export")
            image = str(_task_image_path(task, task_path.parent))
            if _task_image_hash(task, task_path.parent) != frame["image_sha256"]:
                raise ValueError("Review image changed during export; review this task again")
            frames.append(dict(video_id=frame["video_id"], frame_id=frame["frame_id"], timestamp=frame["timestamp"],
                               source_roi=frame.get("source_roi"),
                               image_sha256=frame["image_sha256"],
                               person_count=len(frame["people"]), reviewer=frame["reviewer"]))
            for person in frame["people"]:
                rows.append(dict(person, video_id=frame["video_id"], frame_id=frame["frame_id"], timestamp=frame["timestamp"],
                                 source_roi=frame.get("source_roi"),
                                 image=image, image_sha256=frame["image_sha256"],
                                 reviewer=frame["reviewer"], annotation_source="manual"))
        events = [dict(zip(("video_id","person_id","start","end","reviewer"), r)) for r in
                  self.connection.execute("SELECT video,person,start,end,reviewer FROM events ORDER BY video,start")]
        known = {(r["video_id"],r["person_id"]) for r in rows}
        if any((e["video_id"],e["person_id"]) not in known for e in events):
            raise ValueError("Event identity is absent from completed annotations")
        if any(r["label"] == "falling" and not any(e["video_id"] == r["video_id"] and e["person_id"] == r["person_id"] and
                e["start"] <= r["timestamp"] <= e["end"] for e in events) for r in rows):
            raise ValueError("Every reviewed falling frame needs a manually reviewed event interval")
        from .evaluation import load_frame_manifest, _validate_frame_coverage
        destination.mkdir(parents=True, exist_ok=True)
        write_jsonl(destination / "manual.jsonl", rows)
        write_jsonl(destination / "frames.jsonl", frames)
        with (destination / "events.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id","person_id","start","end","reviewer"])
            writer.writeheader()
            writer.writerows(events)
        load_annotations(destination / "manual.jsonl", allow_empty=True)
        load_truth_events(destination / "events.csv")
        ledger = load_frame_manifest(destination / "frames.jsonl", "person_count")
        groups = {}
        for row in rows:
            groups.setdefault((row["video_id"],float(row["timestamp"])), []).append(row)
        _validate_frame_coverage(groups, ledger)
        manifest = dict(status="human_reviewed_export", authoritative=True, acceptance=audit,
                        tasks_sha256=file_sha256(tasks_path), frames=len(frames),
                        people_instances=len(rows), events=len(events),
                        files={p.name:file_sha256(p) for p in destination.iterdir() if p.is_file()},
                        limitation="Export completeness covers selected tasks; use dense RGB tasks for full-video evaluation")
        (destination / "review-manifest.json").write_text(json.dumps(manifest,indent=2), encoding="utf-8")
        return manifest
