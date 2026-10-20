"""Manual evaluation annotations are independent of extracted model features."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .features import LABELS


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_annotations(path, *, require_image=False, allow_empty=False) -> list[dict]:
    """Accept hand-labelled boxes without requiring model-generated keypoints."""
    records, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("record_type") == "pose_features" or "annotation_bbox" in row:
                    raise ValueError("Extracted pose_features are not ground truth; use original manual annotations")
                if row.get("annotation_source", "manual") != "manual":
                    raise ValueError("Evaluation annotations must be manual, not pending or model labels")
                required = ("video_id", "person_id", "timestamp", "bbox", "label")
                for key in (*required, *(("image",) if require_image else ())):
                    if key not in row:
                        raise ValueError(f"Missing {key}")
                for key in ("video_id", "person_id"):
                    if row[key] is None or not str(row[key]).strip():
                        raise ValueError(f"Empty {key}")
                    row[key] = str(row[key])
                row["timestamp"] = float(row["timestamp"])
                if not np.isfinite(row["timestamp"]) or row["timestamp"] < 0:
                    raise ValueError("Invalid timestamp")
                if row["label"] not in LABELS:
                    raise ValueError(f"Unknown label {row['label']}")
                box = np.asarray(row["bbox"], dtype=float)
                if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                    raise ValueError("bbox must be finite xyxy with positive dimensions")
                if require_image and (not isinstance(row["image"], str) or not row["image"]):
                    raise ValueError("Invalid image path")
                key = (row["video_id"], row["person_id"], row["timestamp"])
                if key in seen:
                    raise ValueError("Duplicate video/person/timestamp in annotations")
                seen.add(key)
                records.append(row)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not records and not allow_empty:
        raise ValueError("Annotations are empty")
    return records


def load_truth_events(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"video_id", "person_id", "start", "end"} <= set(reader.fieldnames or []):
            raise ValueError("Event CSV requires video_id,person_id,start,end")
        rows = list(reader)
    seen = set()
    for row in rows:
        row["start"], row["end"] = float(row["start"]), float(row["end"])
        if (not row["video_id"] or not row["person_id"] or
                not np.isfinite([row["start"], row["end"]]).all() or row["start"] < 0 or row["end"] < row["start"]):
            raise ValueError("Invalid fall event identity or interval")
        key = (row["video_id"], row["person_id"], row["start"], row["end"])
        if key in seen:
            raise ValueError("Duplicate fall event")
        seen.add(key)
    return rows


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
