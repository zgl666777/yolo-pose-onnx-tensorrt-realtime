from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .features import LABELS, extract_features, predict_feature_labels, MIN_VISIBLE_RATIO
from .types import Detection
from .annotations import load_annotations, write_jsonl, file_sha256
from .image_io import read_image


def load_records(path: str) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                for key in ("video_id", "person_id", "timestamp", "bbox", "keypoints", "label"):
                    if key not in row:
                        raise ValueError(f"Missing {key}")
                if not str(row["video_id"]) or not str(row["person_id"]):
                    raise ValueError("Empty video/person ID")
                if row["label"] not in LABELS:
                    raise ValueError(f"Unknown label {row['label']}")
                if not np.isfinite(float(row["timestamp"])) or float(row["timestamp"]) < 0:
                    raise ValueError("Invalid timestamp")
                det = Detection(np.asarray(row["bbox"], dtype=np.float32), 1,
                                np.asarray(row["keypoints"], dtype=np.float32))
                extract_features(det)
                if np.any((det.keypoints[:, 2] < 0) | (det.keypoints[:, 2] > 1)):
                    raise ValueError("Keypoint confidence must be in [0,1]")
                records.append(row)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError("Dataset is empty")
    keys = [(str(r["video_id"]), str(r["person_id"]), float(r["timestamp"])) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate video/person/timestamp in dataset")
    return records


def connected_groups(records: list[dict]) -> np.ndarray:
    # A shared video OR a shared person keeps the entire connected component together.
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        root = key
        while parents[root] != root:
            root = parents[root]
        while parents[key] != key:
            parent = parents[key]
            parents[key] = root
            key = parent
        return root
    def union(a, b):
        parents[find(b)] = find(a)
    for row in records:
        union(("video", str(row["video_id"])), ("person", str(row["person_id"])))
    roots = [find(("video", str(row["video_id"]))) for row in records]
    lookup = {root: i for i, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([lookup[root] for root in roots])


def split_records(records, seed=42):
    groups = connected_groups(records)
    if len(np.unique(groups)) < 5:
        raise ValueError("Need at least 5 independent person/video components for train/val/test")
    indices = np.arange(len(records))
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                           .split(indices, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
                  .split(train_val, groups=groups[train_val]))
    partitions = {"train": train_val[tr], "validation": train_val[va], "test": test}
    for key, part in partitions.items():
        if not len(part):
            raise ValueError(f"Empty {key} split")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for key in ("person_id", "video_id"):
            if {str(records[i][key]) for i in partitions[a]} & {str(records[i][key]) for i in partitions[b]}:
                raise AssertionError("Data leakage detected")
    return partitions


def prepare_splits(annotations, output, seed=42, events=None, frames=None):
    """Freeze person/video partitions from complete human truth before extraction."""
    from .annotations import load_truth_events
    import csv
    rows = load_annotations(annotations)
    frame_ledger = None
    if frames:
        from .evaluation import load_frame_manifest, _validate_frame_coverage
        frame_ledger = load_frame_manifest(frames, "person_count")
        grouped = {}
        for row in rows:
            grouped.setdefault((row["video_id"],row["timestamp"]), []).append(row)
        _validate_frame_coverage(grouped, frame_ledger)
    empty_assignments = {}
    if frame_ledger is not None:
        empty_videos = sorted({v for v,_ in frame_ledger} - {r["video_id"] for r in rows})
        generator = np.random.default_rng(seed)
        empty_assignments = {v:str(generator.choice(["train","validation","test"],p=[.6,.2,.2])) for v in empty_videos}
    if any(r.get("identity_verified") is not True or r.get("annotation_source") != "manual" for r in rows):
        raise ValueError("Every annotation needs identity_verified=true and annotation_source=manual; resolve cross-video identities first")
    parts = split_records(rows, seed)
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Split directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    falls = load_truth_events(events) if events else []
    known = {(r["video_id"], r["person_id"]) for r in rows}
    if any((r["video_id"], r["person_id"]) not in known for r in falls):
        raise ValueError("Event identity is absent from manual annotations")
    manifest = dict(seed=seed, source_sha256=file_sha256(annotations), identity_verified=True,
                    frames_sha256=file_sha256(frames) if frames else None,
                    empty_video_assignments=empty_assignments,
                    split_unit="person/video connected components", splits={})
    for name, indices in parts.items():
        selected = [dict(rows[i]) for i in indices]
        for row in selected:
            if row.get("image"):
                row["image"] = str((Path(annotations).resolve().parent / row["image"]).resolve())
        path = root / f"{name}.ground-truth.jsonl"
        write_jsonl(path, selected)
        videos = {r["video_id"] for r in selected}
        videos.update(v for v,split in empty_assignments.items() if split == name)
        event_path = root / f"{name}.events.csv"
        with event_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id", "person_id", "start", "end"])
            writer.writeheader()
            writer.writerows({k: r[k] for k in writer.fieldnames} for r in falls if r["video_id"] in videos)
        manifest["splits"][name] = dict(path=path.name, sha256=file_sha256(path), events=event_path.name,
                                        events_sha256=file_sha256(event_path), samples=len(selected))
        if frame_ledger is not None:
            frame_path = root / f"{name}.frames.jsonl"
            write_jsonl(frame_path, [dict(r,video_id=v,timestamp=t,person_count=r["count"])
                                    for (v,t),r in frame_ledger.items() if v in videos])
            manifest["splits"][name].update(frames=frame_path.name,frames_sha256=file_sha256(frame_path))
    (root / "split-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _frozen_parts(records, manifest_path):
    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("identity_verified") is not True:
        raise ValueError("Frozen manifest requires verified identities")
    lookup, allowed_hashes = {}, {manifest["source_sha256"]}
    for name, info in manifest["splits"].items():
        path = source.parent / info["path"]
        if file_sha256(path) != info["sha256"]:
            raise ValueError(f"Frozen {name} annotations changed")
        allowed_hashes.add(info["sha256"])
        for row in load_annotations(path):
            if row.get("identity_verified") is not True or row.get("annotation_source") != "manual":
                raise ValueError("Frozen truth must contain verified manual annotations")
            key = (row["video_id"], row["person_id"], float(row["timestamp"]))
            if key in lookup:
                raise ValueError("Overlapping frozen split assignments")
            lookup[key] = (name, row)
    parts = {name: [] for name in ("train", "validation", "test")}
    for i, row in enumerate(records):
        key = (str(row["video_id"]), str(row["person_id"]), float(row["timestamp"]))
        if key not in lookup or lookup[key][1]["label"] != row["label"]:
            raise ValueError("Features do not agree with frozen manual annotations")
        if (row.get("record_type") != "pose_features" or row.get("annotations_sha256") not in allowed_hashes or
                not np.array_equal(row.get("annotation_bbox"), lookup[key][1]["bbox"])):
            raise ValueError("Feature provenance or original manual bbox differs from frozen truth")
        parts[lookup[key][0]].append(i)
    for name, indices in parts.items():
        if not indices:
            raise ValueError(f"No extracted features for {name}")
    # Verify identity isolation even when a supplied manifest was edited or assembled externally.
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for key in ("person_id", "video_id"):
            if {str(records[i][key]) for i in parts[a]} & {str(records[i][key]) for i in parts[b]}:
                raise ValueError("Frozen split leaks person/video identities")
    return {k: np.asarray(v) for k, v in parts.items()}


def train_classifier(dataset, output, report_dir, seed=42, threshold=0.3, split_manifest=None):
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from .features import FEATURE_NAMES, FEATURE_VERSION
    records = load_records(dataset)
    if not all(r.get("synthetic", False) for r in records) and split_manifest is None:
        raise ValueError("Real training requires --split-manifest frozen from verified manual annotations")
    parts = _frozen_parts(records, split_manifest) if split_manifest else split_records(records, seed)
    destination, reports = Path(output), Path(report_dir)
    if destination.exists() or (reports.exists() and any(reports.iterdir())):
        raise FileExistsError("Choose a new model and report directory for a new training experiment")
    # Holdout features are validated and saved, but never fitted, predicted or scored here.
    def xy(indices):
        selected = [records[i] for i in indices]
        return (np.stack([extract_features(Detection(np.asarray(r["bbox"]), 1, np.asarray(r["keypoints"])), threshold)
                          for r in selected]), np.asarray([r["label"] for r in selected]))
    train_x, train_y = xy(parts["train"])
    val_x, val_y = xy(parts["validation"])
    if len(set(train_y)) < 2:
        raise ValueError("Training needs at least two classes in independent subjects/videos")
    if set(val_y) - set(train_y):
        raise ValueError("Validation includes classes absent from training; collect more data")
    trials, models = [], []
    for c in (0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced",
                                                                   random_state=seed))
        model.fit(train_x, train_y)
        score = f1_score(val_y, predict_feature_labels(model, val_x), labels=list(LABELS), average="macro", zero_division=0)
        trials.append(dict(C=c, validation_macro_f1=float(score)))
        models.append(model)
    best = max(range(len(trials)), key=lambda i: trials[i]["validation_macro_f1"])
    model = models[best]
    destination.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=model, feature_version=FEATURE_VERSION, features=FEATURE_NAMES,
                     keypoint_threshold=threshold, labels=LABELS), destination)
    result = dict(seed=seed, feature_version=FEATURE_VERSION, split_unit="person/video connected components",
                  synthetic=any(r.get("synthetic", False) for r in records),
                  dataset=str(dataset), dataset_sha256=file_sha256(dataset), model=str(destination.resolve()),
                  model_sha256=file_sha256(destination), split_manifest_sha256=file_sha256(split_manifest) if split_manifest else None,
                  component_count=int(len(set(connected_groups(records)))), labels=LABELS, splits={},
                  quality_gate=dict(min_visible_ratio=MIN_VISIBLE_RATIO, rejected_label="unknown"),
                  four_class_training_coverage=set(LABELS) <= set(train_y),
                  selection=dict(trials=trials, selected_C=trials[best]["C"], scope="validation only; no test scoring"))
    for name, indices in parts.items():
        info = dict(samples=len(indices), person_count=len({str(records[i]["person_id"]) for i in indices}),
                    evaluated=name != "test")
        if name != "test":
            x, y = (train_x, train_y) if name == "train" else (val_x, val_y)
            prediction = predict_feature_labels(model, x)
            info.update(metrics=classification_report(y, prediction, labels=list(LABELS), output_dict=True, zero_division=0),
                        rejected_samples=int(np.sum(prediction == "unknown")),
                        confusion_labels=[*LABELS, "unknown"],
                        confusion_matrix=confusion_matrix(y, prediction, labels=[*LABELS, "unknown"]).tolist())
        path = reports / f"{name}.jsonl"
        write_jsonl(path, [records[i] for i in indices])
        info["sha256"] = file_sha256(path)
        result["splits"][name] = info
    (reports / "training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def score_classifier(report_dir, output):
    """Explicit final holdout scoring of the already frozen model and test features."""
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix
    root, destination = Path(report_dir), Path(output)
    if destination.exists():
        raise FileExistsError("Final report already exists; preserve it")
    training = json.loads((root / "training.json").read_text(encoding="utf-8"))
    test, model_path = root / "test.jsonl", Path(training["model"])
    if file_sha256(test) != training["splits"]["test"]["sha256"] or file_sha256(model_path) != training["model_sha256"]:
        raise ValueError("Frozen test data or model changed after tuning")
    payload = joblib.load(model_path)
    records = load_records(test)
    x = np.stack([extract_features(Detection(np.asarray(r["bbox"]), 1, np.asarray(r["keypoints"])), payload["keypoint_threshold"])
                  for r in records])
    y, prediction = [r["label"] for r in records], predict_feature_labels(payload["model"], x)
    result = dict(synthetic=training["synthetic"], model_sha256=training["model_sha256"], test_sha256=file_sha256(test),
                  scope="Matched extracted poses only; full detection/event metrics require evaluate with manual truth",
                  classification=classification_report(y, prediction, labels=list(LABELS), output_dict=True, zero_division=0),
                  rejected_samples=int(np.sum(prediction == "unknown")),
                  four_class_test_coverage=set(LABELS) <= set(y),
                  confusion_labels=[*LABELS, "unknown"],
                  confusion_matrix=confusion_matrix(y, prediction, labels=[*LABELS, "unknown"]).tolist())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def extract_dataset(annotations, config, output, ground_truth_output=None):
    """Use annotated per-frame person boxes to select poses without assigning pseudo-labels."""
    import cv2
    from .backends import make_backend
    from .geometry import preprocess, decode, box_iou
    from scipy.optimize import linear_sum_assignment
    annotation_path = Path(annotations)
    rows = load_annotations(annotation_path, require_image=True)
    destination = Path(output)
    truth_path = Path(ground_truth_output) if ground_truth_output else destination.with_suffix(".ground-truth.jsonl")
    audit_path = destination.with_suffix(".extraction.json")
    paths = [annotation_path.resolve(), destination.resolve(), truth_path.resolve(), audit_path.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("Annotations, features, ground truth and audit must use separate files")
    for path in (destination, truth_path, audit_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    annotation_hash = file_sha256(annotation_path)
    # Save every manual instance before inference, including every later miss.
    write_jsonl(truth_path, [dict(r, record_type="manual_annotation",
                                 image=str((annotation_path.resolve().parent / r["image"]).resolve())) for r in rows])
    frames = {}
    for row in rows:
        for key in ("image", "video_id", "person_id", "timestamp", "bbox", "label"):
            if key not in row:
                raise ValueError(f"Annotation missing {key}")
        if row["label"] not in LABELS:
            raise ValueError(f"Invalid annotation label {row['label']}")
        frames.setdefault(row["image"], []).append(row)
    backend = None
    accepted, skipped = 0, 0
    audit = dict(status="running", annotations_sha256=annotation_hash, ground_truth=str(truth_path),
                 annotation_instances=len(rows), output=str(destination), rejected=[])
    try:
        backend = make_backend(config)
        with destination.open("w", encoding="utf-8") as file:
            for path, annotations_for_frame in frames.items():
                image = read_image(annotation_path.parent / path)
                if image is None:
                    raise ValueError(f"Unreadable annotated image {path}")
                tensor, meta = preprocess(image, config.imgsz)
                detections = decode(backend.infer(tensor).output, meta, config.confidence, config.iou)
                if not detections:
                    skipped += len(annotations_for_frame)
                    audit["rejected"].extend(dict(video_id=r["video_id"], person_id=r["person_id"],
                                                  timestamp=r["timestamp"], reason="no_detection") for r in annotations_for_frame)
                    continue
                overlaps = box_iou(np.asarray([r["bbox"] for r in annotations_for_frame]),
                                   np.stack([d.bbox for d in detections]))
                ri, ci = linear_sum_assignment(np.where(overlaps >= 0.5, 1 - overlaps, 1e6))
                matches = {int(i): int(j) for i, j in zip(ri, ci) if overlaps[i, j] >= 0.5}
                for i, row in enumerate(annotations_for_frame):
                    if i not in matches:
                        skipped += 1
                        audit["rejected"].append(dict(video_id=row["video_id"], person_id=row["person_id"],
                                                       timestamp=row["timestamp"], reason="no_box_match"))
                        continue
                    det = detections[matches[i]]
                    row = dict(row, record_type="pose_features", annotation_bbox=list(row["bbox"]),
                               image=str((annotation_path.resolve().parent / row["image"]).resolve()),
                               annotations_sha256=annotation_hash, bbox=det.bbox.tolist(),
                               keypoints=det.keypoints.tolist(), match_iou=float(overlaps[i, matches[i]]))
                    file.write(json.dumps(row) + "\n")
                    accepted += 1
        if accepted == 0:
            raise ValueError("No annotated boxes matched pose detections")
        load_records(str(destination))
        audit["status"] = "completed"
        return dict(accepted=accepted, skipped=skipped, output=str(destination),
                    ground_truth=str(truth_path), audit=str(audit_path))
    except Exception as exc:
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        audit.update(accepted=accepted, skipped=skipped)
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        if backend is not None:
            backend.close()
