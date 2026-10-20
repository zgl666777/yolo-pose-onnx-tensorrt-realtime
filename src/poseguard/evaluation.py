from __future__ import annotations

import csv
import json
from bisect import bisect_left, bisect_right
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.metrics import classification_report, confusion_matrix
from .annotations import load_annotations, load_truth_events, file_sha256
from .geometry import box_iou
from .features import LABELS


def load_frame_manifest(path, count_field):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream)) if str(path).endswith(".csv") else [json.loads(s) for s in stream if s.strip()]
    result, ids = {}, set()
    for row in rows:
        video, timestamp = str(row["video_id"]), float(row["timestamp"])
        frame_id, count = int(row["frame_id"]), int(row[count_field])
        key = (video, timestamp)
        if (not video.strip() or not np.isfinite(timestamp) or timestamp < 0 or frame_id < 0 or count < 0 or
                key in result or (video, frame_id) in ids):
            raise ValueError("Invalid or duplicate frame manifest entry")
        result[key] = dict(row, frame_id=frame_id, count=count)
        ids.add((video, frame_id))
    if not result:
        raise ValueError("Frame manifest is empty")
    return result


def _validate_frame_coverage(instances, ledger):
    for key, rows in instances.items():
        if key not in ledger:
            raise ValueError(f"Instance is absent from frame manifest: {key}")
        if any("frame_id" in r and int(r["frame_id"]) != ledger[key]["frame_id"] for r in rows):
            raise ValueError("Frame ID conflicts with instance timestamp")
    for key, frame in ledger.items():
        if len(instances.get(key, [])) != frame["count"]:
            raise ValueError(f"Frame person count differs from complete annotations/predictions: {key}")


def associate_frames(gt_frames, pd_frames, tolerance, gt_ledger=None, pd_ledger=None):
    """Reserve exact IDs/times, then maximize sparse one-to-one temporal coverage."""
    assignment, used = {}, set()
    if gt_ledger is not None and pd_ledger is not None:
        by_id = {(v, r["frame_id"]): (v, t) for (v, t), r in pd_ledger.items()}
        for key, row in gt_ledger.items():
            candidate = by_id.get((key[0], row["frame_id"]))
            if candidate is not None:
                if abs(candidate[1] - key[1]) > tolerance + 1e-9:
                    raise ValueError("Equal frame IDs have conflicting timestamps")
                assignment[key] = candidate
                used.add(candidate)
        # Different explicit IDs represent different frames, even if their times are near.
        return assignment, used
    for key in gt_frames:
        if key in pd_frames:
            assignment[key] = key
            used.add(key)
    pred_times = {}
    for video, timestamp in pd_frames:
        if (video, timestamp) not in used:
            pred_times.setdefault(video, []).append(timestamp)
    for timestamps in pred_times.values():
        timestamps.sort()
    remaining = [key for key in gt_frames if key not in assignment]
    candidates = [key for key in pd_frames if key not in used]
    candidate_index = {key: i for i, key in enumerate(candidates)}
    row_indices, col_indices, weights = [], [], []
    for i, key in enumerate(remaining):
        video, timestamp = key
        timestamps = pred_times.get(video, [])
        lo, hi = bisect_left(timestamps, timestamp - tolerance), bisect_right(timestamps, timestamp + tolerance)
        for t in timestamps[lo:hi]:
            row_indices.append(i)
            col_indices.append(candidate_index[(video, t)])
            weights.append(1 + abs(t-timestamp) / max(tolerance, 1e-12))
        row_indices.append(i)
        col_indices.append(len(candidates) + i)
        weights.append(2 * len(remaining) + 1)
    if remaining:
        costs = coo_matrix((weights, (row_indices, col_indices)),
                           shape=(len(remaining), len(candidates)+len(remaining))).tocsr()
        rows, cols = min_weight_full_bipartite_matching(costs)
        for i, j in zip(rows, cols):
            if j < len(candidates):
                assignment[remaining[i]] = candidates[j]
                used.add(candidates[j])
    return assignment, used


def evaluate_predictions(ground_truth, predictions, output, events=None, truth_events=None,
                         frame_tolerance=0.05, match_iou=0.5, max_alarm_delay=5.0,
                         ground_truth_frames=None, prediction_frames=None, deadline_scope="media", event_delivery=None):
    if bool(ground_truth_frames) != bool(prediction_frames):
        raise ValueError("Specify both ground truth and prediction frame manifests")
    truth = load_annotations(ground_truth, allow_empty=bool(ground_truth_frames))
    if not np.isfinite([frame_tolerance,match_iou,max_alarm_delay]).all() or frame_tolerance < 0 or not 0 < match_iou <= 1 or max_alarm_delay <= 0:
        raise ValueError("Invalid frame tolerance, IoU or alarm delay")
    pred = [json.loads(line) for line in Path(predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_predictions = set()
    for row in pred:
        for key in ("video_id", "track_id", "timestamp", "bbox", "label"):
            if key not in row:
                raise ValueError(f"Prediction missing {key}")
        box = np.asarray(row["bbox"], dtype=float)
        timestamp = float(row["timestamp"])
        if (row["label"] not in {*LABELS, "unknown"} or not str(row["video_id"]).strip() or
                row["track_id"] is None or not np.isfinite(timestamp) or timestamp < 0 or
                box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])):
            raise ValueError("Invalid prediction identity, label, timestamp or bbox")
        if "frame_id" in row and (int(row["frame_id"]) != row["frame_id"] or int(row["frame_id"]) < 0):
            raise ValueError("Invalid prediction frame ID")
        key = (str(row["video_id"]), timestamp, str(row["track_id"]))
        if key in seen_predictions:
            raise ValueError("Duplicate video/timestamp/track prediction")
        seen_predictions.add(key)
    gt_frames, pd_frames = {}, {}
    for row in truth:
        gt_frames.setdefault((str(row["video_id"]), float(row["timestamp"])), []).append(row)
    for row in pred:
        pd_frames.setdefault((str(row["video_id"]), float(row["timestamp"])), []).append(row)
    for frames in (gt_frames,pd_frames):
        frame_ids = set()
        for (video,_), rows in frames.items():
            ids = {int(r["frame_id"]) for r in rows if "frame_id" in r}
            if len(ids)>1 or any((video,i) in frame_ids for i in ids):
                raise ValueError("Conflicting frame IDs/timestamps")
            frame_ids.update((video,i) for i in ids)
    gt_ledger = load_frame_manifest(ground_truth_frames, "person_count") if ground_truth_frames else None
    pd_ledger = load_frame_manifest(prediction_frames, "people") if prediction_frames else None
    if gt_ledger is not None:
        _validate_frame_coverage(gt_frames, gt_ledger)
        _validate_frame_coverage(pd_frames, pd_ledger)
        for key in gt_ledger:
            gt_frames.setdefault(key, [])
        for key in pd_ledger:
            pd_frames.setdefault(key, [])
    assignment, used_frames = associate_frames(gt_frames, pd_frames, frame_tolerance, gt_ledger, pd_ledger)
    if pd_ledger is not None and set(pd_frames) - used_frames:
        raise ValueError("Processed frames lack complete manual coverage; annotate them before final evaluation")
    expected, actual, matches = [], [], []
    identity_votes = {}
    event_identity = {}
    histories = {}
    matched_ious = []
    unmatched_predictions = 0
    unmatched_labels = []
    for (video, timestamp), annotations in sorted(gt_frames.items()):
        associated = assignment.get((video, timestamp))
        nearest = associated[1] if associated else None
        candidates = pd_frames.get(associated, [])
        paired = {}
        if candidates and annotations:
            overlap = box_iou(np.asarray([r["bbox"] for r in annotations]),
                              np.asarray([r["bbox"] for r in candidates]))
            ri, ci = linear_sum_assignment(np.where(overlap >= match_iou, 1 - overlap, 1e6))
            paired = {int(i): int(j) for i, j in zip(ri, ci) if overlap[i, j] >= match_iou}
        for i, row in enumerate(annotations):
            expected.append(row["label"])
            candidate = candidates[paired[i]] if i in paired else None
            histories.setdefault((video, str(row["person_id"])), []).append(
                (timestamp, str(candidate["track_id"]) if candidate else None))
            actual.append(candidate["label"] if candidate else "missing")
            if candidate:
                matched_ious.append(float(overlap[i, paired[i]]))
                matches.append((row["label"], candidate["label"]))
                key = (video, str(candidate["track_id"]))
                person = str(row["person_id"])
                votes = identity_votes.setdefault(key, {})
                votes[person] = votes.get(person, 0) + 1
                event_identity[(video, str(candidate["track_id"]), nearest)] = person
        paired_predictions = set(paired.values())
        unmatched_labels.extend(row["label"] for j, row in enumerate(candidates) if j not in paired_predictions)
    unmatched_labels.extend(row["label"] for key, rows in pd_frames.items() if key not in used_frames for row in rows)
    unmatched_predictions = len(unmatched_labels)
    expected.extend(["background"] * unmatched_predictions)
    actual.extend(unmatched_labels)
    report = dict(ground_truth_instances=len(truth), matched_instances=len(matches),
                  pose_detection_coverage=len(matches) / len(truth) if truth else None,
                  unmatched_predictions=int(unmatched_predictions), labels=[*LABELS, "unknown", "missing", "background"],
                  classification=classification_report(expected, actual, labels=list(LABELS),
                                                       output_dict=True, zero_division=0),
                  confusion_matrix=confusion_matrix(expected, actual, labels=[*LABELS, "unknown", "missing", "background"]).tolist(),
                  matching=dict(frame_tolerance_seconds=frame_tolerance, bbox_iou=match_iou),
                  frame_coverage=dict(complete_manifests=gt_ledger is not None, truth_frames=len(gt_frames),
                                      processed_frames=len(pd_frames), matched_frames=len(used_frames),
                                      missing_frames=len(gt_frames) - len(assignment),
                                      empty_truth_frames=sum(not rows for rows in gt_frames.values())),
                  classification_scope="Missing detections count as false negatives; unmatched predictions count as "
                                       "false positives. Ground truth must cover all people in all predicted frames.")
    tp, fp, fn = len(matches), unmatched_predictions, len(truth) - len(matches)
    precision, recall = tp / max(tp + fp, 1), tp / max(len(truth), 1)
    report["detection"] = dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                               f1=2 * precision * recall / max(precision + recall, 1e-9),
                               matched_bbox_iou_mean=float(np.mean(matched_ious)) if matched_ious else None,
                               matched_bbox_iou_min=min(matched_ious) if matched_ious else None)
    report["provenance"] = dict(ground_truth_sha256=file_sha256(ground_truth),
                                predictions_sha256=file_sha256(predictions), truth_kind="manual_annotations")
    for name, path in (("ground_truth_frames", ground_truth_frames), ("prediction_frames", prediction_frames)):
        if path:
            report["provenance"][name + "_sha256"] = file_sha256(path)
    if gt_ledger is None:
        report["classification_scope"] += " Frame completeness is unverified; this is a legacy sparse evaluation."
    people = []
    for (video, person), history in histories.items():
        last, in_segment, segments, switches, reacquisitions = None, False, 0, 0, 0
        ids = set()
        for _, track in history:
            if track is None:
                in_segment = False
                continue
            ids.add(track)
            if not in_segment or track != last:
                segments += 1
                if last is not None and not in_segment:
                    reacquisitions += 1
            if last is not None and track != last:
                switches += 1
            last, in_segment = track, True
        people.append(dict(video_id=video, person_id=person, track_ids=sorted(ids),
                           matched_segments=segments, fragments=max(segments - 1, 0),
                           id_switches=switches, reacquisitions_after_miss=reacquisitions))
    report["tracking"] = dict(people=people, fragments=sum(p["fragments"] for p in people),
                               id_switches=sum(p["id_switches"] for p in people),
                               scope="Transitions over annotated observations; unannotated gaps are not measured")
    if matches:
        a, b = zip(*matches)
        report["matched_only_classification"] = classification_report(a, b, labels=list(LABELS),
                                                                       output_dict=True, zero_division=0)
    report["identity_purity"] = sum(max(v.values()) for v in identity_votes.values()) / max(len(matches), 1)
    if bool(events) != bool(truth_events):
        raise ValueError("Specify both --events and --truth-events")
    if events:
        report["falls"] = evaluate_events(events, truth_events, event_identity, max_alarm_delay, frame_tolerance,
                                          deadline_scope, event_delivery)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_events(events, truth_events, identity_map, max_delay=5.0, frame_tolerance=0.05,
                    deadline_scope="media", event_delivery=None):
    if deadline_scope not in {"media", "emitted"} or not np.isfinite(max_delay) or max_delay <= 0:
        raise ValueError("Invalid event deadline scope or delay")
    with Path(events).open(newline="", encoding="utf-8") as file:
        alerts = [r for r in csv.DictReader(file) if r["event"] == "fall"]
    truth = load_truth_events(truth_events)
    receipts = {}
    if event_delivery:
        with Path(event_delivery).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if str(row["event_id"]) in receipts:
                    raise ValueError("Duplicate event delivery receipt")
                receipts[str(row["event_id"])] = row
    used, delays, emitted_delays = set(), [], []
    late_alerts = 0
    identity_times = {}
    for key, person in identity_map.items():
        if len(key) == 3:
            identity_times.setdefault(key[:2], []).append((key[2], person))
    for rows in identity_times.values():
        rows.sort()
    for alert in sorted(alerts, key=lambda a: float(a["timestamp"])):
        video = str(alert["video_id"])
        timestamp = float(alert["timestamp"])
        if not np.isfinite(timestamp) or timestamp < 0 or not video.strip():
            raise ValueError("Invalid alarm timestamp or video")
        delivery = receipts.get(str(alert.get("event_id")), alert)
        latency = delivery.get("capture_to_emit_ms")
        if deadline_scope == "emitted" and (latency in (None, "") or
                                             (event_delivery and str(alert.get("event_id")) not in receipts) or
                                             delivery.get("capture_to_emit_scope") == "write_call"):
            raise ValueError("Emitted deadlines require measured flush delivery latency")
        latency = float(latency) / 1000 if latency not in (None, "") else None
        if latency is not None and (not np.isfinite(latency) or latency < 0):
            raise ValueError("Invalid emitted latency")
        deadline_timestamp = timestamp + (latency or 0) if deadline_scope == "emitted" else timestamp
        key = (video, str(alert["track_id"]))
        person = identity_map.get(key)
        rows = identity_times.get(key, [])
        if rows:
            index = bisect_left(rows, (timestamp, ""))
            nearby = rows[max(0, index - 1):index + 1]
            nearest, match = min(nearby, key=lambda row: abs(row[0] - timestamp))
            person = match if abs(nearest - timestamp) <= frame_tolerance + 1e-6 else None
        eligible = [i for i, row in enumerate(truth) if i not in used and str(row["video_id"]) == video and
                    str(row["person_id"]) == person and float(row["start"]) <= timestamp <= float(row["end"]) and
                    deadline_timestamp <= float(row["start"]) + max_delay]
        if not eligible and any(str(row["video_id"]) == video and str(row["person_id"]) == person and
                                float(row["start"]) <= timestamp <= float(row["end"]) and
                                deadline_timestamp > float(row["start"]) + max_delay for row in truth):
            late_alerts += 1
        if eligible:
            index = min(eligible, key=lambda i: float(truth[i]["start"]))
            used.add(index)
            delays.append(timestamp - float(truth[index]["start"]))
            if latency is not None:
                emitted_delays.append(delays[-1] + latency)
    tp, fp, fn = len(used), len(alerts) - len(used), len(truth) - len(used)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, late_alerts=late_alerts,
                deadline_scope=deadline_scope,
                f1=2 * precision * recall / max(precision + recall, 1e-9),
                alarm_delay_p50_seconds=float(np.percentile(delays, 50)) if delays else None,
                alarm_delay_p95_seconds=float(np.percentile(delays, 95)) if delays else None,
                emitted_alarm_delay_p50_seconds=float(np.percentile(emitted_delays, 50)) if emitted_delays else None,
                emitted_alarm_delay_p95_seconds=float(np.percentile(emitted_delays, 95)) if emitted_delays else None,
                emitted_delay_samples=len(emitted_delays),
                delay_scope="Media delay and measured capture-to-output delay; excludes camera/network buffering",
                max_alarm_delay_seconds=max_delay,
                identity_mapping="Spatial matches at alarm timestamp; unassociated alarms count as false positives")
