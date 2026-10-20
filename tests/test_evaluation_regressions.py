import csv
import json
import numpy as np
import pytest
from poseguard.annotations import write_jsonl
from poseguard.evaluation import evaluate_predictions, evaluate_events
from poseguard.features import predict_feature_labels, FEATURE_NAMES


def annotation(timestamp, label="standing", frame_id=None):
    row = dict(video_id="v", person_id="p", timestamp=timestamp, bbox=[0, 0, 100, 200], label=label)
    if frame_id is not None:
        row["frame_id"] = frame_id
    return row


def test_quality_gate_does_not_call_model_for_rejected_poses():
    class Model:
        classes_ = np.array(["standing", "lying"])
        def predict_proba(self, x):
            assert np.all(x[:, -2] >= .35)
            return np.tile([.8, .2], (len(x), 1))
    x = np.zeros((3, len(FEATURE_NAMES)))
    x[:, -2] = [0, .35, .34]
    assert predict_feature_labels(Model(), x).tolist() == ["unknown", "standing", "unknown"]


def test_exact_next_frame_prediction_is_not_stolen(tmp_path):
    gt, pred = tmp_path / "gt.jsonl", tmp_path / "pred.jsonl"
    rows = [annotation(0), annotation(1/30, "lying")]
    write_jsonl(gt, rows)
    write_jsonl(pred, [dict(rows[1], track_id=1)])
    report = evaluate_predictions(gt, pred, tmp_path / "report.json")
    assert report["classification"]["lying"]["recall"] == 1
    assert report["classification"]["standing"]["recall"] == 0
    assert report["detection"]["fn"] == 1


def test_ambiguous_time_matching_maximizes_one_to_one_coverage():
    from poseguard.evaluation import associate_frames
    gt = {("v", 0): [], ("v", .06): []}
    pd = {("v", .02): [], ("v", -.04): []}
    matches, used = associate_frames(gt, pd, .05)
    assert matches == {("v", 0): ("v", -.04), ("v", .06): ("v", .02)}
    assert len(used) == 2


def test_empty_scene_counts_false_positives_and_requires_coverage(tmp_path):
    gt, pred = tmp_path / "gt.jsonl", tmp_path / "pred.jsonl"
    gf, pf = tmp_path / "gf.jsonl", tmp_path / "pf.jsonl"
    write_jsonl(gt, [])
    write_jsonl(pred, [dict(annotation(0, frame_id=0), track_id=1)])
    write_jsonl(gf, [dict(video_id="v", timestamp=0, frame_id=0, person_count=0)])
    write_jsonl(pf, [dict(video_id="v", timestamp=0, frame_id=0, people=1)])
    report = evaluate_predictions(gt, pred, tmp_path / "report.json", ground_truth_frames=gf, prediction_frames=pf)
    assert report["detection"] == dict(tp=0, fp=1, fn=0, precision=0, recall=0, f1=0,
                                        matched_bbox_iou_mean=None, matched_bbox_iou_min=None)
    assert report["frame_coverage"]["empty_truth_frames"] == 1
    write_jsonl(pf, [dict(video_id="v", timestamp=0, frame_id=0, people=0)])
    with pytest.raises(ValueError, match="count"):
        evaluate_predictions(gt, pred, tmp_path / "bad.json", ground_truth_frames=gf, prediction_frames=pf)
    write_jsonl(pred, [])
    assert evaluate_predictions(gt, pred, tmp_path / "empty.json", ground_truth_frames=gf, prediction_frames=pf)["detection"]["fp"] == 0


def test_explicit_dropped_frame_ids_never_match_neighbor(tmp_path):
    gt, pred = tmp_path / "gt.jsonl", tmp_path / "pred.jsonl"
    gf, pf = tmp_path / "gf.jsonl", tmp_path / "pf.jsonl"
    rows = [annotation(0, frame_id=0), annotation(1/30, "lying", frame_id=1)]
    write_jsonl(gt, rows)
    write_jsonl(pred, [dict(rows[1], track_id=1)])
    write_jsonl(gf, [dict(video_id="v", timestamp=r["timestamp"], frame_id=r["frame_id"], person_count=1) for r in rows])
    write_jsonl(pf, [dict(video_id="v", timestamp=1/30, frame_id=1, people=1)])
    report = evaluate_predictions(gt, pred, tmp_path / "report.json", ground_truth_frames=gf, prediction_frames=pf)
    assert report["classification"]["lying"]["recall"] == 1
    assert report["frame_coverage"]["missing_frames"] == 1
    write_jsonl(pf, [dict(video_id="v", timestamp=.04, frame_id=2, people=1)])
    with pytest.raises(ValueError):
        evaluate_predictions(gt, pred, tmp_path / "bad.json", ground_truth_frames=gf, prediction_frames=pf)


def test_emitted_alarm_after_deadline_counts_fp_and_fn(tmp_path):
    alerts, truth = tmp_path / "events.csv", tmp_path / "truth.csv"
    alerts.write_text("video_id,track_id,timestamp,event,capture_to_emit_ms\nv,1,5,fall,1000\n")
    truth.write_text("video_id,person_id,start,end\nv,p,0,10\n")
    identity = {("v", "1"): "p"}
    assert evaluate_events(alerts, truth, identity)["tp"] == 1
    report = evaluate_events(alerts, truth, identity, deadline_scope="emitted")
    assert (report["tp"], report["fp"], report["fn"], report["late_alerts"]) == (0, 1, 1, 1)
    alerts.write_text("video_id,track_id,timestamp,event\nv,1,5,fall\n")
    with pytest.raises(ValueError, match="latency"):
        evaluate_events(alerts, truth, identity, deadline_scope="emitted")


def test_holdout_rejection_matches_live_predictions(tmp_path):
    from poseguard.dataset import train_classifier, score_classifier
    from poseguard.features import LearnedClassifier
    from poseguard.demo import synthetic_person
    rows = []
    for i in range(10):
        for t, label, angle in [(0, "standing", 0), (1, "lying", np.pi/2)]:
            det = synthetic_person((200, 270), angle)
            det.keypoints[:, 2] = 0
            rows.append(dict(video_id=f"v{i}", person_id=f"p{i}", timestamp=t, label=label,
                             bbox=det.bbox.tolist(), keypoints=det.keypoints.tolist(), synthetic=True))
    dataset = tmp_path / "data.jsonl"
    write_jsonl(dataset, rows)
    train_classifier(dataset, tmp_path / "model.joblib", tmp_path / "training")
    report = score_classifier(tmp_path / "training", tmp_path / "final.json")
    assert report["rejected_samples"] == 4
    assert report["classification"]["standing"]["recall"] == report["classification"]["lying"]["recall"] == 0
    classifier = LearnedClassifier(str(tmp_path / "model.joblib"))
    assert classifier.predict(det) == {"unknown": 1.0}
