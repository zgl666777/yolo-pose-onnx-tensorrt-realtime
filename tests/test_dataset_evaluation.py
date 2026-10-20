from pathlib import Path
import json
import csv
import numpy as np
import pytest
from poseguard.dataset import connected_groups, split_records, train_classifier, load_records
from poseguard.demo import synthetic_person
from poseguard.features import extract_features, LearnedClassifier
from poseguard.evaluation import evaluate_predictions, evaluate_events


def record(video="v1", person="p1", timestamp=0, label="standing", angle=0):
    det = synthetic_person((200, 270), angle)
    return dict(video_id=video, person_id=person, timestamp=timestamp, label=label,
                bbox=det.bbox.tolist(), keypoints=det.keypoints.tolist(), synthetic=True)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_split_connects_both_shared_people_and_shared_videos():
    rows = [record("v1", "p1"), record("v1", "p2"), record("v2", "p2")]
    rows.extend(record(f"v{i}", f"p{i}") for i in range(3, 10))
    groups = connected_groups(rows)
    assert len(set(groups[:3])) == 1
    parts = split_records(rows)
    for key in ("person_id", "video_id"):
        sets = [{rows[i][key] for i in indices} for indices in parts.values()]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_split_refuses_leaky_small_dataset():
    with pytest.raises(ValueError, match="independent"):
        split_records([record("v1", "p1"), record("v2", "p1")])


def test_features_are_translation_and_uniform_scale_invariant():
    a, b = synthetic_person((200, 270), scale=100), synthetic_person((500, 500), scale=100)
    # Include bbox margins in the affine transform.
    b.bbox = a.bbox * 2 + np.tile([100, 200], 2)
    b.keypoints[:, :2] = a.keypoints[:, :2] * 2 + [100, 200]
    np.testing.assert_allclose(extract_features(a), extract_features(b), atol=1e-6)


def test_missing_detections_and_unmatched_predictions_penalize_metrics(tmp_path):
    truth = [record(timestamp=0), record(timestamp=0.1)]
    pred = [dict(truth[0], track_id=3), dict(record(timestamp=0.2), track_id=7)]
    gt_path, pd_path = tmp_path / "gt.jsonl", tmp_path / "pd.jsonl"
    write_jsonl(gt_path, truth)
    write_jsonl(pd_path, pred)
    report = evaluate_predictions(gt_path, pd_path, tmp_path / "report.json", frame_tolerance=0.01)
    assert report["matched_instances"] == 1
    assert report["unmatched_predictions"] == 1
    assert report["classification"]["standing"]["recall"] == 0.5
    assert report["classification"]["standing"]["precision"] == 0.5


def test_duplicate_alerts_and_wrong_person_count_as_false_positive(tmp_path):
    events, truth = tmp_path / "events.csv", tmp_path / "truth.csv"
    events.write_text("video_id,track_id,timestamp,event\nv1,1,2.0,fall\nv1,1,2.2,fall\nv1,2,2.5,fall\n", encoding="utf-8")
    truth.write_text("video_id,person_id,start,end\nv1,p1,1,5\n", encoding="utf-8")
    report = evaluate_events(events, truth, {("v1", "1"): "p1", ("v1", "2"): "p2"})
    assert (report["tp"], report["fp"], report["fn"]) == (1, 2, 0)
    assert report["alarm_delay_p50_seconds"] == 1


def test_train_save_load_and_feature_schema(tmp_path):
    rows = []
    for i in range(10):
        rows.extend([record(f"v{i}", f"p{i}", 0, "standing"),
                     record(f"v{i}", f"p{i}", 1, "lying", np.pi / 2)])
    path, model = tmp_path / "data.jsonl", tmp_path / "model.joblib"
    write_jsonl(path, rows)
    result = train_classifier(path, model, tmp_path / "reports")
    assert sum(p["samples"] for p in result["splits"].values()) == len(rows)
    classifier = LearnedClassifier(model)
    assert max(classifier.predict(synthetic_person((200, 270))), key=classifier.predict(synthetic_person((200, 270))).get) == "standing"
    with pytest.raises(ValueError, match="threshold"):
        classifier.predict(synthetic_person((200, 270)), 0.5)
    invisible = synthetic_person((200, 270))
    invisible.keypoints[:, 2] = 0
    assert classifier.predict(invisible) == {"unknown": 1.0}


def test_duplicate_annotations_rejected(tmp_path):
    path = tmp_path / "duplicates.jsonl"
    write_jsonl(path, [record(), record()])
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(path)


def test_alarm_identity_uses_event_time_instead_of_majority(tmp_path):
    gt_path, pd_path = tmp_path / "gt.jsonl", tmp_path / "pd.jsonl"
    truth = [record(person="p1", timestamp=0), record(person="p1", timestamp=1),
             record(person="p2", timestamp=2)]
    pred = [dict(row, track_id=1) for row in truth]
    write_jsonl(gt_path, truth)
    write_jsonl(pd_path, pred)
    events, falls = tmp_path / "events.csv", tmp_path / "falls.csv"
    events.write_text("video_id,track_id,timestamp,event\nv1,1,2.0,fall\n", encoding="utf-8")
    falls.write_text("video_id,person_id,start,end\nv1,p1,1.5,5\n", encoding="utf-8")
    report = evaluate_predictions(gt_path, pd_path, tmp_path / "evaluation.json", events, falls)
    assert report["falls"]["tp"] == 0
    assert report["falls"]["fp"] == 1
    assert report["falls"]["fn"] == 1


def test_extraction_keeps_complete_manual_truth_and_original_box(tmp_path, monkeypatch):
    import cv2
    from types import SimpleNamespace
    from poseguard.config import Config
    from poseguard.dataset import extract_dataset
    from poseguard import backends, geometry
    from poseguard.types import Detection
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((300, 400, 3), np.uint8))
    rows = [dict(video_id="v", person_id="p1", timestamp=0, bbox=[10, 10, 110, 210], label="standing", image=image.name),
            dict(video_id="v", person_id="p2", timestamp=0, bbox=[200, 10, 300, 210], label="standing", image=image.name)]
    gt, features = tmp_path / "manual.jsonl", tmp_path / "features.jsonl"
    write_jsonl(gt, rows)
    original = gt.read_bytes()
    predicted_box = np.array([15, 15, 115, 215], np.float32)
    class Backend:
        def infer(self, tensor): return SimpleNamespace(output=None)
        def close(self): pass
    monkeypatch.setattr(backends, "make_backend", lambda cfg: Backend())
    monkeypatch.setattr(geometry, "decode", lambda *args: [Detection(predicted_box, .9, np.ones((17, 3)))])
    result = extract_dataset(gt, Config(), features)
    assert (result["accepted"], result["skipped"]) == (1, 1)
    saved_truth = [json.loads(x) for x in Path(result["ground_truth"]).read_text().splitlines()]
    assert len(saved_truth) == 2 and saved_truth[0]["bbox"] == rows[0]["bbox"]
    assert gt.read_bytes() == original
    feature = json.loads(features.read_text())
    assert feature["annotation_bbox"] == rows[0]["bbox"] and feature["bbox"] == predicted_box.tolist()
    pred = tmp_path / "pred.jsonl"
    write_jsonl(pred, [dict(feature, track_id=1)])
    report = evaluate_predictions(result["ground_truth"], pred, tmp_path / "eval.json")
    assert report["detection"]["fn"] == 1
    assert report["classification"]["standing"]["recall"] == .5
    assert 0.8 < report["detection"]["matched_bbox_iou_mean"] < 1
    with pytest.raises(ValueError, match="not ground truth"):
        evaluate_predictions(features, pred, tmp_path / "bad.json")
    with pytest.raises(FileExistsError):
        extract_dataset(gt, Config(), features)


def test_pure_but_fragmented_tracks_are_counted(tmp_path):
    rows = [record(timestamp=i / 10) for i in range(6)]
    gt, pd = tmp_path / "gt.jsonl", tmp_path / "pred.jsonl"
    write_jsonl(gt, rows)
    write_jsonl(pd, [dict(r, track_id=i) for i, r in enumerate(rows)])
    report = evaluate_predictions(gt, pd, tmp_path / "report.json")
    assert report["identity_purity"] == 1
    assert report["tracking"]["fragments"] == report["tracking"]["id_switches"] == 5


def test_holdout_is_unscored_until_explicit_final_step_and_hashes_enforced(tmp_path):
    from poseguard.dataset import score_classifier
    rows = [record(f"v{i}", f"p{i}", t, label, angle) for i in range(10)
            for t, label, angle in [(0, "standing", 0), (1, "lying", np.pi/2)]]
    dataset = tmp_path / "data.jsonl"
    write_jsonl(dataset, rows)
    reports = tmp_path / "training"
    result = train_classifier(dataset, tmp_path / "model.joblib", reports)
    assert not result["splits"]["test"]["evaluated"]
    assert "metrics" not in result["splits"]["test"]
    assert score_classifier(reports, tmp_path / "final.json")["synthetic"]
    with (reports / "test.jsonl").open("a") as f: f.write("\n")
    with pytest.raises(ValueError, match="changed"):
        score_classifier(reports, tmp_path / "bad-final.json")


def test_real_splits_require_verified_cross_video_identities(tmp_path):
    from poseguard.dataset import prepare_splits
    rows = [dict(record(f"v{i}", f"p{i}"), synthetic=False) for i in range(10)]
    path = tmp_path / "manual.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="identity_verified"):
        prepare_splits(path, tmp_path / "splits")
    for row in rows:
        row.update(identity_verified=True, annotation_source="manual")
    write_jsonl(path, rows)
    manifest = prepare_splits(path, tmp_path / "splits")
    assert sum(p["samples"] for p in manifest["splits"].values()) == 10


def test_real_training_uses_frozen_boxes_labels_and_source_hashes(tmp_path):
    from poseguard.dataset import prepare_splits, score_classifier
    from poseguard.annotations import file_sha256
    rows = [dict(record(f"v{i}", f"p{i}", t, label, angle), synthetic=False,
                 identity_verified=True, annotation_source="manual") for i in range(10)
            for t, label, angle in [(0, "standing", 0), (1, "lying", np.pi/2)]]
    manual, features = tmp_path / "manual.jsonl", tmp_path / "features.jsonl"
    write_jsonl(manual, rows)
    prepare_splits(manual, tmp_path / "frozen")
    for row in rows:
        row.update(record_type="pose_features", annotation_bbox=list(row["bbox"]),
                   annotations_sha256=file_sha256(manual))
    write_jsonl(features, rows)
    manifest = tmp_path / "frozen/split-manifest.json"
    result = train_classifier(features, tmp_path / "real.joblib", tmp_path / "reports", split_manifest=manifest)
    assert not result["synthetic"] and not result["splits"]["test"]["evaluated"]
    assert not score_classifier(tmp_path / "reports", tmp_path / "final.json")["synthetic"]
    rows[0]["annotation_bbox"][0] += 1
    write_jsonl(features, rows)
    with pytest.raises(ValueError, match="provenance"):
        train_classifier(features, tmp_path / "bad.joblib", tmp_path / "bad-reports", split_manifest=manifest)
