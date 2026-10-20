import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from poseguard.annotations import file_sha256, load_annotations, write_jsonl
from poseguard.features import RuleClassifier
from poseguard.image_io import write_image
from poseguard.review import ReviewStore
from poseguard.tracking import Tracker
from poseguard.types import Detection, Inference


def tool_module(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_people_stay_unverified_when_opened_for_review(tmp_path):
    builder = tool_module("build_review_draft")
    reviewer = tool_module("annotate")
    task = dict(video_id="v", frame_id=0, timestamp=0, image="frame.jpg")
    points = np.full((17, 3), [50, 100, .9], dtype=np.float32)
    detection = Detection(np.array([0, 0, 100, 200], dtype=np.float32), .9, points)
    tracker = Tracker()
    people = builder.suggested_people(task, [detection], tracker, RuleClassifier())
    assert people[0]["source"] == "model_suggestion"
    assert people[0]["person_id"] == "" and people[0]["label"] is None
    assert people[0]["candidate_local_actor_id"] == "v:local_actor_1"
    assert people[0]["identity_verified"] is False
    assert people[0]["human_review_pending"] is True
    people2 = builder.suggested_people(dict(task, timestamp=.1), [detection], tracker, RuleClassifier())
    assert people2[0]["candidate_local_actor_id"] == people[0]["candidate_local_actor_id"]
    editable = reviewer.editable_people(dict(task, suggested_people=people))
    assert editable[0]["bbox"] == people[0]["bbox"]
    assert editable[0]["label"] == "" and editable[0]["person_id"] == ""
    store = ReviewStore(tmp_path / "review.sqlite")
    with pytest.raises(ValueError, match="unique identity"):
        store.save(task, editable, "reviewer")
    store.connection.close()


def test_full_suggestion_manifest_counts_coverage_and_cannot_be_truth(tmp_path, monkeypatch):
    builder = tool_module("build_review_draft")
    tasks_path = tmp_path / "tasks.jsonl"
    image = tmp_path / "frame.jpg"
    write_image(image, np.zeros((240, 320, 3), dtype=np.uint8))
    write_jsonl(tasks_path, [dict(video_id="v", frame_id=0, timestamp=0, image=image.name, official_label_hint=0)])
    model = tmp_path / "pose.onnx"
    model.write_bytes(b"fake model for contract test")
    raw = np.zeros((1, 56, 1), dtype=np.float32)
    raw[0, :5, 0] = [320, 320, 100, 200, .9]
    raw[0, 5:, 0] = np.tile([320, 320, .9], 17)

    class Backend:
        def infer(self, tensor):
            return Inference(raw)

        def close(self):
            pass

    monkeypatch.setattr("poseguard.backends.make_backend", lambda cfg: Backend())
    output = tmp_path / "draft"
    monkeypatch.setattr("sys.argv", ["build_review_draft", "--tasks", str(tasks_path), "--output", str(output),
                                    "--suggest-backend", "onnx", "--model", str(model)])
    assert builder.main() == 0
    manifest = json.loads((output / "draft-manifest.json").read_text())
    assert manifest["suggestion_generation"]["model_sha256"] == file_sha256(model)
    assert manifest["suggestion_generation"]["coverage"]["processed_tasks"] == 1
    assert manifest["completion"]["reviewed_tasks"] == 0
    assert manifest["human_review_pending"] is True and manifest["authoritative"] is False
    row = json.loads((output / "draft-review.jsonl").read_text())
    assert Path(row["image"]).is_file()
    assert len(row["suggested_people"]) == 1
    assert row["suggested_posture"] == "transition_candidate"
    with pytest.raises(ValueError, match="manual"):
        load_annotations(output / "draft-review.jsonl")
