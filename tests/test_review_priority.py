import importlib.util
import json
from pathlib import Path

import pytest
from poseguard.annotations import load_annotations, write_jsonl, file_sha256


def priority_module():
    path = Path(__file__).resolve().parents[1] / "tools/build_review_priority.py"
    spec = importlib.util.spec_from_file_location("review_priority", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_priority_queue_uses_source_timing_and_preserves_unverified_suggestions(tmp_path):
    module = priority_module()
    tasks = tmp_path / "draft.jsonl"
    suggestion = dict(bbox=[0, 0, 10, 20], source="model_suggestion", person_id="", label=None,
                      identity_verified=False, candidate_local_actor_id="v:local_actor_1")
    rows = [dict(video_id="v", frame_id=i, timestamp=10 + i / 30, source_timestamp=i / 30,
                 image="frame.jpg", person_id="", label=None, bbox=None, annotation_source="draft",
                 suggested_people=[suggestion]) for i in range(15)]
    write_jsonl(tasks, rows)
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(json.dumps({"videos": [dict(video_id="v", fp=1, fn=0, alarms=[.2], onset=None,
                                  selected_frames=[dict(frame_id=5), dict(frame_id=14)])]}))
    manifest = module.build_queue(tasks, diagnostics, tmp_path / "priority", before=.1, after=.1, stride=3)
    queue = [json.loads(line) for line in (tmp_path / "priority/review-priority.jsonl").read_text().splitlines()]
    assert {5, 14} <= {r["frame_id"] for r in queue}
    assert all(.1 <= r["source_timestamp"] <= .3 + 1e-9 or r["frame_id"] == 14 for r in queue)
    assert all(r["person_id"] == "" and r["label"] is None and r["identity_verified"] is False for r in queue)
    assert all(r["suggested_people"] == [suggestion] for r in queue)
    assert all(r["human_review_pending"] is True for r in queue)
    assert manifest["source_tasks_sha256"] == file_sha256(tasks)
    assert manifest["authoritative"] is False
    with pytest.raises(ValueError, match="manual"):
        load_annotations(tmp_path / "priority/review-priority.jsonl")
    assert tasks.read_text() == "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


@pytest.mark.parametrize("before,after,stride", [(float("nan"), 1, 3), (1, float("inf"), 3), (1, 1, 1.2)])
def test_priority_queue_rejects_invalid_sampling(tmp_path, before, after, stride):
    with pytest.raises(ValueError, match="stride"):
        priority_module().build_queue(tmp_path / "none", tmp_path / "none", tmp_path / "out", before, after, stride)
