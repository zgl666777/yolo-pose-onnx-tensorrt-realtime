import json
import pytest
import numpy as np
from poseguard.annotations import write_jsonl, load_annotations
from poseguard.image_io import write_image
from poseguard.review import ReviewStore


@pytest.fixture(autouse=True)
def task_images(tmp_path):
    for name in ("frame.jpg", "frame-0.jpg", "frame-1.jpg"):
        write_image(tmp_path / name, np.zeros((240, 320, 3), np.uint8))


def test_review_export_requires_all_frames_and_identity_evidence(tmp_path):
    tasks = [dict(video_id="v",frame_id=i,timestamp=i/30,image=f"frame-{i}.jpg") for i in range(2)]
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path,tasks)
    store = ReviewStore(tmp_path / "review.sqlite")
    person = dict(bbox=[0,0,100,200],person_id="subject-A",label="sitting",identity_verified=True)
    with pytest.raises(ValueError,match="evidence"):
        store.save(tasks[0],[person],"reviewer",image_root=tmp_path)
    person["identity_evidence"]="Verified same actor against video registry"
    store.save(tasks[0],[person],"reviewer",image_root=tmp_path)
    with pytest.raises(ValueError,match="Every task"):
        store.export(path,tmp_path / "unreviewed")
    assert not (tmp_path / "unreviewed").exists()
    store.save(tasks[1],[],"reviewer",image_root=tmp_path)
    store.save_event("v","subject-A",0,.02,"reviewer")
    report=store.export(path,tmp_path / "export")
    assert report["frames"]==2 and report["people_instances"]==1
    assert load_annotations(tmp_path / "export/manual.jsonl")[0]["bbox"]==[0,0,100,200]
    frames=[json.loads(s) for s in (tmp_path / "export/frames.jsonl").read_text().splitlines()]
    assert frames[1]["person_count"]==0
    with pytest.raises(FileExistsError):
        store.export(path,tmp_path / "export")
    store.connection.close()


def test_review_audit_reports_blockers_without_promoting_draft(tmp_path):
    tasks = [dict(video_id="v", frame_id=0, timestamp=0, image="frame.jpg")]
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path, tasks)
    store = ReviewStore(tmp_path / "review.sqlite")
    audit = store.audit(path)
    assert audit["status"] == "blocked"
    assert audit["ready_for_export"] is False
    assert audit["remaining_tasks"] == 1
    person = dict(bbox=[0, 0, 100, 200], person_id="subject-A", label="standing",
                  identity_verified=True, identity_evidence="registry review")
    store.save(tasks[0], [person], "reviewer", image_root=tmp_path)
    audit = store.audit(path)
    assert audit["status"] == "ready"
    assert audit["ready_for_export"] is True
    store.connection.close()


def test_review_audit_rejects_changed_task_and_uncovered_fall(tmp_path):
    task = dict(video_id="v", frame_id=0, timestamp=0, image="frame.jpg")
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path, [task])
    store = ReviewStore(tmp_path / "review.sqlite")
    person = dict(bbox=[0, 0, 100, 200], person_id="subject-A", label="falling",
                  identity_verified=True, identity_evidence="registry review")
    store.save(task, [person], "reviewer", image_root=tmp_path)
    audit = store.audit(path)
    assert len(audit["falling_without_event"]) == 1
    assert audit["ready_for_export"] is False
    store.save_event("v", "subject-A", 0, .2, "reviewer")
    assert store.audit(path)["ready_for_export"] is True
    write_jsonl(path, [dict(task, timestamp=.1)])
    audit = store.audit(path)
    assert len(audit["changed_tasks"]) == 1
    with pytest.raises(ValueError, match="current tasks"):
        store.export(path, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()
    store.connection.close()


def test_review_audit_rejects_empty_and_duplicate_queue(tmp_path):
    path = tmp_path / "tasks.jsonl"
    store = ReviewStore(tmp_path / "review.sqlite")
    write_jsonl(path, [])
    with pytest.raises(ValueError, match="empty"):
        store.audit(path)
    task = dict(video_id="v", frame_id=0, timestamp=0, image="frame.jpg")
    write_jsonl(path, [task, task])
    with pytest.raises(ValueError, match="Duplicate"):
        store.audit(path)
    store.connection.close()


def test_review_audit_rejects_orphan_event(tmp_path):
    task = dict(video_id="v", frame_id=0, timestamp=0, image="frame.jpg")
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path, [task])
    store = ReviewStore(tmp_path / "review.sqlite")
    store.save(task, [], "reviewer", image_root=tmp_path)
    store.save_event("v", "missing-person", 0, .2, "reviewer")
    assert store.audit(path)["ready_for_export"] is False
    with pytest.raises(ValueError, match="Event identity"):
        store.export(path, tmp_path / "orphan")
    assert not (tmp_path / "orphan").exists()
    store.connection.close()


def test_same_path_changed_pixels_require_fresh_human_review(tmp_path):
    task = dict(video_id="v", frame_id=0, timestamp=0, image="frame.jpg")
    tasks = tmp_path / "tasks.jsonl"
    write_jsonl(tasks, [task])
    store = ReviewStore(tmp_path / "review.sqlite")
    store.save(task, [], "reviewer", image_root=tmp_path)
    before = store.get(task)["image_sha256"]
    assert store.audit(tasks)["ready_for_export"] is True
    write_image(tmp_path / "frame.jpg", np.full((240, 320, 3), 127, np.uint8))
    assert store.audit(tasks)["ready_for_export"] is False
    with pytest.raises(ValueError, match="current tasks"):
        store.export(tasks, tmp_path / "changed-pixels")
    store.save(task, [], "reviewer", image_root=tmp_path)
    assert store.get(task)["image_sha256"] != before
    exported = tmp_path / "rechecked"
    store.export(tasks, exported)
    frames = [json.loads(line) for line in (exported / "frames.jsonl").read_text().splitlines()]
    assert frames[0]["image_sha256"] == store.get(task)["image_sha256"]
    store.connection.close()


def test_legacy_review_without_image_hash_cannot_be_exported(tmp_path):
    task = dict(video_id="v", frame_id=0, timestamp=0, image="frame.jpg")
    tasks = tmp_path / "tasks.jsonl"
    write_jsonl(tasks, [task])
    store = ReviewStore(tmp_path / "review.sqlite")
    store.save(task, [], "reviewer", image_root=tmp_path)
    payload = store.get(task)
    payload.pop("image_sha256")
    store.connection.execute("UPDATE frames SET payload=?", (json.dumps(payload),))
    store.connection.commit()
    assert store.audit(tasks)["ready_for_export"] is False
    (tmp_path / "frame.jpg").unlink()
    assert store.audit(tasks)["ready_for_export"] is False
    store.connection.close()


def test_review_save_rejects_missing_or_unreadable_image(tmp_path):
    task = dict(video_id="v", frame_id=0, timestamp=0, image="missing.jpg")
    store = ReviewStore(tmp_path / "review.sqlite")
    with pytest.raises(ValueError, match="does not exist"):
        store.save(task, [], "reviewer", image_root=tmp_path)
    (tmp_path / "missing.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="unreadable"):
        store.save(task, [], "reviewer", image_root=tmp_path)
    assert store.get(task) is None
    store.connection.close()


def test_absolute_task_image_is_preserved_when_image_root_is_set(tmp_path):
    image = tmp_path / "frame.jpg"
    task = dict(video_id="v", frame_id=0, timestamp=0, image=str(image))
    tasks = tmp_path / "tasks.jsonl"
    write_jsonl(tasks, [task])
    store = ReviewStore(tmp_path / "review.sqlite")
    store.save(task, [], "reviewer", image_root=tmp_path / "different-directory")
    assert store.audit(tasks)["ready_for_export"] is True
    store.connection.close()


def test_image_changed_since_display_cannot_be_saved_even_as_empty_scene(tmp_path):
    from poseguard.annotations import file_sha256
    task = dict(video_id="v", frame_id=0, timestamp=0, image="frame.jpg")
    store = ReviewStore(tmp_path / "review.sqlite")
    displayed_hash = file_sha256(tmp_path / "frame.jpg")
    write_image(tmp_path / "frame.jpg", np.full((240, 320, 3), 200, np.uint8))
    with pytest.raises(ValueError, match="since display"):
        store.save(task, [], "reviewer", image_root=tmp_path, expected_image_sha256=displayed_hash)
    assert store.get(task) is None
    store.connection.close()
