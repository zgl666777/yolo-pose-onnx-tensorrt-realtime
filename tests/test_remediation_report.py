import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load_tool(name):
    source = Path(__file__).resolve().parents[1] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name + "_test_module", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def precision_report():
    return dict(passed=True, tolerances=dict(coordinate_atol=.1, score_atol=.005, rtol=.001),
                results=[dict(candidate="models/mixed.engine", image=f"frames/{index}.jpg", passed=True,
                              raw_coordinates_passed=True, raw_scores_passed=True, postprocess_passed=True,
                              reference_detections=1, candidate_detections=1,
                              matches=[dict(reference_visible_keypoints=8, keypoint_check_sufficient=True,
                                            visibility_passed=True)], keypoint_evidence=True)
                         for index in range(3)])


def test_precision_subset_cannot_pass_full_frame_gate():
    module = load_tool("remediation_report")
    report = precision_report()
    assert module.precision_status(report, 11936)["status"] == "partial"
    assert module.precision_status(report, 3)["status"] == "completed"
    metrics = module.precision_status(report, 3)["metrics"]["candidates"]["models/mixed.engine"]
    assert metrics["insufficient_keypoint_evidence"] == 0
    assert metrics["no_reference_detections"] == 0
    assert metrics["images_with_visible_pose_evidence"] == 3


def test_empty_scenes_or_false_pose_claim_cannot_complete_pose_precision():
    module = load_tool("remediation_report")
    report = precision_report()
    for row in report["results"]:
        row.update(reference_detections=0, candidate_detections=0, matches=[], keypoint_evidence=False)
    assert module.precision_status(report, 3)["status"] == "partial"
    report["pose_evidence"] = dict(passed=True)
    assert module.precision_status(report, 3)["status"] == "partial"
    report = precision_report()
    report["pose_evidence"] = dict(passed=False)
    assert module.precision_status(report, 3)["status"] == "partial"
    report = precision_report()
    for row in report["results"]:
        row["matches"][0]["keypoint_check_sufficient"] = False
    assert module.precision_status(report, 3)["status"] == "partial"


@pytest.mark.parametrize("key,value", [("passed", False), ("raw_coordinates_passed", False),
                                      ("raw_scores_passed", False), ("postprocess_passed", False)])
def test_inconsistent_top_level_pass_cannot_override_failed_rows(key, value):
    module = load_tool("remediation_report")
    report = precision_report()
    report["results"][0][key] = value
    assert module.precision_status(report, 3)["status"] == "partial"


def test_relaxed_thresholds_duplicate_rows_and_malformed_rows_do_not_pass():
    module = load_tool("remediation_report")
    report = precision_report()
    report["tolerances"]["coordinate_atol"] = .2
    assert module.precision_status(report, 3)["status"] == "partial"
    report = precision_report()
    report["results"].append(report["results"][0])
    assert module.precision_status(report, 3)["status"] == "partial"
    report["results"] = [None]
    assert module.precision_status(report, 3)["status"] == "partial"


def test_absolute_candidate_names_are_not_exposed():
    module = load_tool("remediation_report")
    report = precision_report()
    for row in report["results"]:
        row["candidate"] = "C:/private/workspace/models/mixed.engine"
    result = module.precision_status(report, 3)
    assert list(result["metrics"]["candidates"]) == ["candidate:mixed.engine"]


def test_cpu_evidence_does_not_complete_linux_gpu():
    module = load_tool("remediation_report")
    report = dict(target_linux=True, profile="linux-cpu", status="completed", environment=dict(gpu=None),
                  stages=[dict(name="benchmark-onnx", exit_code=0), dict(name="replay-onnx-v0", exit_code=0)])
    assert module.linux_status(report)["status"] == "completed"
    assert module.linux_status(report, gpu=True)["status"] == "partial"
    report["target_linux"] = False
    assert module.linux_status(report)["status"] == "partial"


def test_manual_suggestions_cannot_become_authoritative():
    module = load_tool("remediation_report")
    draft = dict(status="suggestions", authoritative=False,
                 acceptance=dict(tasks=3, reviewed_tasks=3, remaining_tasks=0, ready_for_export=True, blockers=[]))
    assert module.manual_status(draft)["status"] == "blocked"
    draft.update(status="human_reviewed_export", authoritative=True)
    assert module.manual_status(draft)["status"] == "completed"
    draft["acceptance"]["reviewed_tasks"] = 0
    assert module.manual_status(draft)["status"] == "blocked"


def test_integrated_camera_or_claim_only_report_cannot_satisfy_physical_usb():
    module = load_tool("remediation_report")
    report = dict(status="passed_operator_attested", device_kind="integrated-camera")
    assert module.physical_status(report, "external-usb")["status"] == "partial"
    report["device_kind"] = "external-usb"
    assert module.physical_status(report, "external-usb")["status"] == "partial"
    assert module.physical_status(None, "external-usb")["status"] == "blocked"


def write_soak(tmp_path, timestamps=None):
    directory = tmp_path / "run"
    directory.mkdir()
    timestamps = list(range(3601)) if timestamps is None else timestamps
    summary = dict(status="completed", synthetic=False, elapsed_seconds=3601, processed=len(timestamps),
                   captured=len(timestamps), dropped_queue=0, dropped_stale=0, system_fps=1,
                   reconnects=0, failures=0, error=None, capture_error=None, cleanup_errors=[],
                   capture_thread_alive=False, capture_worker_alive=False)
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "provenance.json").write_text(json.dumps(dict(environment=dict(os="Linux-test"))))
    (directory / "health.jsonl").write_text('{}\n')
    with (directory / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "latency_ms", "rss_mb"])
        writer.writeheader()
        for timestamp in timestamps:
            writer.writerow(dict(timestamp=timestamp, latency_ms=20, rss_mb=100))
    return directory


def generate_soak(directory, explicit=True):
    module = load_tool("soak_report")
    command = ["--run", str(directory), "--output", str(directory / "soak.json")]
    if explicit:
        command += ["--min-seconds", "3600", "--max-rss-growth-mb", "32"]
    return module.main(command)


def test_completed_hour_requires_explicit_gate_and_correct_platform(tmp_path):
    module = load_tool("remediation_report")
    directory = write_soak(tmp_path)
    evidence = module.Evidence(tmp_path)
    assert generate_soak(directory, explicit=False) == 0
    assert module.soak_status(evidence, directory, 3600, True)["status"] == "partial"
    assert generate_soak(directory) == 0
    assert module.soak_status(evidence, directory, 3600, True)["status"] == "completed"
    assert module.soak_status(evidence, directory, 3600, False)["status"] == "partial"
    summary = json.loads((directory / "summary.json").read_text())
    summary["capture_error"] = "late worker failure"
    (directory / "summary.json").write_text(json.dumps(summary))
    assert module.soak_status(evidence, directory, 3600, True)["status"] == "partial"


def test_sparse_hour_does_not_pass_even_with_top_level_pass_claim(tmp_path):
    module = load_tool("remediation_report")
    directory = write_soak(tmp_path, [0, 3600])
    assert generate_soak(directory) == 1
    soak = json.loads((directory / "soak.json").read_text())
    soak["acceptance"]["passed"] = True
    for check in soak["acceptance"]["checks"].values():
        check["passed"] = True
    (directory / "soak.json").write_text(json.dumps(soak))
    assert module.soak_status(module.Evidence(tmp_path), directory, 3600, True)["status"] == "partial"


def test_missing_running_hour_has_no_pass(tmp_path):
    module = load_tool("remediation_report")
    directory = tmp_path / "running"
    directory.mkdir()
    (directory / "provenance.json").write_text('{"status":"running"}')
    assert module.soak_status(module.Evidence(tmp_path), directory, 3600, True)["status"] == "blocked"


def test_explicit_soak_report_can_be_selected_without_overwriting_history(tmp_path):
    module = load_tool("remediation_report")
    directory = write_soak(tmp_path)
    assert generate_soak(directory, explicit=False) == 0
    original = (directory / "soak.json").read_bytes()
    report = directory / "soak-frame-coverage.json"
    assert load_tool("soak_report").main(["--run", str(directory), "--output", str(report),
                                          "--min-seconds", "3600", "--max-rss-growth-mb", "32"]) == 0
    assert module.soak_status(module.Evidence(tmp_path), directory, 3600, True, report)["status"] == "completed"
    assert (directory / "soak.json").read_bytes() == original


def test_hashes_are_relative_and_outside_root_is_rejected(tmp_path):
    module = load_tool("remediation_report")
    evidence = module.Evidence(tmp_path)
    source = tmp_path / "sample.json"
    source.write_bytes(b'{"status":"partial"}')
    recorded = evidence.record(source)
    assert recorded["path"] == "sample.json"
    assert recorded["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="inside"):
        evidence.record(tmp_path.parent / "outside.json")


def test_precision_pass_requires_unchanged_model_hashes(tmp_path):
    module = load_tool("remediation_report")
    source = tmp_path / "model.engine"
    source.write_bytes(b"engine-before")
    report = precision_report()
    report["models"] = [dict(path="model.engine", sha256=hashlib.sha256(source.read_bytes()).hexdigest())]
    assert module.precision_status(report, 3, module.Evidence(tmp_path))["status"] == "completed"
    source.write_bytes(b"engine-after")
    result = module.precision_status(report, 3, module.Evidence(tmp_path))
    assert result["status"] == "partial"
    assert result["metrics"]["model_hashes_match"] is False


def test_failed_test_xml_or_nonzero_exit_cannot_be_complete(tmp_path):
    module = load_tool("remediation_report")
    source = tmp_path / "tests.xml"
    source.write_text('<testsuites><testsuite tests="4" failures="1" errors="0" skipped="0"/></testsuites>')
    assert module.test_status(module.Evidence(tmp_path), source)["status"] == "partial"
    source.write_text('<testsuite tests="4" failures="0" errors="0" skipped="1"/>')
    source.with_suffix(".exitcode").write_text("2")
    assert module.test_status(module.Evidence(tmp_path), source)["status"] == "partial"
    source.with_suffix(".exitcode").write_text("0")
    result = module.test_status(module.Evidence(tmp_path), source)
    assert result["status"] == "completed"
    assert result["metrics"]["passed"] == 3


def test_test_status_rejects_source_hash_mismatch(tmp_path):
    module = load_tool("remediation_report")
    source = tmp_path / "tests.xml"
    source.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>')
    source.with_suffix(".exitcode").write_text("0")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("before")
    source.with_suffix(".source.json").write_text(json.dumps(dict(files={
        "tracked.py": hashlib.sha256(tracked.read_bytes()).hexdigest()})))
    assert module.test_status(module.Evidence(tmp_path), source)["status"] == "completed"
    tracked.write_text("after")
    result = module.test_status(module.Evidence(tmp_path), source)
    assert result["status"] == "partial"
    assert result["metrics"]["source_manifest_valid"] is False
    assert result["metrics"]["source_manifest_mismatches"] == 1
    assert result["metrics"]["source_manifest"]["mismatches"][0]["path"] == "tracked.py"


def test_test_status_rejects_source_manifest_missing_file(tmp_path):
    module = load_tool("remediation_report")
    source = tmp_path / "tests.xml"
    source.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>')
    source.with_suffix(".exitcode").write_text("0")
    source.with_suffix(".source.json").write_text(json.dumps(dict(files={
        "missing.py": "0" * 64})))
    result = module.test_status(module.Evidence(tmp_path), source)
    assert result["status"] == "partial"
    assert result["metrics"]["source_manifest_valid"] is False
    assert result["metrics"]["source_manifest_mismatches"] == 1
    source.with_suffix(".source.json").write_text(json.dumps(dict(files={"missing.py": None})))
    assert module.test_status(module.Evidence(tmp_path), source)["status"] == "partial"


def test_missing_evidence_manifest_remains_blocked_and_protects_inputs(tmp_path):
    module = load_tool("remediation_report")
    command = ["--project-root", str(tmp_path), "--output", "delivery.json"]
    assert module.main(command) == 0
    report = json.loads((tmp_path / "delivery.json").read_text())
    assert report["status"] == "blocked"
    assert report["final_acceptance_complete"] is False
    source = tmp_path / "precision.json"
    source.write_text(json.dumps(precision_report()))
    original = source.read_bytes()
    with pytest.raises(SystemExit) as exc:
        module.main(["--project-root", str(tmp_path), "--output", "precision.json", "--precision", "precision.json"])
    assert exc.value.code == 2
    assert source.read_bytes() == original


def test_exploratory_video_claim_cannot_complete_independent_accuracy():
    module = load_tool("remediation_report")
    report = dict(status="passed", final_acceptance_complete=True,
                  reports=dict(fp32=dict(videos=["fall-01"], tp=1, fp=0, fn=0, events=dict(precision=1, recall=1))))
    result = module.coarse_status(report)
    assert result["status"] == "partial"
    assert result["metrics"]["backends"]["fp32"]["event_metrics"]["tp"] == 1
