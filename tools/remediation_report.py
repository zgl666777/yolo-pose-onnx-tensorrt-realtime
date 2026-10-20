"""Build a conservative, hashed delivery manifest from existing evidence only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import xml.etree.ElementTree as ET


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}

    def resolve(self, path):
        path = Path(path)
        resolved = (path if path.is_absolute() else self.root / path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ValueError("Evidence must be inside --project-root") from None
        return resolved

    def record(self, path):
        source = self.resolve(path)
        relative = source.relative_to(self.root).as_posix()
        result = dict(path=relative, exists=source.is_file())
        if source.is_file():
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result.update(bytes=source.stat().st_size, sha256=digest.hexdigest())
        self.files[relative] = result
        return result

    def read(self, path):
        result = self.record(path)
        if not result["exists"]:
            return None
        try:
            value = json.loads(self.resolve(path).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            return value
        except (OSError, ValueError, UnicodeError) as exc:
            result["read_error"] = type(exc).__name__
            return None


def item(status, reason, **metrics):
    return dict(status=status, reason=reason, metrics=metrics)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def candidate_label(value):
    value = str(value).replace("\\", "/")
    if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        return "candidate:" + PurePosixPath(value).name
    return value


def verify_artifacts(evidence, artifacts):
    if evidence is None:
        return None
    if not isinstance(artifacts, list) or not artifacts:
        return False
    verified = True
    for reference in artifacts:
        if not isinstance(reference, dict) or not reference.get("path"):
            verified = False
            continue
        try:
            actual = evidence.record(reference["path"])
            matches = actual.get("sha256") == reference.get("sha256") and bool(actual.get("sha256"))
            actual["referenced_sha256_matches"] = matches
            verified = verified and matches
        except (ValueError, OSError):
            verified = False
    return verified


def verify_source_manifest(evidence, path):
    """Check the per-test source hash manifest against this checkout."""
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            return dict(valid=False, reason="missing files map", checked=0, mismatches=[])
        mismatches = []
        checked = 0
        for relative, expected in files.items():
            checked += 1
            try:
                artifact = evidence.record(relative)
                actual = artifact.get("sha256")
            except (ValueError, OSError):
                actual = None
            valid_digest = isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected)
            if not actual or not valid_digest or actual != expected:
                mismatches.append(dict(path=candidate_label(relative), expected=expected, actual=actual))
        return dict(valid=not mismatches, reason="all source hashes match" if not mismatches else "source hash mismatch",
                    checked=checked, mismatches=mismatches)
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError) as exc:
        return dict(valid=False, reason=f"malformed source manifest: {type(exc).__name__}", checked=0, mismatches=[])


def precision_status(report, expected_frames, evidence=None):
    if not report:
        return item("blocked", "No readable precision evidence")
    groups = defaultdict(list)
    rows = report.get("results", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return item("partial", "Malformed precision rows cannot establish a pass")
    for row in rows:
        groups[candidate_label(row.get("candidate", "unknown"))].append(row)
    candidates = {}
    for name, rows in groups.items():
        counts = Counter()
        images = {str(row.get("image")) for row in rows if row.get("image")}
        for row in rows:
            for key in ("passed", "raw_coordinates_passed", "raw_scores_passed", "postprocess_passed"):
                counts[key] += row.get(key) is True
            matches = row.get("matches") or []
            matches = [m for m in matches if isinstance(m, dict)] if isinstance(matches, list) else []
            counts["insufficient_keypoint_evidence"] += any(m.get("keypoint_check_sufficient") is False for m in matches)
            counts["no_reference_detections"] += row.get("reference_detections") == 0
            counts["images_with_visible_pose_evidence"] += (row.get("keypoint_evidence") is True and bool(matches) and
                all(m.get("keypoint_check_sufficient") is True and finite_number(m.get("reference_visible_keypoints")) and
                    m["reference_visible_keypoints"] >= 4 for m in matches))
            counts["visibility_changed"] += any(m.get("visibility_passed") is False for m in matches)
            counts["detection_count_changed"] += row.get("reference_detections") != row.get("candidate_detections")
        candidates[name] = dict(rows=len(rows), unique_images=len(images),
            passed=counts["passed"], failed=len(rows)-counts["passed"],
            raw_coordinate_failures=len(rows)-counts["raw_coordinates_passed"],
            raw_score_failures=len(rows)-counts["raw_scores_passed"],
            postprocess_failures=len(rows)-counts["postprocess_passed"],
            insufficient_keypoint_evidence=counts["insufficient_keypoint_evidence"],
            visibility_changed=counts["visibility_changed"], detection_count_changed=counts["detection_count_changed"],
            no_reference_detections=counts["no_reference_detections"],
            images_with_visible_pose_evidence=counts["images_with_visible_pose_evidence"])
    tolerances = report.get("tolerances") or {}
    strict_thresholds = isinstance(tolerances, dict) and all(
        finite_number(tolerances.get(key)) and 0 <= tolerances[key] <= maximum
        for key, maximum in dict(coordinate_atol=.1, score_atol=.005, rtol=.001).items())
    model_hashes_match = verify_artifacts(evidence, report.get("models"))
    if evidence is not None:
        for name in candidates:
            if name.startswith("candidate:") or name == "unknown":
                continue
            for suffix in (".json", ".layers.json"):
                path = name + suffix
                if evidence.resolve(path).is_file():
                    evidence.record(path)
    declared_pose = report.get("pose_evidence")
    visible_pose = bool(candidates) and all(c["images_with_visible_pose_evidence"] > 0 for c in candidates.values())
    pose_evidence_passed = visible_pose and (declared_pose is None or
                                           isinstance(declared_pose, dict) and declared_pose.get("passed") is True)
    valid = (pose_evidence_passed and model_hashes_match is not False and report.get("passed") is True and bool(candidates) and strict_thresholds and
             all(c["failed"] == c["raw_coordinate_failures"] == c["raw_score_failures"] == c["postprocess_failures"] == 0 and
                 c["rows"] == c["unique_images"] == expected_frames for c in candidates.values()))
    return item("completed" if valid else "partial",
                "Strict comparison completed for the declared full frame set" if valid else
                "Comparison failed, is inconsistent, or only covers a subset; strict acceptance remains incomplete",
                report_passed=report.get("passed"), expected_frames=expected_frames,
                thresholds=report.get("tolerances"), strict_thresholds=strict_thresholds,
                model_hashes_match=model_hashes_match, pose_evidence_passed=pose_evidence_passed, candidates=candidates,
                scope="Raw and postprocess agreement only; does not establish FP16 execution policy or manual event accuracy")


def test_status(evidence, path):
    artifact = evidence.record(path)
    if not artifact["exists"]:
        return item("blocked", "No test XML found")
    try:
        root = ET.parse(evidence.resolve(path)).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
        counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        passed = all(value >= 0 for value in counts.values()) and counts["passed"] > 0 and counts["failures"] == counts["errors"] == 0
        exit_path = evidence.resolve(path).with_suffix(".exitcode")
        if exit_path.is_file():
            evidence.record(exit_path)
            counts["exit_code"] = int(exit_path.read_text().strip())
            passed = passed and counts["exit_code"] == 0
        log_path = evidence.resolve(path).with_suffix(".log")
        if log_path.is_file():
            evidence.record(log_path)
        source_path = evidence.resolve(path).with_suffix(".source.json")
        if source_path.is_file():
            evidence.record(source_path)
            source_manifest = verify_source_manifest(evidence, source_path)
            counts["source_manifest_valid"] = source_manifest["valid"]
            counts["source_manifest_checked"] = source_manifest["checked"]
            counts["source_manifest_mismatches"] = len(source_manifest["mismatches"])
            counts["source_manifest"] = source_manifest
            passed = passed and source_manifest["valid"]
        else:
            source_manifest = None
        return item("completed" if passed else "partial", "Parsed regression test result; skipped capabilities remain unverified", **counts)
    except (OSError, ValueError, ET.ParseError) as exc:
        return item("partial", f"Unreadable or malformed test result: {type(exc).__name__}")


def linux_status(report, gpu=False):
    if not report:
        return item("blocked", "No readable Linux acceptance report")
    stages = report.get("stages", [])
    if not isinstance(stages, list) or any(not isinstance(s, dict) for s in stages):
        return item("partial", "Malformed acceptance stages cannot establish a Linux pass")
    executed = [s for s in stages if "exit_code" in s]
    skipped = [s.get("name") for s in stages if s.get("status") == "skipped"]
    target = report.get("target_linux") is True
    profile = report.get("profile") == ("linux-gpu" if gpu else "linux-cpu")
    required = {"benchmark-onnx"} if not gpu else {"precision", "cmake-configure", "cmake-build", "build-fp32", "build-fp16"}
    stage_names = {s.get("name") for s in executed}
    passed = target and profile and report.get("status") == "completed" and bool(executed) and all(
        s.get("exit_code") == 0 and not s.get("timed_out") for s in executed) and required <= stage_names
    if not gpu:
        passed = passed and any(name.startswith("replay-onnx-") for name in stage_names if name)
    else:
        environment = report.get("environment") or {}
        passed = passed and isinstance(environment, dict) and bool(environment.get("gpu")) and not skipped
        passed = passed and {"benchmark-trt-fp32", "benchmark-trt-fp16", "benchmark-cpp-fp32", "benchmark-cpp-fp16"} <= stage_names
    return item("completed" if passed else "partial", "Linux GPU rebuild evidence" if gpu else
                "Linux CPU software evidence only; does not establish GPU or physical device acceptance",
                target_linux=target, profile=report.get("profile"), report_status=report.get("status"),
                executed_stages=len(executed), skipped_stages=skipped,
                final_acceptance_complete=report.get("final_acceptance_complete") is True)


def manual_status(report):
    if not report:
        return item("blocked", "No readable manual review audit/export")
    audit = report.get("acceptance", report)
    completed = (isinstance(audit, dict) and report.get("status") == "human_reviewed_export" and report.get("authoritative") is True and
                 audit.get("ready_for_export") is True and audit.get("remaining_tasks") == 0 and
                 isinstance(audit.get("tasks"), int) and audit["tasks"] > 0 and
                 audit.get("reviewed_tasks") == audit["tasks"] and not audit.get("blockers"))
    if not isinstance(audit, dict):
        return item("partial", "Malformed manual acceptance audit")
    return item("completed" if completed else "blocked", "Human-reviewed export available" if completed else
                "Identity, boxes, posture or event review is incomplete; suggestions are not authoritative truth",
                tasks=audit.get("tasks"), reviewed_tasks=audit.get("reviewed_tasks"),
                remaining_tasks=audit.get("remaining_tasks"), people_instances=audit.get("people_instances"),
                identity_verified_people=audit.get("identity_verified_people"), blockers=audit.get("blockers", []),
                scope="Manual selected-task export; final independent split/training/scoring remains separate")


def coarse_status(report):
    if not report:
        return item("blocked", "No exploratory video report")
    metrics = {}
    reports = report.get("reports") or {}
    if not isinstance(reports, dict) or any(not isinstance(row, dict) for row in reports.values()):
        return item("partial", "Malformed exploratory report")
    for backend, row in reports.items():
        counts = row.get("events", row.get("event_metrics", {}))
        # The official-coarse schema keeps event metrics directly in each backend.
        if not counts:
            counts = {key: row.get(key) for key in ("tp", "fp", "fn", "precision", "recall") if key in row}
        if isinstance(counts, dict):
            counts = dict(counts, **{key: row[key] for key in ("tp", "fp", "fn") if key in row})
        metrics[backend] = dict(videos=len(row.get("videos", [])), event_metrics=counts)
    return item("partial", "Official coarse labels are exploratory and cannot complete independent manual accuracy acceptance",
                report_status=report.get("status"), backends=metrics,
                final_acceptance_complete=report.get("final_acceptance_complete") is True)


def rtsp_status(report, evidence=None):
    if not report:
        return item("blocked", "No correlated RTSP fault report")
    summary = report.get("summary", {})
    generations = report.get("delivered_generations", [])
    if not isinstance(summary, dict) or not isinstance(generations, list) or any(not isinstance(g, int) for g in generations):
        return item("partial", "Malformed RTSP recovery report")
    references = list((report.get("evidence") or {}).values()) + [report.get("upstream_evidence")]
    evidence_hashes_match = verify_artifacts(evidence, references)
    passed = (evidence_hashes_match is not False and report.get("status") == "passed" and summary.get("status") == "completed" and
              summary.get("failures") == 0 and summary.get("capture_thread_alive") is False and
              summary.get("capture_worker_alive") is False and summary.get("error") is None and
              summary.get("capture_error") is None and summary.get("cleanup_errors") == [] and len(set(generations)) >= 2 and
              finite_number(report.get("publisher_outage_seconds")) and report["publisher_outage_seconds"] > 0 and
              bool(report.get("failure_events")))
    return item("completed" if passed else "partial", "RTSP publisher fault and frame recovery evidence only",
                elapsed_seconds=summary.get("elapsed_seconds"), processed=summary.get("processed"),
                reconnects=summary.get("reconnects"), publisher_outage_seconds=report.get("publisher_outage_seconds"),
                delivered_generations=generations, evidence_hashes_match=evidence_hashes_match,
                physical_usb_fault=report.get("physical_usb_fault") is True,
                remote_rtsp_camera_hardware_fault=report.get("remote_rtsp_camera_hardware_fault") is True,
                scope=report.get("scope"))


def soak_status(evidence, directory, min_seconds, expect_linux, soak_report=None):
    directory = evidence.resolve(directory)
    summary = evidence.read(directory / "summary.json")
    provenance = evidence.read(directory / "provenance.json")
    soak = evidence.read(soak_report if soak_report is not None else directory / "soak.json")
    frames_artifact = evidence.record(directory / "frames.csv")
    health_artifact = evidence.record(directory / "health.jsonl")
    if not summary or not frames_artifact["exists"]:
        return item("blocked", "No completed summary and frame ledger; a running process is not a soak pass")
    try:
        with (directory / "frames.csv").open(newline="", encoding="utf-8") as stream:
            timestamps = [float(row["timestamp"]) for row in csv.DictReader(stream)]
        span = timestamps[-1]-timestamps[0] if len(timestamps) > 1 else 0
        gaps = [b-a for a, b in zip(timestamps, timestamps[1:])]
        acceptance = (soak or {}).get("acceptance", {})
        checks = acceptance.get("checks", {})
        thresholds = acceptance.get("thresholds", {})
        environment_os = (provenance or {}).get("environment", {}).get("os", "")
        platform_ok = environment_os.startswith("Linux") if expect_linux else environment_os.startswith("Windows")
        required = {"run_completed", "no_inference_failures", "no_run_error", "no_capture_error", "no_cleanup_errors",
                    "capture_thread_alive_false", "capture_worker_alive_false", "recorded_exit_codes_zero", "finite_statistics",
                    "capture_count_consistent", "frame_ledger_complete", "min_seconds", "frame_time_coverage",
                    "frame_timestamps_monotonic", "max_frame_gap_seconds"}
        threshold_seconds = thresholds.get("min_seconds")
        passed = (health_artifact["exists"] and platform_ok and summary.get("synthetic") is False and summary.get("status") == "completed" and
                  summary.get("failures") == 0 and summary.get("capture_thread_alive") is False and
                  summary.get("capture_worker_alive") is False and summary.get("error") is None and
                  summary.get("capture_error") is None and summary.get("cleanup_errors") == [] and
                  all(summary[name] == 0 for name in ("exit_code", "process_exit_code", "returncode", "capture_worker_exit_code", "worker_exit_code")
                      if name in summary and summary[name] is not None) and summary.get("processed") == len(timestamps) and
                  summary.get("elapsed_seconds", 0) >= min_seconds and span >= max(min_seconds-5, min_seconds*.95) and
                  all(math.isfinite(t) for t in timestamps) and bool(gaps) and min(gaps) > 0 and
                  acceptance.get("enabled") is True and acceptance.get("passed") is True and
                  required <= set(checks) and all(isinstance(c, dict) and c.get("passed") is True for c in checks.values()) and
                  finite_number(threshold_seconds) and threshold_seconds >= min_seconds and
                  finite_number(thresholds.get("max_frame_gap_seconds")) and thresholds["max_frame_gap_seconds"] > 0 and
                  max(gaps) <= thresholds["max_frame_gap_seconds"])
        return item("completed" if passed else "partial", "Completed explicit soak gate for this platform" if passed else
                    "Missing explicit threshold pass, sustained frame coverage, clean shutdown or correct platform",
                    elapsed_seconds=summary.get("elapsed_seconds"), processed=len(timestamps), frame_time_span_seconds=span,
                    max_frame_gap_seconds=max(gaps) if gaps else None, environment_os=environment_os,
                    thresholds=thresholds, drop_ratio=(soak or {}).get("drop_ratio"),
                    latency_p95_ms=(soak or {}).get("latency_p95_ms"), rss_change_mb=(soak or {}).get("rss_change_mb"),
                    rss_scope=(soak or {}).get("rss_scope", "Main process only; capture child RSS not recorded"),
                    scope="Stability thresholds only; no independent physical-fault or real-time-performance claim")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        return item("partial", f"Invalid soak evidence: {type(exc).__name__}")


def physical_status(report, kind):
    if not report:
        return item("blocked", "Required physical device and operator-attested evidence are unavailable")
    required = {"disconnect_attested", "reconnect_attested", "read_failure_observed", "resumed_frame_delivery",
                "run_completed", "capture_stopped", "no_capture_error"}
    checks = report.get("checks") or {}
    action_rows = report.get("actions") or []
    if not isinstance(checks, dict) or not isinstance(action_rows, list):
        return item("partial", "Malformed physical fault report")
    actions = {row.get("action") for row in action_rows if isinstance(row, dict)}
    passed = (report.get("status") == "passed_operator_attested" and report.get("device_kind") == kind and
              bool(report.get("operator")) and bool(report.get("device_id")) and report.get("client_exit_code") == 0 and
              required <= set(checks) and all(checks[name] is True for name in required) and
              {"disconnect_completed", "reconnect_completed"} <= actions)
    return item("completed" if passed else "partial", "Operator-attested physical fault result; inspect the source report",
                report_status=report.get("status"), device_kind=report.get("device_kind"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", default="runs/remediation/precision-mixed-all11936.json")
    parser.add_argument("--precision-control", action="append", default=[])
    parser.add_argument("--expected-precision-frames", type=int, default=11936)
    parser.add_argument("--linux-acceptance", default="runs/remediation/ubuntu-evidence/runs/linux-vmware-cpu/acceptance.json")
    parser.add_argument("--linux-tests", default="runs/remediation/ubuntu-final-tests.xml")
    parser.add_argument("--windows-tests", default="runs/remediation/final-windows-gpu-tests.xml")
    parser.add_argument("--manual-audit", default="runs/remediation/review-audit-empty-v2.json")
    parser.add_argument("--coarse", default="runs/remediation/urfd-all70-fp32-v2/official-coarse-report.json")
    parser.add_argument("--rtsp", default="runs/remediation/linux-rtsp-fault-correlated.json")
    parser.add_argument("--linux-soak", default="runs/remediation/ubuntu-evidence/runs/linux-live-rtsp-1h")
    parser.add_argument("--windows-soak", default="runs/remediation/live-camera-1h")
    parser.add_argument("--windows-soak-report", help="explicit threshold report to use instead of the Windows run's historical soak.json")
    parser.add_argument("--soak-min-seconds", type=float, default=3600)
    parser.add_argument("--physical-usb")
    parser.add_argument("--linux-gpu")
    args = parser.parse_args(argv)
    if args.expected_precision_frames <= 0 or not math.isfinite(args.soak_min_seconds) or args.soak_min_seconds <= 0:
        parser.error("Expected frame count and soak minimum must be positive")
    evidence = Evidence(args.project_root)
    parts = dict(
        linux_cpu=linux_status(evidence.read(args.linux_acceptance)),
        linux_tests=test_status(evidence, args.linux_tests), windows_tests=test_status(evidence, args.windows_tests),
        strict_precision=precision_status(evidence.read(args.precision), args.expected_precision_frames, evidence),
        independent_manual_data=manual_status(evidence.read(args.manual_audit)),
        real_video_accuracy=coarse_status(evidence.read(args.coarse)),
        linux_rtsp_protocol_fault=rtsp_status(evidence.read(args.rtsp), evidence),
        linux_hour_soak=soak_status(evidence, args.linux_soak, args.soak_min_seconds, True),
        windows_hour_soak=soak_status(evidence, args.windows_soak, args.soak_min_seconds, False, args.windows_soak_report),
        physical_usb=physical_status(evidence.read(args.physical_usb) if args.physical_usb else None, "external-usb"),
        linux_nvidia=linux_status(evidence.read(args.linux_gpu), gpu=True) if args.linux_gpu else
                     item("blocked", "No Linux NVIDIA GPU rebuild/precision/C++ evidence supplied"))
    controls = {evidence.record(path)["path"]: precision_status(evidence.read(path), args.expected_precision_frames, evidence)
                for path in args.precision_control}
    for name in ("docs/acceptance-status-2026-10-09.md", "runs/remediation/README.md", "tools/remediation_report.py"):
        if evidence.resolve(name).is_file():
            evidence.record(name)
    status_counts = Counter(part["status"] for part in parts.values())
    complete = status_counts["completed"] == len(parts)
    report = dict(schema="poseguard-remediation-manifest/v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  status="completed" if complete else "partial" if status_counts["completed"] or status_counts["partial"] else "blocked",
                  final_acceptance_complete=complete, status_counts=dict(status_counts), parts=parts,
                  precision_controls=controls, evidence=sorted(evidence.files.values(), key=lambda f: f["path"]),
                  limitation="Manifest statuses describe supplied evidence scopes; no missing hardware, annotation or future run is promoted to passed.")
    output = evidence.resolve(args.output)
    if output.relative_to(evidence.root).as_posix() in evidence.files:
        parser.error("--output must not overwrite an evidence source")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], final_acceptance_complete=complete, status_counts=dict(status_counts),
                          output=output.relative_to(evidence.root).as_posix()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
