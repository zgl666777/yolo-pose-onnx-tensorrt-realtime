import csv
import importlib.util
import json
from pathlib import Path

import pytest


def load_soak_report():
    source = Path(__file__).resolve().parents[1] / "tools" / "soak_report.py"
    spec = importlib.util.spec_from_file_location("soak_report_test_module", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(tmp_path, **overrides):
    run = tmp_path / "run"
    run.mkdir()
    summary = dict(status="completed", elapsed_seconds=3601, processed=4, captured=5,
                   dropped_queue=1, dropped_stale=0, system_fps=.001, reconnects=0, failures=0,
                   error=None, capture_error=None, cleanup_errors=[], capture_thread_alive=False,
                   capture_worker_alive=False)
    summary.update(overrides)
    (run / "summary.json").write_text(json.dumps(summary))
    with (run / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "latency_ms", "rss_mb"])
        writer.writeheader()
        for timestamp, rss in [(0, 100), (1200, 102), (2400, 104), (3600, 110)]:
            writer.writerow(dict(timestamp=timestamp, latency_ms=20, rss_mb=rss))
    return run


def test_default_report_remains_descriptive_even_when_run_failed(tmp_path):
    module = load_soak_report()
    run = write_run(tmp_path, status="failed", failures=2)
    output = tmp_path / "report.json"
    assert module.main(["--run", str(run), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["acceptance"]["status"] == "descriptive"
    assert report["acceptance"]["passed"] is None
    assert report["acceptance"]["checks"]["run_completed"]["passed"] is False
    assert "capture child RSS was not recorded" in report["rss_scope"]


def test_short_run_cannot_pass_explicit_one_hour_gate(tmp_path):
    module = load_soak_report()
    run = write_run(tmp_path, elapsed_seconds=60)
    output = tmp_path / "report.json"
    assert module.main(["--run", str(run), "--output", str(output), "--min-seconds", "3600"]) == 1
    report = json.loads(output.read_text())
    assert report["acceptance"]["passed"] is False
    assert report["acceptance"]["checks"]["min_seconds"] == dict(actual=60, expected=3600, passed=False)


@pytest.mark.parametrize("overrides,check", [
    ({"status": "failed"}, "run_completed"),
    ({"failures": 1}, "no_inference_failures"),
    ({"error": "runtime fault"}, "no_run_error"),
    ({"capture_error": "read fault"}, "no_capture_error"),
    ({"cleanup_errors": ["worker could not stop"]}, "no_cleanup_errors"),
    ({"capture_thread_alive": True}, "capture_thread_alive_false"),
    ({"capture_worker_alive": True}, "capture_worker_alive_false"),
    ({"capture_worker_alive": None}, "capture_worker_alive_false"),
    ({"exit_code": 1}, "recorded_exit_codes_zero"),
    ({"capture_worker_exit_code": -9}, "recorded_exit_codes_zero"),
    ({"processed": 5}, "frame_ledger_complete"),
])
def test_failure_or_unclean_shutdown_cannot_pass(tmp_path, overrides, check):
    module = load_soak_report()
    run = write_run(tmp_path, **overrides)
    output = tmp_path / "report.json"
    assert module.main(["--run", str(run), "--output", str(output), "--min-seconds", "3600"]) == 1
    report = json.loads(output.read_text())
    assert report["acceptance"]["checks"][check]["passed"] is False


def test_explicit_growth_and_drop_limits_pass_or_fail(tmp_path):
    module = load_soak_report()
    run = write_run(tmp_path)
    output = tmp_path / "report.json"
    command = ["--run", str(run), "--output", str(output), "--min-seconds", "3600",
               "--max-rss-growth-mb", "10", "--max-frame-gap-seconds", "1200", "--max-drop-ratio", ".2"]
    assert module.main(command) == 0
    report = json.loads(output.read_text())
    assert report["acceptance"]["passed"] is True
    assert report["acceptance"]["checks"]["recorded_exit_codes_zero"]["actual"] == {}
    command[-1] = ".19"
    assert module.main(command) == 1
    command[-1] = ".2"
    command[command.index("--max-rss-growth-mb") + 1] = "9"
    assert module.main(command) == 1


@pytest.mark.parametrize("option,value", [("--min-seconds", "0"), ("--min-seconds", "nan"),
                                           ("--max-rss-growth-mb", "-1"), ("--max-drop-ratio", "1.1"),
                                           ("--max-frame-gap-seconds", "0"), ("--max-frame-gap-seconds", "nan")])
def test_invalid_acceptance_thresholds_are_rejected(tmp_path, option, value):
    module = load_soak_report()
    with pytest.raises(SystemExit) as exc:
        module.main(["--run", str(tmp_path), "--output", str(tmp_path / "report.json"), option, value])
    assert exc.value.code == 2


def write_frame_ledger(run, timestamps):
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(processed=len(timestamps), captured=len(timestamps), dropped_queue=0, dropped_stale=0)
    summary_path.write_text(json.dumps(summary))
    with (run / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "latency_ms", "rss_mb"])
        writer.writeheader()
        for timestamp in timestamps:
            writer.writerow(dict(timestamp=timestamp, latency_ms=20, rss_mb=100))


def test_two_frames_an_hour_apart_cannot_pass_default_hour_gate(tmp_path):
    module = load_soak_report()
    run = write_run(tmp_path)
    write_frame_ledger(run, [0, 3600])
    output = tmp_path / "report.json"
    assert module.main(["--run", str(run), "--output", str(output), "--min-seconds", "3600"]) == 1
    report = json.loads(output.read_text())
    assert report["acceptance"]["checks"]["frame_time_coverage"]["passed"] is True
    assert report["acceptance"]["checks"]["max_frame_gap_seconds"]["passed"] is False


def test_hour_gate_allows_up_to_five_seconds_of_startup_but_requires_delivery(tmp_path):
    module = load_soak_report()
    run = write_run(tmp_path)
    write_frame_ledger(run, list(range(3, 3601)))
    output = tmp_path / "report.json"
    assert module.main(["--run", str(run), "--output", str(output), "--min-seconds", "3600"]) == 0
    report = json.loads(output.read_text())
    assert report["frame_time_span_seconds"] == 3597
    assert report["acceptance"]["checks"]["frame_time_coverage"]["expected"] == 3595
    write_frame_ledger(run, list(range(20, 3601)))
    assert module.main(["--run", str(run), "--output", str(output), "--min-seconds", "3600"]) == 1


def test_increasing_elapsed_runtime_without_frames_does_not_pass_hour_gate(tmp_path):
    module = load_soak_report()
    run = write_run(tmp_path)
    write_frame_ledger(run, [3599])
    output = tmp_path / "report.json"
    assert module.main(["--run", str(run), "--output", str(output), "--min-seconds", "3600"]) == 1
    report = json.loads(output.read_text())
    assert report["acceptance"]["checks"]["min_seconds"]["passed"] is True
    assert report["acceptance"]["checks"]["frame_time_coverage"]["passed"] is False


def test_nonmonotonic_frame_ledger_cannot_pass_explicit_soak_gate(tmp_path):
    module = load_soak_report()
    run = write_run(tmp_path)
    write_frame_ledger(run, [0, 1, 0])
    output = tmp_path / "report.json"
    assert module.main(["--run", str(run), "--output", str(output), "--max-drop-ratio", "1"]) == 1
    report = json.loads(output.read_text())
    assert report["acceptance"]["checks"]["frame_timestamps_monotonic"]["passed"] is False
