"""Summarize a run, optionally enforce explicitly supplied soak thresholds."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def acceptance_checks(summary, report, thresholds):
    checks = {}

    def check(name, actual, expected, passed):
        checks[name] = dict(actual=actual, expected=expected, passed=bool(passed))

    check("run_completed", summary.get("status"), "completed", summary.get("status") == "completed")
    check("no_inference_failures", summary.get("failures"), 0, summary.get("failures") == 0)
    check("no_run_error", summary.get("error"), None, summary.get("error") is None)
    check("no_capture_error", summary.get("capture_error"), None, summary.get("capture_error") is None)
    check("no_cleanup_errors", summary.get("cleanup_errors"), [], summary.get("cleanup_errors") == [])
    for name in ("capture_thread_alive", "capture_worker_alive"):
        check(f"{name}_false", summary.get(name), False, summary.get(name) is False)
    exit_codes = {name: summary[name] for name in ("exit_code", "process_exit_code", "returncode",
                                                  "capture_worker_exit_code", "worker_exit_code")
                  if name in summary and summary[name] is not None}
    check("recorded_exit_codes_zero", exit_codes, "all recorded exit codes are zero",
          all(code == 0 for code in exit_codes.values()))
    check("frame_ledger_complete", summary.get("processed"), report["processed"],
          summary.get("processed") == report["processed"])
    check("finite_statistics", [report[k] for k in ("elapsed_seconds", "rss_change_mb", "drop_ratio")],
          "finite values", all(math.isfinite(report[k]) for k in ("elapsed_seconds", "rss_change_mb", "drop_ratio")))
    check("capture_count_consistent", summary.get("captured"), "captured >= processed > 0",
          isinstance(summary.get("captured"), (int, float)) and summary["captured"] >= report["processed"] > 0)
    if thresholds["min_seconds"] is not None:
        check("min_seconds", report["elapsed_seconds"], thresholds["min_seconds"],
              report["elapsed_seconds"] >= thresholds["min_seconds"])
        # A run's configured duration includes backend/capture startup. Permit
        # at most five seconds at the edges, but require sustained frame delivery.
        required_span = max(thresholds["min_seconds"] - 5, thresholds["min_seconds"] * .95)
        check("frame_time_coverage", report["frame_time_span_seconds"], required_span,
              report["frame_time_span_seconds"] >= required_span)
    max_gap = thresholds.get("max_frame_gap_seconds")
    if max_gap is not None:
        check("max_frame_gap_seconds", report["max_frame_gap_seconds"], max_gap,
              report["max_frame_gap_seconds"] <= max_gap)
    check("frame_timestamps_monotonic", report["frame_timestamps_monotonic"], True,
          report["frame_timestamps_monotonic"])
    if thresholds["max_rss_growth_mb"] is not None:
        check("max_rss_growth_mb", report["rss_change_mb"], thresholds["max_rss_growth_mb"],
              report["rss_change_mb"] <= thresholds["max_rss_growth_mb"])
    if thresholds["max_drop_ratio"] is not None:
        check("max_drop_ratio", report["drop_ratio"], thresholds["max_drop_ratio"],
              report["drop_ratio"] <= thresholds["max_drop_ratio"])
    enabled = any(value is not None for value in thresholds.values())
    passed = all(row["passed"] for row in checks.values()) if enabled else None
    return dict(enabled=enabled, status="passed" if passed is True else "failed" if passed is False else "descriptive",
                passed=passed, thresholds=thresholds, checks=checks,
                scope="Explicit soak thresholds and clean shutdown only; does not prove physical faults, GPU accuracy or absence of leaks.",
                exit_code_scope="Only exit codes present in summary.json are checked; historical child exit codes are not asserted.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-seconds", type=float)
    parser.add_argument("--max-rss-growth-mb", type=float)
    parser.add_argument("--max-drop-ratio", type=float)
    parser.add_argument("--max-frame-gap-seconds", type=float,
                        help="maximum consecutive processed-frame gap; defaults to 5s when --min-seconds is set")
    args = parser.parse_args(argv)
    for name in ("min_seconds", "max_rss_growth_mb", "max_drop_ratio", "max_frame_gap_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.min_seconds == 0:
        parser.error("--min-seconds must be positive")
    if args.max_drop_ratio is not None and args.max_drop_ratio > 1:
        parser.error("--max-drop-ratio must be in [0, 1]")
    if args.max_frame_gap_seconds == 0:
        parser.error("--max-frame-gap-seconds must be positive")
    if args.min_seconds is not None and args.max_frame_gap_seconds is None:
        args.max_frame_gap_seconds = 5.0
    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    with (run / "frames.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        parser.error("run has no processed frames")
    rss = np.array([float(row["rss_mb"]) for row in rows])
    latency = np.array([float(row["latency_ms"]) for row in rows])
    elapsed = np.array([float(row["timestamp"]) for row in rows])
    differences = np.diff(elapsed)
    segment = max(1, len(rows) // 10)
    dropped = summary["dropped_queue"] + summary["dropped_stale"]
    report = dict(status=summary["status"], elapsed_seconds=summary["elapsed_seconds"],
                  processed=len(rows), system_fps=summary["system_fps"],
                  frame_time_span_seconds=float(elapsed[-1] - elapsed[0]) if len(elapsed) > 1 else 0.0,
                  max_frame_gap_seconds=float(differences.max()) if differences.size else 0.0,
                  frame_timestamps_monotonic=bool(np.isfinite(elapsed).all() and (differences > 0).all()),
                  drop_ratio=dropped / max(summary["captured"], 1),
                  latency_p50_ms=float(np.percentile(latency, 50)),
                  latency_p95_ms=float(np.percentile(latency, 95)),
                  rss_peak_mb=float(rss.max()), rss_first_segment_mb=float(rss[:segment].mean()),
                  rss_last_segment_mb=float(rss[-segment:].mean()),
                  rss_change_mb=float(rss[-segment:].mean() - rss[:segment].mean()),
                  rss_linear_slope_mb_per_minute=float(np.polyfit(elapsed - elapsed[0], rss, 1)[0] * 60)
                  if len(rows) > 1 and elapsed[-1] > elapsed[0] else None,
                  reconnects=summary["reconnects"], failures=summary["failures"],
                  capture_thread_alive=summary["capture_thread_alive"],
                  capture_worker_alive=summary.get("capture_worker_alive"),
                  rss_scope="Main inference process RSS only; capture child RSS was not recorded and is not covered by the memory threshold.",
                  note="Measured memory change is descriptive, not proof of a memory leak; GPU memory is not included.")
    if (run / "gpu.csv").exists():
        with (run / "gpu.csv").open(newline="", encoding="utf-8") as stream:
            samples = list(csv.DictReader(stream))
        values = [float(r["memory_used_mb"]) for r in samples if r["memory_used_mb"].strip().isdigit()]
        report["gpu_memory_peak_mb"] = max(values) if values else None
        report["gpu_memory_scope"] = "Whole device; includes other applications, not process allocation"
        report["note"] = "Memory changes are descriptive and do not prove absence/presence of a leak."
    report["acceptance"] = acceptance_checks(summary, report, dict(
        min_seconds=args.min_seconds, max_rss_growth_mb=args.max_rss_growth_mb, max_drop_ratio=args.max_drop_ratio,
        max_frame_gap_seconds=args.max_frame_gap_seconds))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["acceptance"]["passed"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
