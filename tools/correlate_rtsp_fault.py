"""Correlate Linux recovery frames with UTC upstream RTSP publisher fault timestamps."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path

from poseguard.provenance import artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--faults", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    health = [json.loads(line) for line in (run / "health.jsonl").read_text(encoding="utf-8").splitlines()]
    faults = [json.loads(line) for line in Path(args.faults).read_text(encoding="utf-8").splitlines()]
    with (run / "frames.csv").open(newline="", encoding="utf-8") as stream:
        frames = list(csv.DictReader(stream))
    origin = datetime.fromisoformat(provenance["created_utc"])
    stopped = next(row for row in faults if row["event"] == "camera_publisher_terminated")
    restored = next(row for row in faults if row["event"] == "camera_publisher_restarted")
    stopped_at, restored_at = datetime.fromisoformat(stopped["utc"]), datetime.fromisoformat(restored["utc"])
    failures = [dict(row, estimated_utc=(origin + timedelta(seconds=float(row["elapsed_seconds"]))).isoformat())
                for row in health if row["event"] in {"source_read_failed", "capture_watchdog_timeout"}]
    delivered_generations = sorted({int(row["generation"]) for row in frames})
    opens = [dict(row, estimated_utc=(origin + timedelta(seconds=float(row["elapsed_seconds"]))).isoformat())
             for row in health if row["event"] == "source_opened"]
    before = [row for row in opens if int(row["generation"]) in delivered_generations and
              datetime.fromisoformat(row["estimated_utc"]) < stopped_at]
    after = [row for row in opens if int(row["generation"]) in delivered_generations and
             datetime.fromisoformat(row["estimated_utc"]) >= restored_at]
    outage_seconds = (restored_at - stopped_at).total_seconds()
    first_reopen_delay = ((datetime.fromisoformat(after[0]["estimated_utc"]) - restored_at).total_seconds()
                         if after else None)
    report = {"status": "passed" if (summary["status"] == "completed" and before and after and failures and
              summary["reconnects"] > 0 and not summary["capture_thread_alive"] and not summary["capture_worker_alive"])
              else "failed", "scope": "real Windows integrated camera over RTSP into Ubuntu VMware; upstream publisher process fault",
              "physical_usb_fault": False, "remote_rtsp_camera_hardware_fault": False,
              "publisher_outage_seconds": outage_seconds, "estimated_reopen_after_restart_seconds": first_reopen_delay,
              "clock_note": "Linux health UTC estimated from provenance.created_utc + health elapsed_seconds; host/guest wall-clock offset and startup logging add uncertainty",
              "upstream_faults": faults, "source_open_events": opens, "failure_events": failures,
              "delivered_generations": delivered_generations, "summary": summary,
              "evidence": {name: artifact(run / name) for name in
                           ("summary.json", "provenance.json", "health.jsonl", "frames.csv")},
              "upstream_evidence": artifact(args.faults)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "publisher_outage_seconds": outage_seconds,
                      "estimated_reopen_after_restart_seconds": first_reopen_delay,
                      "delivered_generations": delivered_generations}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
