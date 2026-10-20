"""Exploratory three-way posture and sequence-event reports against official URFD labels."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from poseguard.annotations import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    truth = {}
    for line in Path(args.labels).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["timestamp"] is not None:
            truth.setdefault(row["video_id"], []).append(row)
    variants = {}
    for directory in sorted(Path(args.run_root).glob("*-v*")):
        if not (directory / "summary.json").exists():
            continue
        import yaml
        cfg = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        video = cfg["video_id"]
        if video not in truth:
            continue
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "completed":
            continue
        predictions = {}
        for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            predictions.setdefault(float(row["timestamp"]), []).append(row)
        variant = directory.name.rsplit("-v", 1)[0]
        group = variants.setdefault(variant, dict(expected=[], actual=[], videos=[], video_results=[], tp=0, fp=0, fn=0, delays=[], emitted_delays=[], inputs=[]))
        timestamps = np.asarray(sorted(predictions))
        used = set()
        label_map = {-1: "not_lying", 0: "transition", 1: "lying"}
        for row in truth[video]:
            expected = label_map[row["official_label"]]
            actual = "missing"
            if timestamps.size:
                index = int(np.argmin(np.abs(timestamps - row["timestamp"])))
                timestamp = float(timestamps[index])
                if abs(timestamp - row["timestamp"]) <= .02 and timestamp not in used:
                    used.add(timestamp)
                    # This dataset is single-actor; selection is explicitly a heuristic, not a spatial truth match.
                    candidates = predictions[timestamp]
                    target = max(candidates, key=lambda r: (r["bbox"][2]-r["bbox"][0])*(r["bbox"][3]-r["bbox"][1]))
                    actual = ("not_lying" if target["label"] in {"standing", "sitting"} else
                              "transition" if target["label"] == "falling" else target["label"])
            group["expected"].append(expected)
            group["actual"].append(actual)
        with (directory / "events.csv").open(newline="", encoding="utf-8") as stream:
            alerts = [r for r in csv.DictReader(stream) if r["event"] == "fall"]
        # ADL contains temporary posture label 0 during deliberate transitions;
        # it is a negative sequence, never an accidental-fall event.
        onset = [r["timestamp"] for r in truth[video] if r["official_label"] == 0] if video.startswith("fall-") else []
        terminal = [r["timestamp"] for r in truth[video] if r["official_label"] in {0, 1}]
        matched = False
        video_tp = video_fp = 0
        for alert in sorted(alerts, key=lambda r: float(r["timestamp"])):
            timestamp = float(alert["timestamp"])
            eligible = onset and not matched and min(onset) <= timestamp <= min(min(onset) + 5, max(terminal))
            if eligible:
                matched = True
                group["tp"] += 1
                video_tp += 1
                delay = timestamp - min(onset)
                group["delays"].append(delay)
                group["emitted_delays"].append(delay + float(alert["capture_to_emit_ms"]) / 1000)
            else:
                group["fp"] += 1
                video_fp += 1
        group["fn"] += int(bool(onset) and not matched)
        group["video_results"].append(dict(video_id=video, expected_fall=bool(onset),
                                          label_zero_frames=sum(r["official_label"] == 0 for r in truth[video]),
                                          alarms=[float(a["timestamp"]) for a in alerts],
                                          onset=min(onset) if onset else None,
                                          tp=video_tp, fp=video_fp, fn=int(bool(onset) and not matched)))
        group["videos"].append(video)
        group["inputs"].append(dict(predictions_sha256=file_sha256(directory / "predictions.jsonl"),
                                    events_sha256=file_sha256(directory / "events.csv")))
    reports = {}
    for variant, group in variants.items():
        expected, actual = group.pop("expected"), group.pop("actual")
        delay, emitted = group.pop("delays"), group.pop("emitted_delays")
        precision = group["tp"] / max(group["tp"] + group["fp"], 1)
        recall = group["tp"] / max(group["tp"] + group["fn"], 1)
        reports[variant] = dict(**group, samples=len(expected), missing=actual.count("missing"),
                               classification=classification_report(expected, actual, labels=["not_lying", "transition", "lying"], output_dict=True, zero_division=0),
                               confusion_labels=["not_lying", "transition", "lying", "unknown", "missing"],
                               confusion_matrix=confusion_matrix(expected, actual, labels=["not_lying", "transition", "lying", "unknown", "missing"]).tolist(),
                               events=dict(precision=precision, recall=recall, f1=2*precision*recall/max(precision+recall, 1e-9),
                                           media_delay_p50_seconds=float(np.percentile(delay, 50)) if delay else None,
                                           emitted_delay_p50_seconds=float(np.percentile(emitted, 50)) if emitted else None))
    report = dict(status="exploratory_not_final_test", final_acceptance_complete=False,
                  code_sha256=file_sha256(__file__), labels_sha256=file_sha256(args.labels), reports=reports,
                  coarse_label_mapping={"-1": "not_lying", "0": "transition", "1": "lying"},
                  scope="Official depth three-way posture labels aligned by source timestamps; sequence-level alarms",
                  limitations=["No training on these reports; these videos are exploration data",
                               "Fall sequences define positive events; ADL label 0 is a temporary posture, not an accidental-fall event",
                               "Standing/sitting are merged; official temporary transition is not manual accidental falling truth",
                               "Largest-person selection assumes one actor; no manual bbox/detection or identity metrics",
                               "RGB/depth alignment and event boundaries have not been independently reviewed",
                               "Other actors/false detections may contaminate selection; no person independence claim"])
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: dict(samples=v["samples"], tp=v["tp"], fp=v["fp"], fn=v["fn"]) for k,v in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
