import importlib.util
import json
from pathlib import Path

from poseguard.annotations import write_jsonl


def test_adl_temporary_pose_zero_is_not_accidental_fall_event(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/report_urfall_coarse.py"
    spec = importlib.util.spec_from_file_location("report_urfall_coarse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = tmp_path / "labels.jsonl"
    write_jsonl(labels, [dict(video_id=video, timestamp=t, official_label=label)
                         for video in ("adl-01-cam0", "fall-01-cam0")
                         for t, label in ((0, 0), (1, 1))])
    for index, video in enumerate(("adl-01-cam0", "fall-01-cam0")):
        output = tmp_path / f"fp32-v{index}"
        output.mkdir()
        (output / "config.yaml").write_text(f"video_id: {video}\n", encoding="utf-8")
        (output / "summary.json").write_text('{"status": "completed"}', encoding="utf-8")
        write_jsonl(output / "predictions.jsonl", [dict(timestamp=0, label="lying", bbox=[0, 0, 2, 2])])
        (output / "events.csv").write_text("event,timestamp,capture_to_emit_ms\nfall,0.5,1\n", encoding="utf-8")
    report = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["report", "--labels", str(labels), "--run-root", str(tmp_path), "--output", str(report)])
    module.main()
    value = json.loads(report.read_text())["reports"]["fp32"]
    assert (value["tp"], value["fp"], value["fn"]) == (1, 1, 0)
    assert value["video_results"][0]["expected_fall"] is False
    assert "transition" in value["classification"] and "falling" not in value["classification"]
