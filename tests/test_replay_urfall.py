import importlib.util
from pathlib import Path

import numpy as np
import pytest

from poseguard.config import Config
from poseguard.types import Inference


def replay_module():
    path = Path(__file__).resolve().parents[1] / "tools/replay_urfall.py"
    spec = importlib.util.spec_from_file_location("replay_urfall", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("second_source_ms", [42, None])
def test_sequential_replay_preserves_every_frame_source_timing_and_rgb_crop(tmp_path, monkeypatch, second_source_ms):
    module = replay_module()
    video = tmp_path / "fall-01-cam0.mp4"
    video.write_bytes(b"test media")
    (tmp_path / "fall-01-data.csv").write_text("1,0\n" +
         (f"2,{second_source_ms}\n" if second_source_ms is not None else ""), encoding="utf-8")
    image = np.zeros((240, 640, 3), dtype=np.uint8)
    image[:, 320:] = 210

    class Capture:
        remaining = 2

        def isOpened(self):
            return True

        def get(self, prop):
            return 2 if prop == module.cv2.CAP_PROP_FRAME_COUNT else 30

        def read(self):
            if not self.remaining:
                return False, None
            self.remaining -= 1
            return True, image

        def release(self):
            pass

    class Backend:
        calls = 0

        def infer(self, tensor):
            self.calls += 1
            assert tensor[0, 0, 320, 320] == pytest.approx(210 / 255)
            return Inference(np.zeros((1, 56, 1), dtype=np.float32), {"backend_ms": 1})

    monkeypatch.setattr(module.cv2, "VideoCapture", lambda path: Capture())
    backend = Backend()
    report = module.replay_video(video, tmp_path / "run", Config(video_id=video.stem), backend)
    assert backend.calls == report["frames"] == 2
    assert report["dropped_frames"] == 0
    assert report["scope"] == "exploratory_tuning_not_final_test"
    assert report["final_acceptance_complete"] is False
    assert report["missing_source_timestamps"] == int(second_source_ms is None)
    import csv
    with (tmp_path / "run/frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected_second = .042 if second_source_ms is not None else 1 / 30
    assert [float(r["timestamp"]) for r in rows] == pytest.approx([0, expected_second])
    assert float(rows[1]["playback_timestamp"]) == pytest.approx(1 / 30)


def test_sequential_replay_does_not_claim_completion_on_short_decode(tmp_path, monkeypatch):
    module = replay_module()
    video = tmp_path / "adl-01-cam0.mp4"
    video.write_bytes(b"test media")
    (tmp_path / "adl-01-data.csv").write_text("1,0\n", encoding="utf-8")

    class Capture:
        def isOpened(self):
            return True

        def get(self, prop):
            return 2 if prop == module.cv2.CAP_PROP_FRAME_COUNT else 30

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(module.cv2, "VideoCapture", lambda path: Capture())
    with pytest.raises(ValueError, match="Decoded 0/2"):
        module.replay_video(video, tmp_path / "run", Config(video_id=video.stem), object())
    assert not (tmp_path / "run/summary.json").exists()
