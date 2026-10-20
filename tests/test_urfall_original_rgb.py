import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest
import numpy as np
from poseguard.config import Config
from poseguard.image_io import write_image
from poseguard.types import Inference


def rgb_module():
    path = Path(__file__).resolve().parents[1] / "tools/download_urfall_rgb.py"
    spec = importlib.util.spec_from_file_location("download_urfall_rgb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_module():
    path = Path(__file__).resolve().parents[1] / "tools/replay_urfall_rgb.py"
    spec = importlib.util.spec_from_file_location("replay_urfall_rgb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["../outside.png", "/abs.png", "C:/Windows/file.png", "dir/../../escape.png", "..\\escape.png"])
def test_original_rgb_archive_rejects_escaping_paths(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        rgb_module().safe_member_path(tmp_path, zipfile.ZipInfo(name))


def test_original_rgb_archive_rejects_symlink_and_accepts_regular_image(tmp_path):
    module = rgb_module()
    link = zipfile.ZipInfo("link.png")
    link.external_attr = 0o120777 << 16
    with pytest.raises(ValueError, match="symlink"):
        module.safe_member_path(tmp_path, link)
    assert module.safe_member_path(tmp_path, zipfile.ZipInfo("rgb/00001.png")) == (tmp_path / "rgb/00001.png").resolve()


def test_original_rgb_download_limits_sequence_names_before_filesystem_or_network(tmp_path):
    with pytest.raises(ValueError, match="fixed URFD"):
        rgb_module().download_sequence("../../secret", tmp_path / "output", tmp_path)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("parallel,max_seconds", [(0, 1), (1.5, 1), (1, 0), (1, float("nan"))])
def test_original_rgb_download_requires_bounded_timeout_and_parallel_count(tmp_path, parallel, max_seconds):
    with pytest.raises(ValueError):
        rgb_module().download_archive("unused", tmp_path / "unused", parallel, max_seconds)


@pytest.mark.parametrize("content_range,length,body,error", [
    ("bytes 1-3/4", "4", b"1234", "Content-Range"),
    ("bytes 0-3/5", "4", b"1234", "Content-Range"),
    ("bytes 0-3/4", "5", b"1234", "Content-Length"),
    ("bytes 0-3/4", "4", b"12345", "span")])
def test_original_rgb_ranges_validate_headers_and_received_span(tmp_path, monkeypatch, content_range, length, body, error):
    module = rgb_module()

    class Response(io.BytesIO):
        def __init__(self, body, headers, status=200):
            super().__init__(body)
            self.headers, self.status = headers, status

    def request(req, timeout):
        if req.get_method() == "HEAD":
            return Response(b"", {"Content-Length": "4"})
        return Response(body, {"Content-Range": content_range, "Content-Length": length}, 206)

    monkeypatch.setattr(module, "urlopen", request)
    with pytest.raises(ValueError, match=error):
        module.download_archive("https://example.test/clip.zip", tmp_path / "clip.zip", parallel=2)


def test_original_rgb_head_rejects_nonpositive_archive_size(tmp_path, monkeypatch):
    module = rgb_module()

    class Response(io.BytesIO):
        headers = {"Content-Length": "0"}

    monkeypatch.setattr(module, "urlopen", lambda request, timeout: Response())
    with pytest.raises(ValueError, match="limit"):
        module.download_archive("https://example.test/clip.zip", tmp_path / "clip.zip")


def test_original_rgb_replay_preserves_source_timestamps_and_resolution_scope(tmp_path):
    image = tmp_path / "frame.png"
    write_image(image, np.zeros((480, 640, 3), dtype=np.uint8))
    tasks = [dict(image=str(image), frame_id=i, source_timestamp=t) for i, t in enumerate([0, .041])]
    raw = np.zeros((1, 56, 1), dtype=np.float32)
    raw[0, :5, 0] = [320, 320, 100, 200, .9]
    raw[0, 5:, 0] = np.tile([320, 320, .9], 17)

    class Backend:
        calls = 0

        def infer(self, tensor):
            self.calls += 1
            return Inference(raw)

    backend = Backend()
    report = replay_module().replay_tasks(tasks, tmp_path / "run", Config(video_id="fall-17-cam0"), backend,
                                          "original_full_resolution_RGB")
    assert report["frames"] == backend.calls == 2
    assert report["media_scope"] == "original_full_resolution_RGB"
    assert report["final_acceptance_complete"] is False
    rows = [json.loads(line) for line in (tmp_path / "run/predictions.jsonl").read_text().splitlines()]
    assert [r["timestamp"] for r in rows] == [0, .041]
