import json
import threading
import numpy as np
import pytest
from poseguard.config import Config
from poseguard.pipeline import Capture, run_pipeline
from poseguard.types import Frame
from poseguard.backends import ONNXBackend
from poseguard.model_tools import compare_outputs
from poseguard.geometry import preprocess


@pytest.fixture
def constant_onnx(tmp_path):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    row = np.zeros((1, 56, 1), np.float32)
    row[0, :5, 0] = [32, 32, 20, 40, 0.9]
    row[0, 5:, 0] = np.tile([32, 32, 0.9], 17)
    graph = helper.make_graph([helper.make_node("Constant", [], ["output"],
                                               value=numpy_helper.from_array(row))], "pose",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 56, 1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    path = tmp_path / "test.onnx"
    onnx.save(model, path)
    return path


def test_bounded_queue_drops_oldest():
    capture = Capture(Config(queue_size=2), threading.Event())
    for i in range(10):
        capture.offer(Frame(np.zeros((2, 2, 3), np.uint8), i, i, i))
    assert capture.frames.qsize() == 2
    assert capture.dropped == 8
    assert capture.frames.get().frame_id == 8


def test_real_onnxruntime_contract_and_output_comparison(constant_onnx):
    backend = ONNXBackend(str(constant_onnx), 64)
    tensor, meta = preprocess(np.zeros((100, 200, 3), np.uint8), 64)
    output = backend.infer(tensor).output
    assert output.shape == (1, 56, 1)
    assert compare_outputs(output, output, meta, 0.01, 0.001, 0)["passed"]
    changed = output.copy()
    changed[0, 5, 0] += 20
    assert not compare_outputs(output, changed, meta, 0.01, 0.001, 0)["passed"]
    backend.close()


def test_onnx_backend_rejects_wrong_input_size(constant_onnx):
    with pytest.raises(ValueError, match="static"):
        ONNXBackend(str(constant_onnx), 640)


def test_demo_headless_generates_run_artifacts(tmp_path):
    output = tmp_path / "run"
    summary = run_pipeline(Config(show=False, duration=0.3, output=str(output)), demo=True)
    assert summary["status"] == "completed"
    assert summary["processed"] > 0
    assert not summary["capture_thread_alive"]
    assert summary["model_fps"] is None
    rows = (output / "predictions.jsonl").read_text().splitlines()
    assert len(rows) == summary["processed"] * 2
    assert (output / "frames.csv").exists()
    assert (output / "preview.jpg").exists()
    with pytest.raises(FileExistsError):
        run_pipeline(Config(show=False, duration=0.1, output=str(output)), demo=True)


def test_onnx_video_pipeline(constant_onnx, tmp_path):
    import cv2
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    assert writer.isOpened()
    for _ in range(6):
        writer.write(np.zeros((120, 160, 3), np.uint8))
    writer.release()
    summary = run_pipeline(Config(model=str(constant_onnx), imgsz=64, source=str(path),
                                  output=str(tmp_path / "onnx"), show=False))
    assert summary["processed"] == 6
    assert summary["model_fps"] > 0
    assert summary["status"] == "completed" and not summary["capture_thread_alive"]
    assert summary["timings"]["capture_ms"]["count"] == 6


@pytest.mark.parametrize("values", [{"imgsz": 639}, {"queue_size": 0}, {"duration": -1},
                                  {"fall_threshold": 1.5}, {"max_reconnects": 0.5}])
def test_invalid_config_fails_early(values):
    with pytest.raises(ValueError):
        Config(**values).validate()


def test_inference_failure_rebuilds_backend_and_finishes(constant_onnx, tmp_path, monkeypatch):
    import cv2
    from poseguard import pipeline
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    for _ in range(15):
        writer.write(np.zeros((120, 160, 3), np.uint8))
    writer.release()
    created = []
    class OnceFailingBackend:
        def __init__(self):
            self.closed = False
        def infer(self, tensor):
            raise RuntimeError("transient failure")
        def close(self):
            self.closed = True
    def factory(cfg):
        backend = OnceFailingBackend() if not created else ONNXBackend(cfg.model, cfg.imgsz)
        created.append(backend)
        return backend
    monkeypatch.setattr(pipeline, "make_backend", factory)
    summary = run_pipeline(Config(model=str(constant_onnx), imgsz=64, source=str(path),
                                  inference_retry_delay=0.01, output=str(tmp_path / "recovery"), show=False))
    assert summary["status"] == "completed" and summary["processed"] > 0
    assert summary["inference_retries"] == 1
    assert len(created) == 2 and created[0].closed


def test_source_unavailable_writes_failed_summary(tmp_path, monkeypatch):
    from poseguard import pipeline
    from poseguard import capture_worker
    class BrokenCamera:
        def open(self, *args):
            return False
        def set(self, *args):
            return False
        def isOpened(self):
            return False
        def release(self):
            pass
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", BrokenCamera)
    def unavailable(capture):
        raise RuntimeError("Source unavailable after 1 retries")
    monkeypatch.setattr(capture_worker, "run_live_capture", unavailable)
    class NoopBackend:
        def close(self):
            pass
    monkeypatch.setattr(pipeline, "make_backend", lambda cfg: NoopBackend())
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Source unavailable"):
        run_pipeline(Config(show=False, output=str(output), max_reconnects=1, reconnect_delay=0.01), demo=False)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed" and not summary["capture_thread_alive"]


def test_startup_native_error_still_writes_failed_summary(tmp_path):
    output = tmp_path / "startup-failed"
    with pytest.raises(Exception):
        run_pipeline(Config(model=str(tmp_path / "missing.onnx"), output=str(output), show=False))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["failure_phase"] == "startup"
    assert summary["processed"] == 0 and not summary["capture_thread_alive"]
    assert (output / "provenance.json").exists()


def test_rebuild_failure_retries_factory_until_success(constant_onnx, tmp_path, monkeypatch):
    import cv2
    from poseguard import pipeline
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    for _ in range(15): writer.write(np.zeros((120, 160, 3), np.uint8))
    writer.release()
    calls = []
    class BrokenBackend:
        def infer(self, tensor): raise RuntimeError("device lost")
        def close(self): raise RuntimeError("close also failed")
    def factory(cfg):
        calls.append(1)
        if len(calls) == 1: return BrokenBackend()
        if len(calls) == 2: raise RuntimeError("temporary recreation failure")
        return ONNXBackend(cfg.model, cfg.imgsz)
    monkeypatch.setattr(pipeline, "make_backend", factory)
    summary = run_pipeline(Config(model=str(constant_onnx), imgsz=64, source=str(path),
                                  inference_retry_delay=.01, output=str(tmp_path / "rebuild"), show=False))
    assert len(calls) == 3 and summary["inference_retries"] == 2
    assert summary["status"] == "completed" and summary["processed"] > 0


def test_visibility_threshold_crossing_cannot_hide_coordinate_error():
    from poseguard.geometry import Letterbox
    a = np.zeros((1, 56, 1), np.float32)
    a[0, :5, 0] = [550, 320, 150, 400, .9]
    a[0, 5:, 0] = np.tile([600, 320, .3001], 17)
    b = a.copy()
    b[0, 5::3, 0] += .6
    b[0, 7::3, 0] = .2999
    result = compare_outputs(a, b, Letterbox(1/6, 0, 140, 3840, 2160, 640), .1, .005, .001)
    assert not result["postprocess_passed"]
    assert result["matches"][0]["lost_visible_keypoints"] == 17
    assert result["matches"][0]["visible_keypoint_max_px"] > 3


def test_invisible_reference_is_insufficient_for_keypoint_validation(constant_onnx):
    backend = ONNXBackend(str(constant_onnx), 64)
    tensor, meta = preprocess(np.zeros((100, 200, 3), np.uint8), 64)
    output = backend.infer(tensor).output
    output[0, 7::3, 0] = .1
    result = compare_outputs(output, output, meta, .1, .005, .001)
    assert not result["postprocess_passed"]
    assert not result["matches"][0]["keypoint_check_sufficient"]
    backend.close()
