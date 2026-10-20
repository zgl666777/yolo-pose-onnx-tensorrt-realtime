import threading
import time
from pathlib import Path
import numpy as np
from poseguard.capture_worker import run_live_capture
from poseguard.pipeline import Capture
from poseguard.config import Config


def blocked_then_frames(source, timeout_ms, storage, state, lock, notices, stop):
    notices.put(("worker_ready", {}))
    marker = Path(source)
    if not marker.exists():
        marker.write_text("first worker entered blocked read")
        time.sleep(100)
    notices.put(("source_opened", {"fps": 30}))
    i = 0
    while not stop.wait(.03):
        with lock:
            np.frombuffer(storage, dtype=np.uint8, count=12)[:] = 7
            i += 1
            state[:] = (i, 2, 2, time.perf_counter_ns(), .1)


def always_blocked(source, timeout_ms, storage, state, lock, notices, stop):
    notices.put(("worker_ready", {}))
    time.sleep(100)


def slow_open_then_frames(source, timeout_ms, storage, state, lock, notices, stop):
    notices.put(("worker_ready", {}))
    time.sleep(.5)
    notices.put(("source_opened", {"fps": 30}))
    sequence = 0
    while not stop.wait(.02):
        with lock:
            np.frombuffer(storage, dtype=np.uint8, count=12)[:] = 9
            sequence += 1
            state[:] = (sequence, 2, 2, time.perf_counter_ns(), .1)


def unavailable_worker(source, timeout_ms, storage, state, lock, notices, stop):
    notices.put(("worker_ready", {}))
    notices.put(("source_read_failed", {"error": "device unavailable"}))


def test_real_child_hung_read_is_terminated_and_restarted(tmp_path):
    stop = threading.Event()
    cfg = Config(source=str(tmp_path / "marker"), read_timeout_ms=200, capture_startup_timeout=1,
                 capture_max_pixels=100, reconnect_delay=.01, queue_size=2)
    capture = Capture(cfg, stop)
    logs = []
    capture.log = lambda event, **details: logs.append(event)
    thread = threading.Thread(target=run_live_capture, args=(capture, blocked_then_frames))
    started = time.monotonic()
    thread.start()
    frame = capture.frames.get(timeout=12)
    assert frame.generation == 1 and np.all(frame.image == 7)
    assert capture.reconnections == 1 and "capture_watchdog_timeout" in logs
    stop.set()
    thread.join(timeout=3)
    assert not thread.is_alive() and not capture.worker.is_alive()
    assert time.monotonic() - started < 15


def test_camera_open_uses_startup_deadline_until_source_is_opened():
    stop = threading.Event()
    cfg = Config(read_timeout_ms=100, capture_startup_timeout=3,
                 capture_max_pixels=100, reconnect_delay=.01)
    capture = Capture(cfg, stop)
    logs = []
    capture.log = lambda event, **details: logs.append(event)
    thread = threading.Thread(target=run_live_capture, args=(capture, slow_open_then_frames))
    thread.start()
    frame = capture.frames.get(timeout=8)
    assert frame.generation == 0 and np.all(frame.image == 9)
    assert capture.reconnections == 0 and "capture_watchdog_timeout" not in logs
    stop.set()
    thread.join(timeout=3)
    assert not thread.is_alive() and not capture.worker.is_alive()
    assert "capture_worker_stopped" in logs


def test_shutdown_terminates_blocked_native_worker(tmp_path):
    stop = threading.Event()
    capture = Capture(Config(read_timeout_ms=10000, capture_max_pixels=100), stop)
    thread = threading.Thread(target=run_live_capture, args=(capture, always_blocked))
    thread.start()
    deadline = time.monotonic() + 5
    while capture.worker is None and time.monotonic() < deadline:
        time.sleep(.01)
    stop.set()
    thread.join(timeout=3)
    assert not thread.is_alive() and not capture.worker.is_alive()


def test_persistently_unavailable_device_exhausts_bounded_retries():
    stop = threading.Event()
    capture = Capture(Config(max_reconnects=1, reconnect_delay=.01,
                             capture_startup_timeout=3, capture_max_pixels=100), stop)
    logs, errors = [], []
    capture.log = lambda event, **details: logs.append((event, details))

    def invoke():
        try:
            run_live_capture(capture, unavailable_worker)
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive() and not capture.worker.is_alive()
    assert capture.reconnections == 1
    assert errors and "Source unavailable after 1 retries" in errors[0]
    retries = [details for event, details in logs if event == "source_reconnect_scheduled"]
    assert len(retries) == 1 and retries[0]["generation"] == 1
