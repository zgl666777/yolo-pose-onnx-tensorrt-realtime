"""Isolate native camera calls; shared storage never waits on an IPC frame pipe."""
from __future__ import annotations
import multiprocessing as mp
import queue
import time
import numpy as np
import cv2
from .types import Frame


def camera_worker(source, timeout_ms, storage, state, lock, notices, stop):
    cap = cv2.VideoCapture()
    try:
        notices.put(("worker_ready", {}))
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms]
        opened = cap.open(source, cv2.CAP_FFMPEG, params) if isinstance(source, str) else cap.open(source)
        if not opened:
            raise RuntimeError("Camera open failed")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        notices.put(("source_opened", {"fps": fps if np.isfinite(fps) and fps > 0 else 30.0}))
        sequence = 0
        while not stop.is_set():
            before = time.perf_counter()
            ok, image = cap.read()
            captured = time.perf_counter_ns()
            if not ok or image is None:
                raise RuntimeError("Camera read failed")
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size > len(storage):
                raise RuntimeError("Camera image exceeds configured shared buffer or has invalid format")
            sequence += 1
            # Native driver calls occur outside this lock. Both sides acquire with a deadline.
            if lock.acquire(timeout=.1):
                try:
                    np.frombuffer(storage, dtype=np.uint8, count=image.size)[:] = image.ravel()
                    state[:] = (sequence, image.shape[0], image.shape[1], captured,
                                (time.perf_counter() - before) * 1000)
                finally:
                    lock.release()
    except Exception as exc:
        try:
            notices.put_nowait(("source_read_failed", {"error": f"{type(exc).__name__}: {exc}"}))
        except queue.Full:
            pass
    finally:
        cap.release()


def run_live_capture(capture, worker_target=camera_worker):
    cfg = capture.config
    context = mp.get_context("spawn")
    origin_ns, generation, retries = time.perf_counter_ns(), 0, 0
    while not capture.stop.is_set():
        storage = context.RawArray("B", cfg.capture_max_pixels * 3)
        state, lock = context.RawArray("d", 5), context.Lock()
        notices, child_stop = context.Queue(maxsize=8), context.Event()
        child = context.Process(target=worker_target, args=(capture.source, cfg.read_timeout_ms,
                                storage, state, lock, notices, child_stop), daemon=True)
        capture.worker = child
        last_progress, opened, last_sequence = time.monotonic(), False, 0
        stable_since = None
        child.start()
        reason = "Capture stopped"
        try:
            while not capture.stop.wait(.005):
                now = time.monotonic()
                while True:
                    try:
                        event, details = notices.get_nowait()
                    except queue.Empty:
                        break
                    if event == "worker_ready":
                        last_progress = now
                    elif event == "source_opened":
                        opened, last_progress = True, now
                        capture.fps = details.pop("fps")
                    capture.log(event, generation=generation, **details)
                packet = None
                if lock.acquire(timeout=.02):
                    try:
                        sequence, h, w, captured_ns, capture_ms = state[:]
                        if int(sequence) > last_sequence:
                            image = np.frombuffer(storage, dtype=np.uint8, count=int(h*w*3)).reshape(int(h), int(w), 3).copy()
                            packet = (int(sequence), image, int(captured_ns), capture_ms)
                    finally:
                        lock.release()
                if packet is not None:
                    sequence, image, captured_ns, capture_ms = packet
                    skipped = sequence - last_sequence - 1
                    capture.captured += skipped
                    capture.dropped += skipped
                    last_sequence, last_progress = sequence, now
                    if stable_since is None:
                        stable_since = now
                    if now - stable_since >= 5:
                        retries = 0
                    capture.offer(Frame(image, capture.captured, (captured_ns - origin_ns) / 1e9,
                                        captured_ns, generation, capture_ms))
                deadline = cfg.read_timeout_ms / 1000 if opened else cfg.capture_startup_timeout
                if now - last_progress > deadline:
                    reason = "Capture watchdog timed out"
                    capture.log("capture_watchdog_timeout", generation=generation, worker_pid=child.pid)
                    break
                if not child.is_alive():
                    reason = f"Capture worker exited ({child.exitcode})"
                    break
        finally:
            child_stop.set()
            child.join(timeout=.2)
            if child.is_alive():
                child.terminate()
                child.join(timeout=1)
            if child.is_alive():
                child.kill()
                child.join(timeout=1)
            notices.cancel_join_thread()
            notices.close()
        if child.is_alive():
            raise RuntimeError("Capture worker could not be terminated")
        capture.log("capture_worker_stopped", generation=generation, worker_pid=child.pid,
                    worker_exit_code=child.exitcode, reason=reason)
        if capture.stop.is_set():
            break
        retries += 1
        if retries > cfg.max_reconnects:
            raise RuntimeError(f"Source unavailable after {cfg.max_reconnects} retries: {reason}")
        capture.reconnections += 1
        generation += 1
        capture.log("source_reconnect_scheduled", generation=generation, retry=retries,
                    delay_seconds=cfg.reconnect_delay, reason=reason)
        # Purge all old-generation frames before the new source can produce anything.
        while True:
            try:
                capture.frames.get_nowait()
                capture.dropped += 1
            except queue.Empty:
                break
        if capture.stop.wait(cfg.reconnect_delay):
            break
