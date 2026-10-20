from __future__ import annotations

import csv
import json
import queue
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import psutil
from .backends import make_backend
from .demo import demo_frame
from .features import LearnedClassifier, RuleClassifier
from .geometry import preprocess, decode
from .metrics import Metrics
from .tracking import Tracker, TemporalDetector
from .types import Frame
from .provenance import run_provenance
from .telemetry import GpuTelemetry
from .image_io import write_image

SKELETON = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16))


class Capture(threading.Thread):
    def __init__(self, config, stop: threading.Event, demo=False):
        super().__init__(name="poseguard-capture", daemon=True)
        self.config, self.stop, self.demo = config, stop, demo
        self.frames: queue.Queue[Frame] = queue.Queue(maxsize=config.queue_size)
        self.finished = threading.Event()
        self.captured = self.dropped = self.reconnections = 0
        self.error: str | None = None
        self.source = int(config.source) if str(config.source).isdigit() else config.source
        self.is_file = isinstance(self.source, str) and Path(self.source).is_file()
        self.fps = 30.0
        self.log = lambda event, **details: None
        self.worker = None

    def offer(self, frame):
        self.captured += 1
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            try:
                self.frames.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.frames.put_nowait(frame)

    def run(self):
        cap = None
        cfg = self.config
        origin = time.monotonic()
        retries, generation = 0, 0
        stable_since = None
        try:
            if not self.demo and not self.is_file:
                from .capture_worker import run_live_capture
                run_live_capture(self)
                return
            if self.demo:
                index = 0
                while not self.stop.is_set():
                    due = origin + index / self.fps
                    if self.stop.wait(max(0, due - time.monotonic())):
                        break
                    mark = time.perf_counter()
                    image, _ = demo_frame(index / self.fps)
                    capture_ms = (time.perf_counter() - mark) * 1000
                    self.offer(Frame(image, index, index / self.fps, time.perf_counter_ns(), capture_ms=capture_ms))
                    index += 1
                return
            file_index = 0
            file_origin = None
            previous_file_ts = -1.0
            while not self.stop.is_set():
                if cap is None:
                    cap = cv2.VideoCapture()
                    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, cfg.read_timeout_ms,
                              cv2.CAP_PROP_READ_TIMEOUT_MSEC, cfg.read_timeout_ms]
                    if isinstance(self.source, str):
                        opened = cap.open(self.source, cv2.CAP_FFMPEG, params)
                    else:
                        opened = cap.open(self.source)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if opened:
                        self.log("source_opened", generation=generation)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        self.fps = fps if np.isfinite(fps) and fps > 0 else 30.0
                        file_origin = time.monotonic()
                mark = time.perf_counter()
                ok, image = cap.read() if cap.isOpened() else (False, None)
                capture_ms = (time.perf_counter() - mark) * 1000
                captured_ns = time.perf_counter_ns()
                if not ok:
                    self.log("source_read_failed", generation=generation, was_open=cap.isOpened())
                    was_open = cap.isOpened()
                    cap.release()
                    cap = None
                    if self.is_file:
                        if not was_open or file_index == 0:
                            raise RuntimeError(f"Cannot read video: {self.source}")
                        break
                    retries += 1
                    stable_since = None
                    if retries > cfg.max_reconnects:
                        raise RuntimeError(f"Source unavailable after {cfg.max_reconnects} retries")
                    self.reconnections += 1
                    generation += 1
                    if self.stop.wait(cfg.reconnect_delay):
                        break
                    continue
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since > 5:
                    retries = 0
                if self.is_file:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if not np.isfinite(timestamp) or timestamp <= previous_file_ts:
                        timestamp = file_index / self.fps
                    timestamp = max(timestamp, previous_file_ts + 1 / self.fps)
                    if self.stop.wait(max(0, file_origin + timestamp - time.monotonic())):
                        break
                    # For replay, latency starts at scheduled delivery, not ahead-of-time decoding.
                    captured_ns = time.perf_counter_ns()
                    previous_file_ts = timestamp
                    file_index += 1
                else:
                    timestamp = time.monotonic() - origin
                self.offer(Frame(image, self.captured, timestamp, captured_ns, generation, capture_ms))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if cap is not None:
                cap.release()
            self.finished.set()


def draw_tracks(image, tracks, config, fps, latency_ms, dropped):
    for track in tracks:
        color = (72, 84, 238) if track.alert_active else (126, 218, 85)
        box = track.detection.bbox.astype(int)
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, 2)
        kps = track.detection.keypoints
        for a, b in SKELETON:
            if min(kps[a, 2], kps[b, 2]) >= config.keypoint_confidence:
                cv2.line(image, tuple(kps[a, :2].astype(int)), tuple(kps[b, :2].astype(int)),
                         (227, 199, 92), 2, cv2.LINE_AA)
        for x, y, confidence in kps:
            if confidence >= config.keypoint_confidence:
                cv2.circle(image, (int(x), int(y)), 3, (228, 235, 229), -1, cv2.LINE_AA)
        label = f"#{track.id} {track.label} {track.confidence:.2f}" + (" / FALL" if track.alert_active else "")
        x = max(0, min(int(box[0]), image.shape[1] - 250))
        y = max(56, int(box[1]) - 10)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, image.shape[0] - 42), (image.shape[1], image.shape[0]), (22, 26, 24), -1)
    text = f"FPS {fps:.1f}   latency {latency_ms:.1f}ms   dropped {dropped}   people {len(tracks)}"
    cv2.putText(image, text, (20, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (206, 217, 212), 1, cv2.LINE_AA)
    return image


def run_pipeline(config, demo: bool = False) -> dict:
    cfg = config.validate()
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or (output / "predictions.jsonl").exists():
        raise FileExistsError(f"Output already contains a run: {output}; choose another --output")
    backend = None
    stop = threading.Event()
    capture = Capture(cfg, stop, demo)
    tracker, temporal = Tracker(cfg.track_ttl), TemporalDetector(cfg)
    metrics, process = Metrics(), psutil.Process()
    writer = None
    processed = stale = failures = alerts = 0
    inference_retries = consecutive_inference_failures = 0
    generation = 0
    latest_shape = None
    start = time.perf_counter()
    status, error = "completed", None
    phase, failure_phase = "startup", None
    health_file = None
    telemetry = None
    health_lock = threading.Lock()
    def health(event, **details):
        if health_file is not None:
            with health_lock:
                health_file.write(json.dumps(dict(event=event, elapsed_seconds=time.perf_counter() - start,
                                                  **details)) + "\n")
                health_file.flush()
    try:
        cfg.save(output / "config.yaml")
        (output / "provenance.json").write_text(json.dumps(run_provenance(output / "config.yaml", cfg, demo),
                                                          indent=2), encoding="utf-8")
        health_file = (output / "health.jsonl").open("w", encoding="utf-8")
        capture.log = health
        classifier = LearnedClassifier(cfg.classifier) if cfg.classifier else RuleClassifier()
        backend = None if demo else make_backend(cfg)
        if not demo and cfg.backend in {"trt", "cpp"}:
            telemetry = GpuTelemetry(output / "gpu.csv")
            telemetry.start()
        health("startup_complete")
        phase = "running"
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions, \
                (output / "events.csv").open("w", newline="", encoding="utf-8") as event_file, \
                (output / "event-delivery.jsonl").open("w", encoding="utf-8") as delivery_file, \
                (output / "frames.csv").open("w", newline="", encoding="utf-8") as frame_file:
            events = csv.DictWriter(event_file, fieldnames=["video_id", "track_id", "timestamp", "event",
                                                            "event_id", "capture_to_emit_ms", "emitted_unix_ns",
                                                            "capture_to_emit_scope"])
            events.writeheader()
            event_index = 0
            frame_fields = ["video_id", "frame_id", "timestamp", "generation", "capture_ms", "preprocess_ms", "backend_ms", "h2d_ms", "gpu_ms",
                            "d2h_ms", "postprocess_ms", "classify_ms", "render_ms", "output_ms",
                            "queue_ms", "latency_ms", "rss_mb", "people"]
            frame_log = csv.DictWriter(frame_file, fieldnames=frame_fields)
            frame_log.writeheader()
            capture.start()
            run_start = time.perf_counter()
            while True:
                if cfg.duration and time.perf_counter() - run_start >= cfg.duration:
                    break
                if cfg.max_frames and processed >= cfg.max_frames:
                    break
                try:
                    frame = capture.frames.get(timeout=0.1)
                except queue.Empty:
                    if capture.finished.is_set():
                        if capture.error:
                            raise RuntimeError(capture.error)
                        break
                    continue
                queue_ms = (time.perf_counter_ns() - frame.captured_ns) / 1e6
                if queue_ms > cfg.max_frame_age * 1000:
                    stale += 1
                    continue
                if frame.generation != generation:
                    tracker.reset()
                    generation = frame.generation
                if cfg.source_roi:
                    x1,y1,x2,y2 = cfg.source_roi
                    if x2 > frame.image.shape[1] or y2 > frame.image.shape[0]:
                        raise ValueError("source_roi exceeds captured image dimensions")
                    frame.image = frame.image[y1:y2,x1:x2].copy()
                timings = {"capture_ms": frame.capture_ms}
                timings["queue_ms"] = queue_ms
                try:
                    mark = time.perf_counter()
                    if demo:
                        _, detections = demo_frame(frame.timestamp)
                    else:
                        try:
                            tensor, meta = preprocess(frame.image, cfg.imgsz)
                            timings["preprocess_ms"] = (time.perf_counter() - mark) * 1000
                            result = backend.infer(tensor)
                            timings.update(result.timings)
                            mark = time.perf_counter()
                            detections = decode(result.output, meta, cfg.confidence, cfg.iou)
                            timings["postprocess_ms"] = (time.perf_counter() - mark) * 1000
                            consecutive_inference_failures = 0
                        except Exception as exc:
                            failures += 1
                            health("inference_failed", error=f"{type(exc).__name__}: {exc}")
                            if backend is not None:
                                try:
                                    backend.close()
                                except Exception as close_exc:
                                    health("recovery_close_failed", error=str(close_exc))
                            backend = None
                            tracker.reset()
                            # A failed factory consumes one attempt, then retries just like a failed inference.
                            while backend is None:
                                if consecutive_inference_failures >= cfg.max_inference_retries:
                                    raise RuntimeError("Inference recovery exhausted") from exc
                                consecutive_inference_failures += 1
                                inference_retries += 1
                                stop.wait(cfg.inference_retry_delay)
                                try:
                                    backend = make_backend(cfg)
                                    health("backend_rebuilt", attempt=consecutive_inference_failures)
                                except Exception as rebuild_exc:
                                    failures += 1
                                    health("backend_rebuild_failed", attempt=consecutive_inference_failures,
                                           error=f"{type(rebuild_exc).__name__}: {rebuild_exc}")
                            continue
                    mark = time.perf_counter()
                    tracks = tracker.update(detections, frame.timestamp)
                    records = []
                    for track in tracks:
                        probs = classifier.predict(track.detection, cfg.keypoint_confidence)
                        event = temporal.update(track, probs, frame.timestamp)
                        if event:
                            event_index += 1
                            events.writerow(dict(video_id=cfg.video_id, track_id=track.id,
                                                 timestamp=f"{frame.timestamp:.6f}", event=event,
                                                 event_id=event_index, capture_to_emit_scope="write_call",
                                                 capture_to_emit_ms=(time.perf_counter_ns() - frame.captured_ns) / 1e6,
                                                 emitted_unix_ns=time.time_ns()))
                            event_file.flush()
                            delivery_file.write(json.dumps(dict(event_id=event_index,
                                capture_to_emit_ms=(time.perf_counter_ns() - frame.captured_ns) / 1e6,
                                emitted_unix_ns=time.time_ns(), capture_to_emit_scope="events_csv_flush")) + "\n")
                            delivery_file.flush()
                            if event == "fall":
                                alerts += 1
                                print(f"FALL video={cfg.video_id} track={track.id} t={frame.timestamp:.3f}", flush=True)
                        records.append(dict(video_id=cfg.video_id, frame_id=frame.frame_id,
                                            timestamp=frame.timestamp, track_id=track.id,
                                            bbox=track.detection.bbox.tolist(),
                                            keypoints=track.detection.keypoints.tolist(),
                                            label=track.label, confidence=track.confidence,
                                            probabilities=probs, motion_features=track.motion_features,
                                            alert=track.alert_active))
                    timings["classify_ms"] = (time.perf_counter() - mark) * 1000
                    mark = time.perf_counter()
                    age = (time.perf_counter_ns() - frame.captured_ns) / 1e6
                    display = draw_tracks(frame.image.copy(), tracks, cfg,
                                          processed / max(time.perf_counter() - start, 1e-6),
                                          age, capture.dropped + stale)
                    timings["render_ms"] = (time.perf_counter() - mark) * 1000
                    mark = time.perf_counter()
                    if cfg.save_video:
                        shape = (display.shape[1], display.shape[0])
                        if writer is None:
                            latest_shape = shape
                            writer = cv2.VideoWriter(str(output / "annotated.mp4"),
                                                     cv2.VideoWriter_fourcc(*"mp4v"), capture.fps, shape)
                            if not writer.isOpened():
                                raise RuntimeError("Cannot open mp4v output writer")
                        if shape != latest_shape:
                            display = cv2.resize(display, latest_shape)
                        writer.write(display)
                    if cfg.show:
                        cv2.imshow("PoseGuard", display)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            stop.set()
                    if processed == 0:
                        write_image(output / "preview.jpg", display)
                    if any(r["alert"] for r in records) and not (output / "fall_preview.jpg").exists():
                        write_image(output / "fall_preview.jpg", display)
                    for record in records:
                        predictions.write(json.dumps(record) + "\n")
                    predictions.flush()
                    timings["output_ms"] = (time.perf_counter() - mark) * 1000
                    timings["latency_ms"] = (time.perf_counter_ns() - frame.captured_ns) / 1e6
                    rss = process.memory_info().rss / 1024 ** 2
                    frame_log.writerow(dict(video_id=cfg.video_id, frame_id=frame.frame_id, timestamp=frame.timestamp,
                                            generation=frame.generation,
                                            rss_mb=rss, people=len(tracks), **timings))
                    frame_file.flush()
                    metrics.add(**timings, rss_mb=rss)
                    processed += 1
                    if processed % 150 == 0:
                        print(f"frames={processed} dropped={capture.dropped + stale} rss={rss:.1f}MB", flush=True)
                    if stop.is_set():
                        break
                except Exception:
                    failures += 1
                    raise
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        failure_phase = phase
        health("run_failed", phase=phase, error=error)
        raise
    finally:
        stop.set()
        if capture.ident is not None:
            capture.join(timeout=max(cfg.read_timeout_ms / 1000 + 1, 3))
        if writer is not None:
            writer.release()
        if telemetry is not None:
            telemetry.stop.set()
            telemetry.join(timeout=5)
        cleanup_errors = []
        if cfg.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                cleanup_errors.append(str(exc))
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if cleanup_errors:
            status = "failed"
        if capture.is_alive():
            status = "failed"
            cleanup_errors.append("Capture thread did not stop within timeout")
        if capture.error:
            status = "failed"
            error = error or capture.error
            failure_phase = failure_phase or "capture"
        health("run_finished", status=status)
        if health_file is not None and not capture.is_alive():
            health_file.close()
            health_file = None
        elapsed = time.perf_counter() - start
        summary = dict(status=status, error=error, failure_phase=failure_phase,
                       cleanup_errors=cleanup_errors, synthetic=demo,
                       classifier="learned" if cfg.classifier else "untrained_rules",
                       processed=processed, captured=capture.captured,
                       dropped_queue=capture.dropped, dropped_stale=stale,
                       discarded_on_shutdown=capture.frames.qsize(),
                       reconnects=capture.reconnections, failures=failures, alerts=alerts,
                       inference_retries=inference_retries, capture_error=capture.error,
                       gpu_telemetry_error=telemetry.error if telemetry is not None else None,
                       gpu_memory_scope="whole device incl other applications; see gpu.csv",
                       capture_thread_alive=capture.is_alive(), elapsed_seconds=elapsed,
                       capture_worker_alive=capture.worker.is_alive() if capture.worker is not None else False,
                       system_fps=processed / max(elapsed, 1e-9), timings=metrics.report(),
                       percentile_scope="most recent 20000 processed frames",
                       video_timing="constant source FPS; dropped frames shorten saved video; use JSONL timestamps")
        summary["coordinate_space"] = dict(source_roi=cfg.source_roi, bbox_keypoints="cropped input pixels" if cfg.source_roi else "source pixels")
        if "backend_ms" in metrics.distributions and not demo:
            summary["model_fps"] = 1000 / max(metrics.distributions["backend_ms"].report()["mean"], 1e-9)
        else:
            summary["model_fps"] = None
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
