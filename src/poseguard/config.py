from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
import math
import yaml


@dataclass
class Config:
    backend: str = "onnx"
    model: str = "models/yolo11n-pose.onnx"
    classifier: str | None = None
    library: str | None = None
    source: str = "0"
    source_roi: list[int] | None = None
    video_id: str = "camera-0"
    imgsz: int = 640
    confidence: float = 0.35
    iou: float = 0.45
    keypoint_confidence: float = 0.3
    queue_size: int = 2
    max_frame_age: float = 0.5
    reconnect_delay: float = 1.0
    max_reconnects: int = 10
    max_inference_retries: int = 3
    inference_retry_delay: float = 0.25
    read_timeout_ms: int = 3000
    capture_startup_timeout: float = 15.0
    capture_max_pixels: int = 8294400
    track_ttl: float = 1.0
    window_seconds: float = 0.8
    fall_hold_seconds: float = 1.0
    transition_seconds: float = 3.0
    motion_window_seconds: float = 0.8
    min_fall_vertical_speed: float = 0.35
    min_fall_tilt_rate: float = 0.45
    min_fall_tilt_change: float = 0.2
    motion_evidence_hold_seconds: float = 1.5
    post_fall_posture_threshold: float = 0.35
    motion_fall_hold_seconds: float = 0.25
    min_fall_motion_strength: float = 0.55
    recovery_seconds: float = 2.0
    min_observations: int = 3
    fall_threshold: float = 0.65
    show: bool = True
    save_video: bool = False
    output: str = "runs/camera"
    duration: float = 0
    max_frames: int = 0
    ort_provider: str = "CPUExecutionProvider"

    def validate(self) -> "Config":
        if self.backend not in {"onnx", "trt", "cpp"}:
            raise ValueError("backend must be onnx, trt or cpp")
        if not isinstance(self.imgsz, int) or self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        for name in ("queue_size", "read_timeout_ms", "min_observations", "capture_max_pixels"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("confidence", "iou", "keypoint_confidence", "fall_threshold"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("max_frame_age", "reconnect_delay", "track_ttl", "window_seconds",
                     "fall_hold_seconds", "transition_seconds", "recovery_seconds", "inference_retry_delay",
                     "capture_startup_timeout", "motion_window_seconds", "min_fall_vertical_speed",
                     "min_fall_tilt_rate", "min_fall_tilt_change", "motion_evidence_hold_seconds",
                     "motion_fall_hold_seconds", "min_fall_motion_strength"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.post_fall_posture_threshold <= 1:
            raise ValueError("post_fall_posture_threshold must be in (0, 1]")
        for name in ("duration", "max_frames", "max_reconnects", "max_inference_retries"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("max_frames", "max_reconnects", "max_inference_retries"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        for name in ("show", "save_video"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.min_observations > 4096:
            raise ValueError("min_observations cannot exceed the bounded window capacity (4096)")
        if self.transition_seconds <= self.fall_hold_seconds:
            raise ValueError("transition_seconds must exceed fall_hold_seconds")
        if self.backend == "cpp" and not self.library:
            raise ValueError("cpp backend requires library")
        if not self.video_id:
            raise ValueError("video_id cannot be empty")
        if self.source_roi is not None:
            if (not isinstance(self.source_roi, list) or len(self.source_roi) != 4 or
                    any(not isinstance(v,int) or isinstance(v,bool) or v < 0 for v in self.source_roi) or
                    self.source_roi[2] <= self.source_roi[0] or self.source_roi[3] <= self.source_roi[1]):
                raise ValueError("source_roi must be nonnegative integer xyxy with positive dimensions")
        return self

    @classmethod
    def load(cls, path: str | None, overrides: dict | None = None) -> "Config":
        values = {}
        if path:
            values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            if not isinstance(values, dict):
                raise ValueError("configuration must be a mapping")
        values.update({k: v for k, v in (overrides or {}).items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values).validate()

    def save(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")
