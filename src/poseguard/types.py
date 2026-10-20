from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Letterbox:
    scale: float
    left: int
    top: int
    width: int
    height: int
    size: int


@dataclass
class Detection:
    bbox: np.ndarray
    confidence: float
    keypoints: np.ndarray


@dataclass
class Frame:
    image: np.ndarray
    frame_id: int
    timestamp: float
    captured_ns: int
    generation: int = 0
    capture_ms: float = 0.0


@dataclass
class Inference:
    output: np.ndarray
    timings: dict[str, float] = field(default_factory=dict)
