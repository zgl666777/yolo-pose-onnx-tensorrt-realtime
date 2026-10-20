from __future__ import annotations

import math
from pathlib import Path
import joblib
import numpy as np
from .types import Detection

LABELS = ("standing", "sitting", "lying", "falling")
FEATURE_VERSION = 1
MIN_VISIBLE_RATIO = 0.35
FEATURE_NAMES = tuple(f"kp{i}_{axis}" for i in range(17) for axis in ("x", "y", "c")) + (
    "torso_tilt", "body_tilt", "log_aspect", "visible_ratio", "torso_length")


def extract_features(det: Detection, threshold: float = 0.3) -> np.ndarray:
    box, kps = np.asarray(det.bbox), np.asarray(det.keypoints)
    if box.shape != (4,) or kps.shape != (17, 3) or not np.isfinite(kps).all():
        raise ValueError("Invalid bbox/keypoints")
    if not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("bbox must have positive finite dimensions")
    w, h = np.maximum(box[2:] - box[:2], 1)
    normalized = kps.copy().astype(np.float32)
    visible = kps[:, 2] >= threshold
    normalized[:, :2] = (kps[:, :2] - box[:2]) / np.array([w, h])
    normalized[~visible, :2] = 0
    normalized[:, 2] = np.clip(normalized[:, 2], 0, 1)

    def center(indices):
        valid = [i for i in indices if visible[i]]
        return kps[valid, :2].mean(axis=0) if valid else None

    shoulder, hip = center((5, 6)), center((11, 12))
    head, ankle = center((0, 1, 2)), center((15, 16))

    def tilt(a, b):
        if a is None or b is None:
            return 0.0
        delta = b - a
        return math.atan2(abs(float(delta[0])), abs(float(delta[1])) + 1e-6) / (math.pi / 2)

    length = 0.0 if shoulder is None or hip is None else np.linalg.norm(hip - shoulder) / h
    extra = [tilt(shoulder, hip), tilt(head, ankle), np.log(w / h), visible.mean(), length]
    return np.concatenate((normalized.ravel(), extra)).astype(np.float32)


def predict_feature_probabilities(model, features):
    """Apply the runtime quality gate to every batch, including offline scoring."""
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Invalid classifier feature batch")
    accepted = x[:, -2] >= MIN_VISIBLE_RATIO
    results = [{"unknown": 1.0} for _ in x]
    if accepted.any():
        values = model.predict_proba(x[accepted])
        for index, probabilities in zip(np.flatnonzero(accepted), values):
            results[index] = {str(k): float(v) for k, v in zip(model.classes_, probabilities)}
    return results


def predict_feature_labels(model, features):
    return np.asarray([max(p, key=p.get) for p in predict_feature_probabilities(model, features)])


class RuleClassifier:
    """An untrained baseline. Temporal transitions distinguish lying from falls."""

    def predict(self, det: Detection, threshold: float = 0.3) -> dict[str, float]:
        x = extract_features(det, threshold)
        probs = dict.fromkeys(LABELS, 0.0)
        if x[-2] < MIN_VISIBLE_RATIO:
            return {"unknown": 1.0}
        horizontal = x[-5] > 0.66 or x[-4] > 0.72
        if horizontal and x[-3] > np.log(0.9):
            probs["lying"] = 0.9
            probs["sitting"] = 0.1
        elif x[-3] > np.log(0.6):
            probs["sitting"] = 0.8
            probs["standing"] = 0.2
        else:
            probs["standing"] = 0.9
            probs["sitting"] = 0.1
        return probs


class LearnedClassifier:
    def __init__(self, path: str):
        bundle = joblib.load(Path(path))
        if bundle.get("feature_version") != FEATURE_VERSION or tuple(bundle.get("features", ())) != FEATURE_NAMES:
            raise ValueError("Classifier feature schema does not match this version")
        self.model = bundle["model"]
        self.threshold = bundle["keypoint_threshold"]

    def predict(self, det: Detection, threshold: float = 0.3) -> dict[str, float]:
        if abs(threshold - self.threshold) > 1e-6:
            raise ValueError("Runtime keypoint threshold differs from classifier training")
        features = extract_features(det, threshold)[None]
        return predict_feature_probabilities(self.model, features)[0]
