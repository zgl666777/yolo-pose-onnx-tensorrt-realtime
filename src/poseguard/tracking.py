from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import box_iou
from .types import Detection


@dataclass
class Track:
    id: int
    detection: Detection
    last_seen: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    observations: deque = field(default_factory=lambda: deque(maxlen=4096))
    last_upright: float | None = None
    candidate_since: float | None = None
    recovery_since: float | None = None
    alert_active: bool = False
    label: str = "unknown"
    confidence: float = 0.0
    motion: deque = field(default_factory=lambda: deque(maxlen=256))
    motion_evidence_at: float | None = None
    motion_evidence_valid_until: float | None = None
    motion_strength: float = 0.0
    motion_features: dict = field(default_factory=dict)
    landing_evidence_at: float | None = None


class Tracker:
    def __init__(self, ttl: float = 1.0):
        self.ttl = ttl
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.last_timestamp = -float("inf")

    def reset(self):
        self.tracks.clear()
        self.last_timestamp = -float("inf")

    def update(self, detections: list[Detection], timestamp: float) -> list[Track]:
        if timestamp < self.last_timestamp:
            raise ValueError("Tracking timestamps must be monotonic")
        self.last_timestamp = timestamp
        self.tracks = {k: t for k, t in self.tracks.items() if timestamp - t.last_seen <= self.ttl}
        existing = list(self.tracks.values())
        matched = set()
        visible = []
        if existing and detections:
            predicted = []
            for track in existing:
                dt = timestamp - track.last_seen
                shift = np.tile(track.velocity * dt, 2)
                predicted.append(track.detection.bbox + shift)
            boxes = np.stack([d.bbox for d in detections])
            predicted = np.stack(predicted)
            overlaps = box_iou(predicted, boxes)
            cost = 1 - overlaps
            for i, track in enumerate(existing):
                for j, det in enumerate(detections):
                    a, b = track.detection, det
                    scale = max(np.linalg.norm(predicted[i, 2:] - predicted[i, :2]),
                                np.linalg.norm(b.bbox[2:] - b.bbox[:2]), 1)
                    center_dist = np.linalg.norm((predicted[i, :2] + predicted[i, 2:] -
                                                  b.bbox[:2] - b.bbox[2:]) / 2) / scale
                    valid = (a.keypoints[:, 2] >= 0.3) & (b.keypoints[:, 2] >= 0.3)
                    if valid.sum() >= 4:
                        shift = track.velocity * (timestamp - track.last_seen)
                        pose_dist = np.linalg.norm(a.keypoints[valid, :2] + shift -
                                                   b.keypoints[valid, :2], axis=1).mean() / scale
                    else:
                        pose_dist = center_dist
                    cost[i, j] += 0.6 * center_dist + 0.4 * pose_dist
                    if center_dist > 0.8 or (overlaps[i, j] < 0.02 and center_dist > 0.5):
                        cost[i, j] = 1e6
            row_ids, col_ids = linear_sum_assignment(cost)
            for i, j in zip(row_ids, col_ids):
                if cost[i, j] >= 1.6:
                    continue
                track, det = existing[i], detections[j]
                dt = timestamp - track.last_seen
                if dt > 1e-6:
                    old_center = (track.detection.bbox[:2] + track.detection.bbox[2:]) / 2
                    new_center = (det.bbox[:2] + det.bbox[2:]) / 2
                    track.velocity = 0.5 * track.velocity + 0.5 * (new_center - old_center) / dt
                track.detection, track.last_seen = det, timestamp
                matched.add(int(j))
                visible.append(track)
        for j, det in enumerate(detections):
            if j not in matched:
                track = Track(self.next_id, det, timestamp)
                self.tracks[track.id] = track
                self.next_id += 1
                visible.append(track)
        return sorted(visible, key=lambda t: t.id)


class TemporalDetector:
    def __init__(self, config):
        self.config = config

    def update_motion(self, track, probs, timestamp):
        cfg, det = self.config, track.detection
        if (track.motion_evidence_valid_until is None or
                timestamp > track.motion_evidence_valid_until):
            track.motion_evidence_at = None
            track.motion_evidence_valid_until = None
            track.motion_strength = 0.0
        kps = det.keypoints
        visible = kps[:, 2] >= cfg.keypoint_confidence
        if probs.get("unknown", 0) >= .5 or not visible[[5, 6, 11, 12]].all():
            track.motion.clear()
            track.motion_features = dict(reliable=False)
            return
        shoulder, hip = kps[[5, 6], :2].mean(axis=0), kps[[11, 12], :2].mean(axis=0)
        delta = hip - shoulder
        tilt = float(np.arctan2(abs(delta[0]), abs(delta[1]) + 1e-6) / (np.pi / 2))
        height = float(det.bbox[3] - det.bbox[1])
        track.motion.append((timestamp, float(hip[1]), height, tilt))
        while track.motion and timestamp - track.motion[0][0] > cfg.motion_window_seconds:
            track.motion.popleft()
        vertical_speed = tilt_rate = tilt_change = 0.0
        for old_time, old_y, old_height, old_tilt in track.motion:
            dt = timestamp - old_time
            if dt < .12:
                continue
            scale = max(height, old_height, 1)
            speed = (float(hip[1]) - old_y) / scale / dt
            change = tilt - old_tilt
            rate = change / dt
            vertical_speed = max(vertical_speed, speed)
            tilt_rate = max(tilt_rate, rate)
            tilt_change = max(tilt_change, change)
            supported = track.last_upright is not None and timestamp - track.last_upright <= cfg.transition_seconds
            if supported and speed >= cfg.min_fall_vertical_speed and rate >= cfg.min_fall_tilt_rate and change >= cfg.min_fall_tilt_change:
                track.motion_evidence_at = timestamp
                track.motion_evidence_valid_until = timestamp + cfg.motion_evidence_hold_seconds
                track.motion_strength = max(track.motion_strength, speed)
        track.motion_features = dict(reliable=True, hip_down_body_heights_per_second=vertical_speed,
                                     torso_tilt_rate=tilt_rate, torso_tilt_change=tilt_change,
                                     torso_tilt=tilt, evidence_timestamp=track.motion_evidence_at)

    def update(self, track: Track, probs: dict[str, float], timestamp: float) -> str | None:
        cfg = self.config
        if track.observations and timestamp - track.observations[-1][0] > min(cfg.track_ttl, cfg.window_seconds):
            track.observations.clear()
            track.candidate_since = track.recovery_since = None
            track.motion.clear()
            track.motion_evidence_at = track.last_upright = None
            track.motion_evidence_valid_until = None
            track.motion_strength = 0.0
            track.landing_evidence_at = None
        self.update_motion(track, probs, timestamp)
        track.observations.append((timestamp, probs))
        while track.observations and timestamp - track.observations[0][0] > cfg.window_seconds:
            track.observations.popleft()
        labels = {label for _, p in track.observations for label in p}
        smooth = {label: sum(p.get(label, 0) for _, p in track.observations) / len(track.observations)
                  for label in labels}
        track.label = max(smooth, key=smooth.get)
        track.confidence = smooth[track.label]
        # Both standing and supported sitting can precede a fall to the floor.
        current_reliable = probs.get("unknown", 0) < 0.5
        if (current_reliable and track.label in {"standing", "sitting"} and
                track.confidence >= cfg.fall_threshold):
            track.last_upright = timestamp
        transitioned = track.last_upright is not None and timestamp - track.last_upright <= cfg.transition_seconds
        motion_supported = (track.motion_evidence_at is not None and
                            track.motion_evidence_valid_until is not None and
                            timestamp <= track.motion_evidence_valid_until)
        strong_motion = motion_supported and track.motion_strength >= cfg.min_fall_motion_strength
        width, height = track.detection.bbox[2:] - track.detection.bbox[:2]
        horizontal_box = width >= max(height, 1)
        reliable_torso = track.motion_features.get("reliable", False)
        current_landing = current_reliable and reliable_torso and (
            probs.get("lying", 0) >= cfg.post_fall_posture_threshold or
            (horizontal_box and track.motion_features.get("torso_tilt", 0) >= .7))
        if current_landing:
            track.landing_evidence_at = timestamp
        # Impact occlusion may continue an existing candidate only when the
        # current box is horizontal and a recent visible landing established it.
        occluded_landing = (track.candidate_since is not None and not reliable_torso and horizontal_box and
                            track.landing_evidence_at is not None and
                            timestamp - track.landing_evidence_at <= min(cfg.window_seconds,
                                                                         cfg.motion_evidence_hold_seconds))
        landed = current_landing or occluded_landing
        fall_score = smooth.get("falling", 0) + (smooth.get("lying", 0)
                                               if transitioned and motion_supported and landed else 0)
        if strong_motion and landed:
            fall_score = max(fall_score, cfg.fall_threshold)
        enough_observations = len(track.observations) >= cfg.min_observations
        reliable = current_reliable and enough_observations
        candidate = enough_observations and (reliable or (strong_motion and landed)) and fall_score >= cfg.fall_threshold
        if candidate:
            if track.candidate_since is None:
                track.candidate_since = timestamp
            track.recovery_since = None
            hold = cfg.motion_fall_hold_seconds if strong_motion and landed else cfg.fall_hold_seconds
            if not track.alert_active and timestamp - track.candidate_since >= hold:
                track.alert_active = True
                return "fall"
        else:
            track.candidate_since = None
            recovered = reliable and track.label in {"standing", "sitting"} and track.confidence >= cfg.fall_threshold
            if track.alert_active and recovered:
                if track.recovery_since is None:
                    track.recovery_since = timestamp
                if timestamp - track.recovery_since >= cfg.recovery_seconds:
                    track.alert_active = False
                    track.recovery_since = None
                    return "recovered"
            else:
                track.recovery_since = None
        return None
