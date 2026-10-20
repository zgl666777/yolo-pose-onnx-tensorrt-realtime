from __future__ import annotations

import math
import cv2
import numpy as np
from .types import Detection

# COCO's 17-keypoint ordering, with the person upright in local coordinates.
TEMPLATE = np.array([
    [0, -1.0], [-0.08, -1.04], [0.08, -1.04], [-0.16, -1], [0.16, -1],
    [-0.26, -0.65], [0.26, -0.65], [-0.35, -0.15], [0.35, -0.15],
    [-0.42, 0.3], [0.42, 0.3], [-0.2, 0.1], [0.2, 0.1],
    [-0.22, 0.65], [0.22, 0.65], [-0.24, 1.2], [0.24, 1.2]
], dtype=np.float32)


def synthetic_person(center, angle=0.0, scale=120) -> Detection:
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    xy = TEMPLATE @ rotation.T * scale + np.asarray(center)
    keypoints = np.column_stack((xy, np.full(17, 0.98))).astype(np.float32)
    bbox = np.concatenate((xy.min(axis=0) - 14, xy.max(axis=0) + 14))
    return Detection(bbox.astype(np.float32), 0.98, keypoints)


def demo_frame(timestamp: float, width=960, height=540):
    image = np.full((height, width, 3), (30, 35, 34), np.uint8)
    cv2.rectangle(image, (0, 405), (width, height), (45, 51, 49), -1)
    cv2.line(image, (0, 405), (width, 405), (95, 105, 98), 2)
    cv2.putText(image, "POSEGUARD / SYNTHETIC DEMO", (28, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (222, 231, 225), 1, cv2.LINE_AA)
    phase = timestamp % 14
    angle = min(max((phase - 3.0) / 0.6, 0), 1) * math.pi / 2
    if phase > 9:
        angle = max(0, 1 - (phase - 9) / 0.8) * math.pi / 2
    center = (300 + 15 * math.sin(timestamp), 275 + 65 * math.sin(angle))
    return image, [synthetic_person(center, angle),
                   synthetic_person((720 + 18 * math.sin(timestamp * 0.6), 275), scale=108)]

