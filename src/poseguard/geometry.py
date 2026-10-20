from __future__ import annotations

import cv2
import numpy as np
from .types import Detection, Letterbox


def preprocess(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, Letterbox]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 HWC BGR image")
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image")
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    left = round((size - new_w) / 2 - 0.1)
    top = round((size - new_h) / 2 - 0.1)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, size - new_h - top,
                               left, size - new_w - left, cv2.BORDER_CONSTANT,
                               value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None],
                                 dtype=np.float32) / 255.0
    return tensor, Letterbox(scale, left, top, w, h, size)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float,
        max_detections: int = 100) -> list[int]:
    order = np.argsort(-scores, kind="stable")[:3000]
    keep = []
    while order.size and len(keep) < max_detections:
        idx = int(order[0])
        keep.append(idx)
        rest = order[1:]
        order = rest[box_iou(boxes[idx:idx + 1], boxes[rest])[0] <= threshold]
    return keep


def raw_rows(output: np.ndarray) -> np.ndarray:
    raw = np.asarray(output)
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"Expected batch-one raw YOLO Pose output; got {raw.shape}")
    if raw.shape[1] == 56:
        return raw[0].T.astype(np.float32)
    if raw.shape[2] == 56:
        return raw[0].astype(np.float32)
    raise ValueError(f"Expected 56 channels (4 box + 1 score + 17x3); got {raw.shape}. "
                     "Export a one-class COCO pose model with nms=False.")


def decode(output: np.ndarray, meta: Letterbox, confidence: float = 0.35,
           iou: float = 0.45) -> list[Detection]:
    rows = raw_rows(output)
    rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= confidence) &
                (rows[:, 2] > 0) & (rows[:, 3] > 0)]
    if not len(rows):
        return []
    boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2,
                            rows[:, :2] + rows[:, 2:4] / 2), axis=1)
    selected = nms(boxes, rows[:, 4], iou)
    detections = []
    for idx in selected:
        box = boxes[idx].copy()
        kps = rows[idx, 5:].reshape(17, 3).copy()
        box[[0, 2]] = (box[[0, 2]] - meta.left) / meta.scale
        box[[1, 3]] = (box[[1, 3]] - meta.top) / meta.scale
        kps[:, 0] = (kps[:, 0] - meta.left) / meta.scale
        kps[:, 1] = (kps[:, 1] - meta.top) / meta.scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, meta.width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, meta.height)
        kps[:, 0] = np.clip(kps[:, 0], 0, meta.width)
        kps[:, 1] = np.clip(kps[:, 1], 0, meta.height)
        kps[:, 2] = np.clip(kps[:, 2], 0, 1)
        if box[2] > box[0] and box[3] > box[1]:
            detections.append(Detection(box, float(rows[idx, 4]), kps))
    return detections

