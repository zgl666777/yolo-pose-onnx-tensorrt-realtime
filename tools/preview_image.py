"""Run the actual ONNX model on one image and save an annotated preview."""
import argparse
from pathlib import Path
import cv2
from poseguard.backends import ONNXBackend
from poseguard.config import Config
from poseguard.features import RuleClassifier
from poseguard.geometry import preprocess, decode
from poseguard.pipeline import draw_tracks
from poseguard.tracking import Tracker, TemporalDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/yolo11n-pose.onnx")
    parser.add_argument("--output", default="runs/onnx-preview.jpg")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        parser.error("unreadable image")
    cfg = Config(imgsz=args.imgsz)
    backend = ONNXBackend(args.model, args.imgsz)
    try:
        tensor, meta = preprocess(image, args.imgsz)
        result = backend.infer(tensor)
        detections = decode(result.output, meta)
        tracks = Tracker().update(detections, 0)
        classifier, temporal = RuleClassifier(), TemporalDetector(cfg)
        for track in tracks:
            temporal.update(track, classifier.predict(track.detection), 0)
        preview = draw_tracks(image, tracks, cfg, 1000 / max(result.timings["backend_ms"], 1e-9),
                              result.timings["backend_ms"], 0)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), preview):
            raise RuntimeError("Cannot write preview")
        print(f"people={len(tracks)} output={path} inference_ms={result.timings['backend_ms']:.3f}")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
