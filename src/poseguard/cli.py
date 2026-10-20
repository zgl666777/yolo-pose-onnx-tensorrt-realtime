from __future__ import annotations

import argparse
import json
import sys
from .config import Config


def parser():
    root = argparse.ArgumentParser(description="PoseGuard: YOLO Pose deployment and fall detection")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("run", "demo"):
        p = commands.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--backend", choices=["onnx", "trt", "cpp"])
        p.add_argument("--model")
        p.add_argument("--classifier")
        p.add_argument("--library")
        p.add_argument("--source")
        p.add_argument("--source-roi", type=int, nargs=4)
        p.add_argument("--video-id")
        p.add_argument("--imgsz", type=int)
        p.add_argument("--output", default=None)
        p.add_argument("--duration", type=float)
        p.add_argument("--max-frames", type=int)
        p.add_argument("--headless", dest="show", action="store_false", default=None)
        p.add_argument("--save-video", action="store_true", default=None)
        p.add_argument("--ort-provider")
    p = commands.add_parser("export")
    p.add_argument("--weights", default="yolo11n-pose.pt")
    p.add_argument("--output", default="models/yolo11n-pose.onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p = commands.add_parser("build-engine")
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--precision", choices=["fp32", "fp16", "fp16-strict", "fp16-explicit", "fp16-explicit-lowres"], default="fp16")
    p.add_argument("--workspace-gb", type=float, default=2)
    p = commands.add_parser("verify")
    p.add_argument("--onnx", required=True)
    p.add_argument("--weights")
    p.add_argument("--engine", action="append")
    p.add_argument("--images", required=True)
    p.add_argument("--output", default="runs/verification.json")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--coordinate-atol", type=float, default=0.1)
    p.add_argument("--score-atol", type=float, default=0.005)
    p.add_argument("--rtol", type=float, default=0.001)
    p = commands.add_parser("benchmark")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=["onnx", "trt", "cpp"], default="onnx")
    p.add_argument("--library")
    p.add_argument("--image")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--ort-provider", default="CPUExecutionProvider")
    p.add_argument("--output", default="runs/benchmark.json")
    p = commands.add_parser("train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="models/pose_classifier.joblib")
    p.add_argument("--report-dir", default="runs/training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keypoint-threshold", type=float, default=0.3)
    p.add_argument("--split-manifest")
    p = commands.add_parser("prepare-splits")
    p.add_argument("--annotations", required=True)
    p.add_argument("--frames")
    p.add_argument("--events")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("score-classifier")
    p.add_argument("--report-dir", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("extract")
    p.add_argument("--annotations", required=True)
    p.add_argument("--config", default="configs/camera.yaml")
    p.add_argument("--output", default="data/poses.jsonl")
    p.add_argument("--ground-truth-output")
    p = commands.add_parser("evaluate")
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--events")
    p.add_argument("--truth-events")
    p.add_argument("--frame-tolerance", type=float, default=0.05)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--max-alarm-delay", type=float, default=5)
    p.add_argument("--ground-truth-frames")
    p.add_argument("--prediction-frames")
    p.add_argument("--deadline-scope", choices=["media", "emitted"], default="media")
    p.add_argument("--event-delivery")
    p.add_argument("--output", default="runs/evaluation.json")
    p = commands.add_parser("doctor")
    p = commands.add_parser("review-audit", help="Audit manual review completion without promoting draft data")
    p.add_argument("--tasks", required=True)
    p.add_argument("--database", required=True)
    p.add_argument("--output")
    return root


def doctor():
    import importlib.util
    import platform
    modules = ["numpy", "cv2", "onnxruntime", "sklearn", "torch", "ultralytics", "tensorrt", "cuda"]
    result = dict(python=sys.version, platform=platform.platform(),
                  modules={m: importlib.util.find_spec(m) is not None for m in modules})
    if result["modules"]["onnxruntime"]:
        import onnxruntime as ort
        result["ort_providers"] = ort.get_available_providers()
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in {"run", "demo"}:
            from .pipeline import run_pipeline
            values = vars(args).copy()
            for key in ("command", "config"):
                values.pop(key)
            if args.command == "demo":
                values["duration"] = 14 if args.duration is None else args.duration
                values["output"] = args.output or "runs/demo"
            cfg = Config.load(args.config, values)
            result = run_pipeline(cfg, demo=args.command == "demo")
            return 1 if result["status"] == "failed" else 0
        if args.command == "doctor":
            result = doctor()
        elif args.command == "review-audit":
            from .review import ReviewStore
            from pathlib import Path
            database_exists = Path(args.database).is_file()
            store = ReviewStore(args.database if database_exists else ":memory:")
            try:
                result = store.audit(args.tasks)
            finally:
                store.connection.close()
            result["review_database_exists"] = database_exists
            if args.output:
                Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        elif args.command == "export":
            from .model_tools import export_model
            result = export_model(args.weights, args.output, args.imgsz, args.opset)
        elif args.command == "build-engine":
            from .model_tools import build_engine
            result = build_engine(args.onnx, args.output, args.precision, args.workspace_gb)
        elif args.command == "verify":
            from .model_tools import verify_model
            result = verify_model(args.onnx, args.images, args.output, args.imgsz, args.weights, args.engine,
                                  args.coordinate_atol, args.score_atol, args.rtol)
        elif args.command == "benchmark":
            from .model_tools import benchmark
            result = benchmark(args.model, args.backend, args.output, args.imgsz, args.image,
                               args.warmup, args.iterations, args.ort_provider, args.library)
        elif args.command == "train":
            from .dataset import train_classifier
            result = train_classifier(args.dataset, args.output, args.report_dir, args.seed,
                                      args.keypoint_threshold, args.split_manifest)
        elif args.command == "prepare-splits":
            from .dataset import prepare_splits
            result = prepare_splits(args.annotations, args.output, args.seed, args.events, args.frames)
        elif args.command == "score-classifier":
            from .dataset import score_classifier
            result = score_classifier(args.report_dir, args.output)
        elif args.command == "extract":
            from .dataset import extract_dataset
            result = extract_dataset(args.annotations, Config.load(args.config), args.output, args.ground_truth_output)
        elif args.command == "evaluate":
            from .evaluation import evaluate_predictions
            result = evaluate_predictions(args.ground_truth, args.predictions, args.output, args.events,
                                          args.truth_events, args.frame_tolerance, args.match_iou, args.max_alarm_delay,
                                          args.ground_truth_frames, args.prediction_frames, args.deadline_scope, args.event_delivery)
        if args.command == "review-audit" and args.output:
            print(json.dumps({k: result[k] for k in ("status", "ready_for_export", "tasks", "reviewed_tasks",
                                                      "remaining_tasks", "blockers")}, indent=2))
        else:
            print(json.dumps(result, indent=2))
        if args.command == "review-audit" and not result["ready_for_export"]:
            return 2
        return 1 if args.command == "verify" and not result["passed"] else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
