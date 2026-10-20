from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .backends import ONNXBackend, TensorRTBackend, CppBackend
from .geometry import preprocess, decode, box_iou, raw_rows
from .metrics import Metrics
from .provenance import environment, artifact
from .image_io import read_image


def export_model(weights: str, output: str, size: int = 640, opset: int = 17) -> dict:
    from ultralytics import YOLO
    import onnx
    import shutil
    model = YOLO(weights, task="pose")
    if model.task != "pose" or list(model.model.model[-1].kpt_shape) != [17, 3] or model.model.model[-1].nc != 1:
        raise ValueError("Use a one-class COCO 17-keypoint YOLO pose model")
    exported = Path(model.export(format="onnx", imgsz=size, batch=1, dynamic=False,
                                 simplify=False, opset=opset, half=False, nms=False, device="cpu"))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)
    onnx.checker.check_model(onnx.load(destination))
    session = ONNXBackend(str(destination), size)
    sample = session.infer(np.zeros((1, 3, size, size), np.float32)).output
    session.close()
    report = dict(weights=weights, output=str(destination), input_shape=[1, 3, size, size],
                  output_shape=list(sample.shape), opset=opset,
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    destination.with_suffix(".onnx.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def explicit_fp16_score_graph(onnx_path, output, *, scales=(0, 1, 2)):
    """Keep graph arithmetic FP32 except explicitly cast final person-logit convolutions."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    model = onnx.load(onnx_path)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    rebuilt, selected = [], []
    for node in model.graph.node:
        scale_match = re.search(r"/cv3\.(\d+)/cv3\.\d+\.2/Conv$", node.name)
        lower = node.op_type == "Conv" and scale_match is not None and int(scale_match.group(1)) in scales
        if not lower:
            rebuilt.append(node)
            continue
        if len(node.input) < 2 or len(node.output) != 1:
            raise ValueError("Explicit FP16 score policy requires a single-output Conv with constant weights")
        selected.append(node.name)
        activation, original_output = node.input[0], node.output[0]
        cast_activation = activation + "__poseguard_score_fp16"
        rebuilt.append(helper.make_node("Cast", [activation], [cast_activation],
                                        name=node.name + "/CastInputFP16", to=TensorProto.FLOAT16))
        node.input[0] = cast_activation
        for index in range(1, len(node.input)):
            constant = initializers.get(node.input[index])
            if constant is None or constant.data_type != TensorProto.FLOAT:
                raise ValueError("Explicit FP16 score policy requires FP32 constant weights and bias")
            name = constant.name + "__poseguard_score_fp16"
            model.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(constant).astype(np.float16), name))
            node.input[index] = name
        node.output[0] = original_output + "__poseguard_score_fp16"
        rebuilt.append(node)
        rebuilt.append(helper.make_node("Cast", [node.output[0]], [original_output],
                                        name=node.name + "/CastOutputFP32", to=TensorProto.FLOAT))
    if not selected:
        raise ValueError("Explicit FP16 score policy requires the supported YOLO pose person-logit head")
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    onnx.checker.check_model(model)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return {"selected_nodes": selected, "graph": artifact(destination),
            "selected_scales": list(scales),
            "policy": "Explicit FP16 score Conv input/weights/output, cast back to FP32; other arithmetic unchanged"}


def build_engine(onnx_path: str, output: str, precision: str, workspace_gb: float = 2) -> dict:
    from .runtime import configure_gpu_runtime
    configure_gpu_runtime()
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_graph = None
    parsed_path = Path(onnx_path)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if precision in {"fp16-explicit", "fp16-explicit-lowres"}:
        parsed_path = Path(output).with_suffix(".explicit-fp16.onnx")
        explicit_graph = explicit_fp16_score_graph(onnx_path, parsed_path,
                                                   scales=(2,) if precision == "fp16-explicit-lowres" else (0, 1, 2))
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(parsed_path.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError("Expected a single input and raw output")
    inp, out = network.get_input(0), network.get_output(0)
    if len(inp.shape) != 4 or tuple(inp.shape[:2]) != (1, 3) or any(x <= 0 for x in inp.shape):
        raise ValueError("Export a static batch-one NCHW ONNX model first")
    if inp.shape[2] != inp.shape[3] or inp.dtype != trt.float32:
        raise ValueError("Expected square float32 input")
    if workspace_gb <= 0:
        raise ValueError("workspace-gb must be positive")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024 ** 3))
    cfg.clear_flag(trt.BuilderFlag.TF32)
    fp16_layers, fp32_layers = [], []
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if precision in {"fp16-explicit", "fp16-explicit-lowres"}:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU does not report fast FP16 support")
        fp16_layers = explicit_graph["selected_nodes"]
    elif precision in {"fp16", "fp16-strict"}:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU does not report fast FP16 support")
        cfg.set_flag(trt.BuilderFlag.FP16)
        if precision == "fp16-strict":
            # Raw coordinates are sensitive to backbone rounding. The strict
            # mixed policy only lowers final person-logit convolutions; pose
            # and backbone computations remain FP32. Verification is still
            # required on the application's images after building the engine.
            cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            for index in range(network.num_layers):
                layer = network.get_layer(index)
                if any(layer.get_output(j).dtype == trt.float32 for j in range(layer.num_outputs)):
                    lower = (layer.type == trt.LayerType.CONVOLUTION and
                             re.search(r"/cv3\.\d+/cv3\.\d+\.2/Conv$", layer.name) is not None)
                    dtype = trt.float16 if lower else trt.float32
                    (fp16_layers if lower else fp32_layers).append(layer.name)
                    layer.precision = dtype
                    for output_index in range(layer.num_outputs):
                        if layer.get_output(output_index).dtype == trt.float32:
                            layer.set_output_type(output_index, dtype)
            if not fp16_layers:
                raise ValueError("fp16-strict requires the supported YOLO pose person-logit head")
    elif precision != "fp32":
        raise ValueError("precision must be fp32, fp16, fp16-strict, fp16-explicit or fp16-explicit-lowres")
    if precision not in {"fp16-explicit", "fp16-explicit-lowres"}:
        out.dtype = trt.float32
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Cannot deserialize newly built engine for precision inspection")
    inspector = engine.create_engine_inspector()
    inspection = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    inspection_path = destination.with_suffix(".engine.layers.json")
    inspection_path.write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    from .backends import Cuda
    cuda = Cuda()
    props = cuda.call("cudaGetDeviceProperties", 0)
    name = props.name.decode() if isinstance(props.name, bytes) else str(props.name)
    report = dict(onnx=str(onnx_path), onnx_sha256=hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest(),
                  engine_sha256=artifact(destination)["sha256"], environment=environment(),
                  precision=precision, tf32=False, input_shape=list(inp.shape),
                  precision_policy=("Strongly typed explicit FP16 lowest-resolution person-logit Conv only; other arithmetic FP32"
                                    if precision == "fp16-explicit-lowres" else "Strongly typed explicit FP16 person-logit Conv; FP32 backbone and pose head"
                                    if precision == "fp16-explicit" else "FP16 person-logit final convolutions; FP32 backbone and pose head"
                                    if precision == "fp16-strict" else "TensorRT tactic selection"),
                  explicit_precision_graph=explicit_graph,
                  constrained_fp16_layers=fp16_layers, constrained_fp32_layers=fp32_layers,
                  engine_inspection=artifact(inspection_path),
                  tensorrt=trt.__version__, gpu=name, compute_capability=[props.major, props.minor],
                  workspace_gb=workspace_gb, bytes=destination.stat().st_size,
                  portability="Rebuild on deployment GPU with matching TensorRT/CUDA")
    destination.with_suffix(".engine.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def image_paths(path: str) -> list[Path]:
    root = Path(path)
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in
                                                {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise ValueError(f"No images found: {path}")
    return paths


def compare_outputs(reference, candidate, meta, coordinate_atol, score_atol, rtol,
                    confidence=0.35, iou=0.45) -> dict:
    a, b = raw_rows(reference), raw_rows(candidate)
    if a.shape != b.shape:
        raise ValueError(f"Output shape mismatch: {a.shape} vs {b.shape}")
    coords = [0, 1, 2, 3] + [i for k in range(17) for i in (5 + 3 * k, 6 + 3 * k)]
    scores = [4] + [7 + 3 * k for k in range(17)]
    coord_ok = np.allclose(a[:, coords], b[:, coords], atol=coordinate_atol, rtol=rtol)
    score_ok = np.allclose(a[:, scores], b[:, scores], atol=score_atol, rtol=rtol)
    ref, cand = decode(reference, meta, confidence, iou), decode(candidate, meta, confidence, iou)
    matches = []
    if ref and cand:
        overlap = box_iou(np.stack([d.bbox for d in ref]), np.stack([d.bbox for d in cand]))
        ri, ci = linear_sum_assignment(1 - overlap)
        for i, j in zip(ri, ci):
            visible = ref[i].keypoints[:, 2] >= 0.3
            candidate_visible = cand[j].keypoints[:, 2] >= 0.3
            diff = np.linalg.norm(ref[i].keypoints[visible, :2] - cand[j].keypoints[visible, :2], axis=1)
            matches.append(dict(iou=float(overlap[i, j]), reference_visible_keypoints=int(visible.sum()),
                                lost_visible_keypoints=int((visible & ~candidate_visible).sum()),
                                gained_visible_keypoints=int((~visible & candidate_visible).sum()),
                                visibility_passed=bool(np.array_equal(visible, candidate_visible)),
                                keypoint_check_sufficient=bool(visible.sum() >= 4),
                                visible_keypoint_max_px=float(diff.max()) if diff.size else None))
    post_ok = len(ref) == len(cand) and all(m["iou"] >= 0.9 and
                                           m["visibility_passed"] and m["keypoint_check_sufficient"] and
                                           m["visible_keypoint_max_px"] <= 3
                                           for m in matches)
    return dict(passed=bool(coord_ok and score_ok and post_ok), raw_coordinates_passed=bool(coord_ok),
                raw_scores_passed=bool(score_ok), postprocess_passed=bool(post_ok),
                coordinate_abs_max=float(np.abs(a[:, coords] - b[:, coords]).max()),
                score_abs_max=float(np.abs(a[:, scores] - b[:, scores]).max()),
                reference_detections=len(ref), candidate_detections=len(cand), matches=matches,
                keypoint_evidence=bool(matches) and all(m["keypoint_check_sufficient"] for m in matches),
                visibility_threshold=0.3, keypoint_scope="Reference-visible points; visibility changes fail; >=4 points per detection required")


def verification_evidence(rows, candidate_names):
    """Require actual visible people per candidate before claiming pose accuracy."""
    evidence = {}
    for name in candidate_names:
        candidate_rows = [row for row in rows if row["candidate"] == name]
        visible_rows = [row for row in candidate_rows if row.get("keypoint_evidence") is True]
        evidence[name] = {"status": "sufficient" if visible_rows else "insufficient",
                          "images_compared": len(candidate_rows), "images_with_visible_pose_evidence": len(visible_rows),
                          "reason": ("At least one image has matched people with four or more reference-visible keypoints"
                                     if visible_rows else "No matched visible people; empty scenes verify raw output only")}
    return {"passed": bool(evidence) and all(row["status"] == "sufficient" for row in evidence.values()),
            "candidates": evidence, "scope": "Visible-pose evidence gate; does not establish representative dataset coverage"}


def verify_model(onnx_path, images, output, size=640, weights=None, engines=None,
                 coordinate_atol=0.1, score_atol=0.005, rtol=0.001) -> dict:
    reference = ONNXBackend(onnx_path, size)
    candidates = []
    try:
        if weights:
            import torch
            from ultralytics import YOLO
            torch_model = YOLO(weights, task="pose").model.cpu().float().eval()
            def torch_infer(tensor):
                with torch.inference_mode():
                    result = torch_model(torch.from_numpy(tensor))
                return (result[0] if isinstance(result, tuple) else result).detach().cpu().numpy()
            candidates.append(("pytorch", torch_infer))
        for path in engines or []:
            backend = TensorRTBackend(path, size)
            candidates.append((path, backend))
        if not candidates:
            raise ValueError("Specify --weights and/or --engine to compare against ONNX Runtime")
        rows = []
        for path in image_paths(images):
            image = read_image(path)
            if image is None:
                raise ValueError(f"Unreadable image: {path}")
            tensor, meta = preprocess(image, size)
            baseline = reference.infer(tensor).output
            for name, backend in candidates:
                candidate = backend(tensor) if callable(backend) else backend.infer(tensor).output
                rows.append(dict(image=str(path), candidate=name,
                                 **compare_outputs(baseline, candidate, meta, coordinate_atol, score_atol, rtol)))
        evidence = verification_evidence(rows, [name for name, _ in candidates])
        numerics_passed = bool(rows) and all(row["passed"] for row in rows)
        report = dict(passed=numerics_passed and evidence["passed"],
                      numerics_passed=numerics_passed, pose_evidence=evidence,
                      status=("passed" if numerics_passed and evidence["passed"] else
                              "insufficient_pose_evidence" if not evidence["passed"] else "failed"),
                      environment=environment(), models=[artifact(p) for p in [onnx_path, weights, *(engines or [])] if p],
                      images=[artifact(p) for p in image_paths(images)],
                      tolerances=dict(coordinate_atol=coordinate_atol, score_atol=score_atol, rtol=rtol),
                      reference="ONNX Runtime CPU / shared preprocessing", results=rows)
        dest = Path(output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        reference.close()
        for _, backend in candidates:
            if not callable(backend):
                backend.close()


def benchmark(model, backend_name, output, size=640, image=None, warmup=20, iterations=200,
              provider="CPUExecutionProvider", library=None):
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations positive")
    backend = (ONNXBackend(model, size, provider) if backend_name == "onnx" else
               TensorRTBackend(model, size) if backend_name == "trt" else CppBackend(model, size, library))
    try:
        if image:
            source = read_image(image)
            if source is None:
                raise ValueError(f"Unreadable image: {image}")
        else:
            source = np.zeros((720, 1280, 3), np.uint8)
        tensor, _ = preprocess(source, size)
        for _ in range(warmup):
            backend.infer(tensor)
        stats = Metrics()
        start = time.perf_counter()
        for _ in range(iterations):
            tick = time.perf_counter()
            result = backend.infer(tensor)
            stats.add(**result.timings, call_ms=(time.perf_counter() - tick) * 1000)
        elapsed = time.perf_counter() - start
        report = dict(model=model, backend=backend_name, warmup=warmup, iterations=iterations,
                      environment=environment(), model_artifact=artifact(model), library_artifact=artifact(library),
                      fps=iterations / elapsed, timings=stats.report(),
                      scope="batch-one inference call including transfers; excludes capture/pre/postprocessing")
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        backend.close()
