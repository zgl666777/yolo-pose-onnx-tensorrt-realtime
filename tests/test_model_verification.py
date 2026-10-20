import json

import numpy as np
import pytest

from poseguard.image_io import write_image
from poseguard.model_tools import compare_outputs, verify_model, verification_evidence, explicit_fp16_score_graph
from poseguard.geometry import preprocess
from poseguard.types import Inference


def test_empty_scene_raw_comparison_does_not_supply_pose_evidence():
    _, meta = preprocess(np.zeros((640, 640, 3), np.uint8))
    output = np.zeros((1, 56, 1), np.float32)
    result = compare_outputs(output, output, meta, .1, .005, .001)
    assert result["passed"] is True
    assert result["keypoint_evidence"] is False
    gate = verification_evidence([dict(result, candidate="engine")], ["engine"])
    assert gate["passed"] is False
    assert gate["candidates"]["engine"]["status"] == "insufficient"


@pytest.mark.parametrize("candidate_has_person", [False, True])
def test_verify_report_requires_visible_pose_evidence_per_candidate(tmp_path, monkeypatch, candidate_has_person):
    import poseguard.model_tools as model_tools

    image = tmp_path / "image.png"
    write_image(image, np.zeros((640, 640, 3), np.uint8))
    raw = np.zeros((1, 56, 1), np.float32)
    if candidate_has_person:
        raw[0, :5, 0] = [320, 320, 100, 200, .9]
        raw[0, 5:, 0] = np.tile([320, 320, .9], 17)
    model = tmp_path / "model.onnx"
    engine = tmp_path / "candidate.engine"
    model.write_bytes(b"contract-only model")
    engine.write_bytes(b"contract-only engine")

    class Backend:
        def __init__(self, *args):
            pass

        def infer(self, tensor):
            return Inference(raw.copy())

        def close(self):
            pass

    monkeypatch.setattr(model_tools, "ONNXBackend", Backend)
    monkeypatch.setattr(model_tools, "TensorRTBackend", Backend)
    report = verify_model(str(model), str(image), str(tmp_path / "report.json"), engines=[str(engine)])
    assert report["numerics_passed"] is True
    assert report["passed"] is candidate_has_person
    assert report["status"] == ("passed" if candidate_has_person else "insufficient_pose_evidence")
    assert json.loads((tmp_path / "report.json").read_text())["passed"] is candidate_has_person


def test_pose_evidence_cannot_be_borrowed_from_another_candidate():
    rows = [dict(candidate="fp32", passed=True, keypoint_evidence=True),
            dict(candidate="fp16", passed=True, keypoint_evidence=False)]
    result = verification_evidence(rows, ["fp32", "fp16"])
    assert result["passed"] is False
    assert result["candidates"]["fp32"]["status"] == "sufficient"
    assert result["candidates"]["fp16"]["status"] == "insufficient"


def test_explicit_mixed_graph_changes_only_supported_score_convolution(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper, TensorProto

    source = tmp_path / "source.onnx"
    output = tmp_path / "explicit.onnx"
    weights = numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "weight")
    bias = numpy_helper.from_array(np.zeros(1, np.float32), "bias")
    backbone = helper.make_node("Relu", ["image"], ["features"], name="backbone/relu")
    score = helper.make_node("Conv", ["features", "weight", "bias"], ["score"], name="/model.23/cv3.0/cv3.0.2/Conv")
    graph = helper.make_graph([backbone, score], "test", [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 2, 2])],
                              [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1, 2, 2])], [weights, bias])
    original = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(original, source)
    report = explicit_fp16_score_graph(source, output)
    transformed = onnx.load(output)
    onnx.checker.check_model(transformed)
    assert onnx.load(source).SerializeToString() == original.SerializeToString()
    assert transformed.graph.node[0].SerializeToString() == backbone.SerializeToString()
    assert [node.op_type for node in transformed.graph.node] == ["Relu", "Cast", "Conv", "Cast"]
    assert report["selected_nodes"] == [score.name]
    assert transformed.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    floats = {tensor.name: tensor.data_type for tensor in transformed.graph.initializer}
    assert floats["weight"] == TensorProto.FLOAT
    assert floats["weight__poseguard_score_fp16"] == TensorProto.FLOAT16

    source_model = onnx.load(source)
    source_model.graph.node[1].name = "/model.23/cv3.2/cv3.2.2/Conv"
    onnx.save(source_model, source)
    lowres = explicit_fp16_score_graph(source, tmp_path / "lowres.onnx", scales=(2,))
    assert lowres["selected_scales"] == [2]
    assert lowres["selected_nodes"] == ["/model.23/cv3.2/cv3.2.2/Conv"]


def test_explicit_mixed_graph_refuses_an_unsupported_head(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto

    source = tmp_path / "unsupported.onnx"
    graph = helper.make_graph([helper.make_node("Identity", ["image"], ["output"])], "unsupported",
                              [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])])
    onnx.save(helper.make_model(graph), source)
    with pytest.raises(ValueError, match="supported YOLO"):
        explicit_fp16_score_graph(source, tmp_path / "refused.onnx")
