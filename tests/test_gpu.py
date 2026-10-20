"""Optional integration checks on a real CUDA GPU and compiled C++ SDK."""
import os
from pathlib import Path
import cv2
import numpy as np
import pytest
from poseguard.backends import TensorRTBackend, CppBackend
from poseguard.geometry import preprocess, decode

pytestmark = pytest.mark.skipif(os.environ.get("POSEGUARD_TEST_GPU") != "1",
                                reason="Set POSEGUARD_TEST_GPU=1 after building TensorRT engines and C++ library")


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_cpp_and_python_trt_use_identical_model_contract(precision):
    engine = f"models/pose-{precision}.engine"
    default_library = "build-gpu/cpp/Release/pose_trt.dll" if os.name == "nt" else "build/cpp/libpose_trt.so"
    library = os.environ.get("POSEGUARD_CPP_LIBRARY", default_library)
    assert Path(engine).exists() and Path(library).exists()
    image = cv2.imread("data/bus.jpg")
    assert image is not None
    tensor, meta = preprocess(image)
    python_backend, cpp_backend = TensorRTBackend(engine), CppBackend(engine, library=library)
    try:
        reference = python_backend.infer(tensor)
        candidate = cpp_backend.infer(tensor)
        np.testing.assert_allclose(candidate.output, reference.output, atol=0.001, rtol=0.00002)
        assert len(decode(candidate.output, meta)) >= 3
        for name in ("h2d_ms", "gpu_ms", "d2h_ms"):
            assert candidate.timings[name] >= 0 and reference.timings[name] >= 0
    finally:
        cpp_backend.close()
        python_backend.close()


def test_trt_buffers_survive_repeated_calls_without_output_aliasing():
    image = cv2.imread("data/bus.jpg")
    assert image is not None
    tensor, _ = preprocess(image)
    backend = TensorRTBackend("models/pose-fp16.engine")
    try:
        first = backend.infer(tensor).output
        original = first.copy()
        backend.infer(np.zeros_like(tensor))
        np.testing.assert_array_equal(first, original)
        with pytest.raises(ValueError, match="contiguous"):
            backend.infer(tensor.astype(np.float16))
    finally:
        backend.close()

