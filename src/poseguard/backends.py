from __future__ import annotations

import ctypes
from pathlib import Path
from time import perf_counter
import numpy as np
from .geometry import raw_rows
from .types import Inference
from .runtime import configure_gpu_runtime


def validate_tensor(tensor: np.ndarray, size: int) -> None:
    if tensor.shape != (1, 3, size, size) or tensor.dtype != np.float32 or not tensor.flags.c_contiguous:
        raise ValueError(f"Input must be contiguous float32 [1,3,{size},{size}]")


class ONNXBackend:
    def __init__(self, model: str, size: int = 640, provider: str = "CPUExecutionProvider"):
        import onnxruntime as ort
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"Unavailable ORT provider {provider}; available: {ort.get_available_providers()}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=[provider])
        if len(self.session.get_inputs()) != 1 or len(self.session.get_outputs()) != 1:
            raise ValueError("Expected one input and one raw pose output")
        inp = self.session.get_inputs()[0]
        expected = [1, 3, size, size]
        if inp.type != "tensor(float)" or list(inp.shape) != expected:
            raise ValueError(f"Export static float32 input {expected}; got {inp.type} {inp.shape}")
        self.input_name, self.size = inp.name, size

    def infer(self, tensor: np.ndarray) -> Inference:
        validate_tensor(tensor, self.size)
        start = perf_counter()
        try:
            output = self.session.run(None, {self.input_name: tensor})[0]
        except Exception as exc:
            raise RuntimeError(f"ONNX Runtime inference failed: {exc}") from exc
        elapsed = (perf_counter() - start) * 1000
        raw_rows(output)
        return Inference(output, {"backend_ms": elapsed})

    def close(self):
        self.session = None


def cuda_runtime():
    configure_gpu_runtime()
    try:
        from cuda.bindings import runtime
    except ImportError:
        from cuda import cudart as runtime
    return runtime


class Cuda:
    def __init__(self):
        self.api = cuda_runtime()

    def call(self, name: str, *args):
        result = getattr(self.api, name)(*args)
        if int(result[0]) != 0:
            raise RuntimeError(f"{name} failed with CUDA error {result[0]}")
        values = result[1:]
        return values[0] if len(values) == 1 else values


class TensorRTBackend:
    """TensorRT 10 name-based I/O with explicit ownership and CUDA events."""

    def __init__(self, model: str, size: int = 640):
        configure_gpu_runtime()
        import tensorrt as trt
        self.trt, self.size = trt, size
        self.cuda = Cuda()
        self.stream = None
        self.pointers, self.events = [], []
        try:
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(model).read_bytes())
            if self.engine is None:
                raise RuntimeError("TensorRT could not deserialize engine; rebuild on this GPU")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT could not create execution context")
            inputs, outputs = [], []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                (inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(name)
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("Expected one input and one output")
            self.input_name, self.output_name = inputs[0], outputs[0]
            if tuple(self.engine.get_tensor_shape(self.input_name)) != (1, 3, size, size):
                raise ValueError("Engine must have static batch-one NCHW shape")
            for name in (self.input_name, self.output_name):
                if self.engine.get_tensor_dtype(name) != trt.float32:
                    raise ValueError("Engine I/O must be float32 (FP16 internal computation is supported)")
                if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR or self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                    raise ValueError("Engine tensors must use linear device memory")
            shape = tuple(self.context.get_tensor_shape(self.output_name))
            if any(d <= 0 for d in shape):
                raise ValueError("Dynamic engines are unsupported; use poseguard build-engine")
            self.output = np.empty(shape, dtype=np.float32)
            raw_rows(self.output)
            self.stream = self.cuda.call("cudaStreamCreate")
            self.input_ptr = self.cuda.call("cudaMalloc", 3 * size * size * 4)
            self.pointers.append(self.input_ptr)
            self.output_ptr = self.cuda.call("cudaMalloc", self.output.nbytes)
            self.pointers.append(self.output_ptr)
            for name, ptr in ((self.input_name, self.input_ptr), (self.output_name, self.output_ptr)):
                if not self.context.set_tensor_address(name, int(ptr)):
                    raise RuntimeError(f"Cannot bind tensor {name}")
            for _ in range(4):
                self.events.append(self.cuda.call("cudaEventCreate"))
        except Exception:
            self.close()
            raise

    def infer(self, tensor: np.ndarray) -> Inference:
        validate_tensor(tensor, self.size)
        start = perf_counter()
        api, call = self.cuda.api, self.cuda.call
        call("cudaEventRecord", self.events[0], self.stream)
        call("cudaMemcpyAsync", self.input_ptr, tensor.ctypes.data, tensor.nbytes,
             api.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        call("cudaEventRecord", self.events[1], self.stream)
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError("TensorRT execution failed")
        call("cudaEventRecord", self.events[2], self.stream)
        call("cudaMemcpyAsync", self.output.ctypes.data, self.output_ptr, self.output.nbytes,
             api.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        call("cudaEventRecord", self.events[3], self.stream)
        call("cudaStreamSynchronize", self.stream)
        times = {key: float(call("cudaEventElapsedTime", self.events[i], self.events[i + 1]))
                 for i, key in enumerate(("h2d_ms", "gpu_ms", "d2h_ms"))}
        times["backend_ms"] = (perf_counter() - start) * 1000
        return Inference(self.output.copy(), times)

    def close(self):
        if self.stream is not None:
            try:
                self.cuda.call("cudaStreamSynchronize", self.stream)
            except RuntimeError:
                pass
        for event in self.events:
            self.cuda.call("cudaEventDestroy", event)
        self.events.clear()
        for ptr in self.pointers:
            self.cuda.call("cudaFree", ptr)
        self.pointers.clear()
        if self.stream is not None:
            self.cuda.call("cudaStreamDestroy", self.stream)
            self.stream = None
        self.context = self.engine = self.runtime = None


class CppBackend:
    def __init__(self, model: str, size: int = 640, library: str | None = None):
        configure_gpu_runtime()
        if not library:
            raise ValueError("Set library to build/cpp/libpose_trt.so")
        self.size = size
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.pg_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.pg_create.restype = ctypes.c_void_p
        self.lib.pg_output_size.argtypes = [ctypes.c_void_p]
        self.lib.pg_output_size.restype = ctypes.c_size_t
        self.lib.pg_infer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_float)]
        self.lib.pg_infer.restype = ctypes.c_int
        self.lib.pg_last_error.restype = ctypes.c_char_p
        self.lib.pg_destroy.argtypes = [ctypes.c_void_p]
        self.handle = self.lib.pg_create(str(model).encode("utf-8"), size)
        if not self.handle:
            raise RuntimeError(self.lib.pg_last_error().decode("utf-8"))
        count = self.lib.pg_output_size(self.handle)
        if count == 0 or count % 56:
            self.close()
            raise ValueError("Invalid C++ output shape")
        self.output = np.empty((1, 56, count // 56), dtype=np.float32)

    def infer(self, tensor: np.ndarray) -> Inference:
        validate_tensor(tensor, self.size)
        times = np.zeros(3, dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        start = perf_counter()
        result = self.lib.pg_infer(self.handle, tensor.ctypes.data_as(pointer), tensor.size,
                                   self.output.ctypes.data_as(pointer), self.output.size,
                                   times.ctypes.data_as(pointer))
        if result:
            raise RuntimeError(self.lib.pg_last_error().decode("utf-8"))
        return Inference(self.output.copy(), dict(zip(("h2d_ms", "gpu_ms", "d2h_ms"), map(float, times)),
                                                  backend_ms=(perf_counter() - start) * 1000))

    def close(self):
        if self.handle:
            self.lib.pg_destroy(self.handle)
            self.handle = None


def make_backend(config):
    if config.backend == "onnx":
        return ONNXBackend(config.model, config.imgsz, config.ort_provider)
    if config.backend == "trt":
        return TensorRTBackend(config.model, config.imgsz)
    return CppBackend(config.model, config.imgsz, config.library)
