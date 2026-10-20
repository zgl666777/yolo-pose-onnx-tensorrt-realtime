#include "poseguard/engine.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <mutex>

namespace poseguard {
namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};
struct DeviceBuffer {
    void* ptr = nullptr;
    void allocate(std::size_t bytes) { check(cudaMalloc(&ptr, bytes)); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreate(&value)); }
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { if (value) cudaEventDestroy(value); }
};
}

struct Engine::Impl {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    DeviceBuffer input, output;
    Stream stream;
    std::array<Event, 4> events;
    std::size_t count = 0;
    int size;
    std::mutex mutex;
    Impl(const std::string& path, int image_size) : size(image_size) {
        if (size < 32 || size % 32) throw std::runtime_error("Invalid input image size");
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open TensorRT engine: " + path);
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("Cannot create TensorRT runtime");
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine) throw std::runtime_error("Cannot deserialize engine; rebuild on target GPU");
        context.reset(engine->createExecutionContext());
        if (!context || engine->getNbIOTensors() != 2) throw std::runtime_error("Expected one input/output");
        const char* input_name = nullptr;
        const char* output_name = nullptr;
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const auto name = engine->getIOTensorName(i);
            if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
                throw std::runtime_error("Engine I/O must be float32");
            if (engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR ||
                engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE)
                throw std::runtime_error("Expected linear device I/O");
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name = name;
            else output_name = name;
        }
        if (!input_name || !output_name) throw std::runtime_error("Invalid tensor modes");
        const auto in = engine->getTensorShape(input_name);
        const auto out = engine->getTensorShape(output_name);
        if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3 || in.d[2] != size || in.d[3] != size)
            throw std::runtime_error("Expected static [1,3,size,size] input");
        if (out.nbDims != 3 || out.d[0] != 1 || out.d[1] != 56 || out.d[2] <= 0)
            throw std::runtime_error("C++ requires raw [1,56,N] COCO pose output");
        count = static_cast<std::size_t>(out.d[1]) * out.d[2];
        input.allocate(static_cast<std::size_t>(3) * size * size * sizeof(float));
        output.allocate(count * sizeof(float));
        if (!context->setTensorAddress(input_name, input.ptr) || !context->setTensorAddress(output_name, output.ptr))
            throw std::runtime_error("Cannot bind device I/O");
    }
    ~Impl() { cudaStreamSynchronize(stream.value); }
};

Engine::Engine(const std::string& path, int size) : impl_(std::make_unique<Impl>(path, size)) {}
Engine::~Engine() = default;
std::size_t Engine::output_size() const { return impl_->count; }
int Engine::image_size() const { return impl_->size; }
std::array<float, 3> Engine::infer(const float* input, std::size_t input_count,
                                 float* output, std::size_t output_count) {
    auto& p = *impl_;
    const auto expected = static_cast<std::size_t>(3) * p.size * p.size;
    if (!input || !output || input_count != expected || output_count != p.count)
        throw std::runtime_error("Input/output buffer size mismatch");
    std::lock_guard<std::mutex> lock(p.mutex);
    check(cudaEventRecord(p.events[0].value, p.stream.value));
    check(cudaMemcpyAsync(p.input.ptr, input, expected * sizeof(float), cudaMemcpyHostToDevice, p.stream.value));
    check(cudaEventRecord(p.events[1].value, p.stream.value));
    if (!p.context->enqueueV3(p.stream.value)) throw std::runtime_error("TensorRT enqueueV3 failed");
    check(cudaEventRecord(p.events[2].value, p.stream.value));
    check(cudaMemcpyAsync(output, p.output.ptr, p.count * sizeof(float), cudaMemcpyDeviceToHost, p.stream.value));
    check(cudaEventRecord(p.events[3].value, p.stream.value));
    check(cudaStreamSynchronize(p.stream.value));
    std::array<float, 3> times{};
    for (int i = 0; i < 3; ++i)
        check(cudaEventElapsedTime(&times[i], p.events[i].value, p.events[i + 1].value));
    return times;
}
}

