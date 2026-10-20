#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void add_one(float* values) {
    const int i = threadIdx.x;
    if (i < 4) values[i] += 1.0f;
}
int main() {
    float* device = nullptr;
    try {
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0));
        float values[] = {0, 1, 2, 3};
        check(cudaMalloc(&device, sizeof(values)));
        check(cudaMemcpy(device, values, sizeof(values), cudaMemcpyHostToDevice));
        add_one<<<1, 4>>>(device);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(values, device, sizeof(values), cudaMemcpyDeviceToHost));
        check(cudaFree(device));
        device = nullptr;
        for (int i = 0; i < 4; ++i)
            if (values[i] != i + 1) throw std::runtime_error("Kernel output mismatch");
        std::cout << "CUDA kernel passed on " << properties.name << '\n';
    } catch (const std::exception& e) {
        if (device) cudaFree(device);
        std::cerr << e.what() << '\n';
        return 1;
    }
}

