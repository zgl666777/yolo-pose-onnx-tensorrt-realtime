#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace poseguard {
class Engine {
public:
    explicit Engine(const std::string& path, int size);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    std::size_t output_size() const;
    int image_size() const;
    std::array<float, 3> infer(const float* input, std::size_t input_count,
                               float* output, std::size_t output_count);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

