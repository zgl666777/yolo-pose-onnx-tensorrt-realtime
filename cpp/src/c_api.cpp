#include "poseguard/c_api.h"
#include "poseguard/engine.hpp"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

namespace { thread_local std::string error; }
extern "C" {
void* pg_create(const char* path, int size) {
    try {
        error.clear();
        if (!path) throw std::runtime_error("Null engine path");
        return new poseguard::Engine(path, size);
    } catch (const std::exception& e) { error = e.what(); return nullptr; }
}
size_t pg_output_size(void* handle) {
    return handle ? static_cast<poseguard::Engine*>(handle)->output_size() : 0;
}
int pg_infer(void* handle, const float* input, size_t input_count,
             float* output, size_t output_count, float* stage_ms) {
    try {
        error.clear();
        if (!handle || !stage_ms) throw std::runtime_error("Null handle or timing buffer");
        auto times = static_cast<poseguard::Engine*>(handle)->infer(input, input_count, output, output_count);
        std::copy(times.begin(), times.end(), stage_ms);
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
const char* pg_last_error(void) { return error.c_str(); }
void pg_destroy(void* handle) { delete static_cast<poseguard::Engine*>(handle); }
}
