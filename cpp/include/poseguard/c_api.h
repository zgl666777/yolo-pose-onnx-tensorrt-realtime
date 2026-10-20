#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void* pg_create(const char* path, int image_size);
size_t pg_output_size(void* handle);
int pg_infer(void* handle, const float* input, size_t input_count,
             float* output, size_t output_count, float* stage_ms);
const char* pg_last_error(void);
void pg_destroy(void* handle);
#ifdef __cplusplus
}
#endif

