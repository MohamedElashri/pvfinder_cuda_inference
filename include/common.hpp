// include/common.hpp
#pragma once
#include <cuda_runtime.h>
#include "/usr/include/cudnn.h"
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <cassert>


namespace pvfinder {

// Global cuDNN handle
extern cudnnHandle_t cudnnHandle;

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct Tensor {
    float* data;
    int dims[4];  // NCHW format
    size_t size;

    Tensor(int n, int c, int h, int w) {
        dims[0] = n; dims[1] = c; dims[2] = h; dims[3] = w;
        size = n * c * h * w * sizeof(float);
        CUDA_CHECK(cudaMalloc(&data, size));
    }

    ~Tensor() {
        if (data) CUDA_CHECK(cudaFree(data));
    }
};

} // namespace pvfinder