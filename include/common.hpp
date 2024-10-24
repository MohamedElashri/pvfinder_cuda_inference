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

    Tensor(int n, int c, int h, int w) : data(nullptr) {
        dims[0] = n;
        dims[1] = c;
        dims[2] = h;
        dims[3] = w;
        size = n * c * h * w * sizeof(float);
        
        if (size == 0) {
            throw std::runtime_error("Attempting to create tensor with zero size");
        }
        
        CUDA_CHECK(cudaMalloc(&data, size));
        if (data == nullptr) {
            throw std::runtime_error("Failed to allocate tensor memory");
        }
    }

    ~Tensor() {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
    }

    // Disable copy constructor and assignment
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Allow move constructor and assignment
    Tensor(Tensor&& other) noexcept 
        : data(other.data), size(other.size) {
        std::copy(other.dims, other.dims + 4, dims);
        other.data = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (data) {
                cudaFree(data);
            }
            data = other.data;
            size = other.size;
            std::copy(other.dims, other.dims + 4, dims);
            other.data = nullptr;
        }
        return *this;
    }
};

} // namespace pvfinder