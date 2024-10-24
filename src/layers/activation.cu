#include "layers/activation.hpp"
#include <iostream>

namespace pvfinder {

__global__ void leakyReLUKernel(float* x, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = x[idx] > 0 ? x[idx] : alpha * x[idx];
    }
}

__global__ void softplusKernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Numerical stability improvement
        const float THRESHOLD = 20.0f;  // Threshold for numerical stability
        if (x[idx] > THRESHOLD) {
            x[idx] = x[idx];  // For large x, softplus(x) ≈ x
        } else if (x[idx] < -THRESHOLD) {
            x[idx] = 0.0f;    // For very negative x, softplus(x) ≈ 0
        } else {
            x[idx] = logf(1.0f + expf(x[idx]));
        }
    }
}

__global__ void scaleKernel(float* x, int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = x[idx] * factor;
        // Clamp values to prevent infinity
        if (x[idx] > 1e6f) x[idx] = 1e6f;
        if (x[idx] < -1e6f) x[idx] = -1e6f;
    }
}

void leakyReLU(Tensor& x, float alpha) {
    int total_size = x.dims[0] * x.dims[1] * x.dims[2] * x.dims[3];
    int threads_per_block = 256;
    int blocks = (total_size + threads_per_block - 1) / threads_per_block;
    leakyReLUKernel<<<blocks, threads_per_block>>>(x.data, total_size, alpha);
    CUDA_CHECK(cudaGetLastError());
}

void softplus(Tensor& x) {
    int total_size = x.dims[0] * x.dims[1] * x.dims[2] * x.dims[3];
    int threads_per_block = 256;
    int blocks = (total_size + threads_per_block - 1) / threads_per_block;
    softplusKernel<<<blocks, threads_per_block>>>(x.data, total_size);
    CUDA_CHECK(cudaGetLastError());
}

void scale(Tensor& x, float factor) {
    int total_size = x.dims[0] * x.dims[1] * x.dims[2] * x.dims[3];
    int threads_per_block = 256;
    int blocks = (total_size + threads_per_block - 1) / threads_per_block;
    scaleKernel<<<blocks, threads_per_block>>>(x.data, total_size, factor);
    CUDA_CHECK(cudaGetLastError());
}

void concatenate(Tensor& dest, const Tensor& src) {
    // Verify dimensions
    if (dest.dims[0] != src.dims[0] || 
        dest.dims[2] != src.dims[2] || 
        dest.dims[3] != src.dims[3]) {
        throw std::runtime_error("Incompatible dimensions for concatenation");
    }

    int elements_per_channel = dest.dims[2] * dest.dims[3];
    int dest_offset_channels = dest.dims[1] - src.dims[1];

    for (int n = 0; n < dest.dims[0]; ++n) {
        float* dest_ptr = dest.data + 
                         n * dest.dims[1] * elements_per_channel + 
                         dest_offset_channels * elements_per_channel;
        const float* src_ptr = src.data + 
                              n * src.dims[1] * elements_per_channel;
        
        size_t copy_size = src.dims[1] * elements_per_channel * sizeof(float);
        CUDA_CHECK(cudaMemcpy(dest_ptr, src_ptr, copy_size, cudaMemcpyDeviceToDevice));
    }
}

Tensor reshapeToUNet(const Tensor& input) {
    Tensor reshaped(input.dims[0], 8, 100, 1);
    CUDA_CHECK(cudaMemcpy(
        reshaped.data,
        input.data,
        input.size,
        cudaMemcpyDeviceToDevice));
    return reshaped;
}

} // namespace pvfinder