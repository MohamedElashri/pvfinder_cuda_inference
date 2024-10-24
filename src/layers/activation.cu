#include "layers/activation.hpp"

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
        x[idx] = logf(1.0f + expf(x[idx]));
    }
}

__global__ void scaleKernel(float* x, int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] *= factor;
    }
}

void leakyReLU(Tensor& x, float alpha) {
    int threads = 256;
    int blocks = (x.size / sizeof(float) + threads - 1) / threads;
    leakyReLUKernel<<<blocks, threads>>>(x.data, x.size / sizeof(float), alpha);
}

void softplus(Tensor& x) {
    int threads = 256;
    int blocks = (x.size / sizeof(float) + threads - 1) / threads;
    softplusKernel<<<blocks, threads>>>(x.data, x.size / sizeof(float));
}

void scale(Tensor& x, float factor) {
    int threads = 256;
    int blocks = (x.size / sizeof(float) + threads - 1) / threads;
    scaleKernel<<<blocks, threads>>>(x.data, x.size / sizeof(float), factor);
}

void concatenate(Tensor& dest, const Tensor& src) {
    assert(dest.dims[0] == src.dims[0] && 
           dest.dims[2] == src.dims[2] && 
           dest.dims[3] == src.dims[3]);
    
    CUDA_CHECK(cudaMemcpy(
        dest.data + dest.dims[1] * dest.dims[2] * dest.dims[3],
        src.data,
        src.size,
        cudaMemcpyDeviceToDevice));
}

Tensor reshapeToUNet(const Tensor& input) {
    Tensor reshaped(input.dims[0], 8, 100, 1);
    CUDA_CHECK(cudaMemcpy(reshaped.data, input.data, input.size,
                         cudaMemcpyDeviceToDevice));
    return reshaped;
}

} // namespace pvfinder