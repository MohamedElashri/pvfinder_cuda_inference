// src/layers/linear.cu
#include "layers/linear.hpp"

namespace pvfinder {

LinearLayer::LinearLayer(int in_f, int out_f) 
    : in_features(in_f), out_features(out_f) {
    CUDA_CHECK(cudaMalloc(&weights, in_features * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias, out_features * sizeof(float)));
}

LinearLayer::~LinearLayer() {
    if (weights) CUDA_CHECK(cudaFree(weights));
    if (bias) CUDA_CHECK(cudaFree(bias));
}

void LinearLayer::loadWeights(const float* w, const float* b) {
    CUDA_CHECK(cudaMemcpy(weights, w,
                         in_features * out_features * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias, b,
                         out_features * sizeof(float),
                         cudaMemcpyHostToDevice));
}

__global__ void linearForwardKernel(const float* input,
                                  const float* weights,
                                  const float* bias,
                                  float* output,
                                  int batch_size,
                                  int in_features,
                                  int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;
    
    int batch_idx = idx / out_features;
    int feat_idx = idx % out_features;
    
    float sum = bias[feat_idx];
    for (int i = 0; i < in_features; ++i) {
        sum += input[batch_idx * in_features + i] *
               weights[feat_idx * in_features + i];
    }
    output[idx] = sum;
}

void LinearLayer::forward(const Tensor& input, Tensor& output) {
    int batch_size = input.dims[0];
    int threads = 256;
    int blocks = (batch_size * out_features + threads - 1) / threads;
    
    linearForwardKernel<<<blocks, threads>>>(
        input.data, weights, bias, output.data,
        batch_size, in_features, out_features);
}

} // namespace pvfinder