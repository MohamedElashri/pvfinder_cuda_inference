#include "layers/conv.hpp"
#include <iostream>

namespace pvfinder {

__global__ void conv1dForwardKernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size) {
    
    // Calculate global position
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Get output position
    const int n = idx / (out_channels * input_length);  // batch index
    const int remain = idx % (out_channels * input_length);
    const int oc = remain / input_length;               // output channel
    const int ol = remain % input_length;               // output length position
    
    if (n >= batch_size || oc >= out_channels || ol >= input_length) {
        return;
    }

    float sum = bias[oc];  // Initialize with bias
    const int pad = kernel_size / 2;
    
    // Compute convolution
    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            const int il = ol - pad + k;  // input length position
            
            if (il >= 0 && il < input_length) {
                const int input_idx = ((n * in_channels + ic) * input_length + il);
                const int weight_idx = ((oc * in_channels + ic) * kernel_size + k);
                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }
    
    // Write output
    const int output_idx = ((n * out_channels + oc) * input_length + ol);
    output[output_idx] = sum;
}

ConvLayer::ConvLayer(int in_ch, int out_ch, int kernel) 
    : in_channels(in_ch), out_channels(out_ch), kernel_size(kernel) {
    
    // Allocate weights and bias
    size_t weights_size = out_channels * in_channels * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&weights, weights_size));
    CUDA_CHECK(cudaMalloc(&bias, bias_size));
}

ConvLayer::~ConvLayer() {
    if (weights) CUDA_CHECK(cudaFree(weights));
    if (bias) CUDA_CHECK(cudaFree(bias));
}

void ConvLayer::loadWeights(const float* w, const float* b) {
    size_t weights_size = out_channels * in_channels * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    
    CUDA_CHECK(cudaMemcpy(weights, w, weights_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias, b, bias_size, cudaMemcpyHostToDevice));
}

void ConvLayer::forward(const Tensor& input, Tensor& output) {
    std::cout << "Input dimensions: N=" << input.dims[0] 
              << ", C=" << input.dims[1]
              << ", H=" << input.dims[2]
              << ", W=" << input.dims[3] << std::endl;
              
    std::cout << "Output dimensions: N=" << output.dims[0]
              << ", C=" << output.dims[1]
              << ", H=" << output.dims[2]
              << ", W=" << output.dims[3] << std::endl;
              
    std::cout << "Filter dimensions: out_channels=" << out_channels
              << ", in_channels=" << in_channels
              << ", kernel_size=" << kernel_size << std::endl;

    // Launch parameters
    const int batch_size = input.dims[0];
    const int input_length = input.dims[2];  // Using H dimension for sequence length
    const int total_elements = batch_size * out_channels * input_length;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    conv1dForwardKernel<<<blocks, threads_per_block>>>(
        input.data,
        weights,
        bias,
        output.data,
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size
    );

    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace pvfinder
