#include "layers/deconv.hpp"
#include <iostream>

namespace pvfinder {

__global__ void deconv1dForwardKernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size) {
    
    // Calculate global position
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Get output position
    const int n = idx / (out_channels * output_length);  // batch index
    const int remain = idx % (out_channels * output_length);
    const int oc = remain / output_length;               // output channel
    const int ol = remain % output_length;               // output length position
    
    if (n >= batch_size || oc >= out_channels || ol >= output_length) {
        return;
    }

    float sum = bias[oc];  // Initialize with bias

    // For deconv, we need to consider the stride
    const int stride = 2;  // Since we're upsampling by factor of 2
    const int il_start = ol / stride;  // Starting input position
    const int il_end = min(il_start + kernel_size, input_length);
    
    // Compute transposed convolution
    for (int ic = 0; ic < in_channels; ic++) {
        for (int il = il_start; il < il_end; il++) {
            const int k = ol - il * stride;  // kernel position
            if (k >= 0 && k < kernel_size) {
                const int input_idx = ((n * in_channels + ic) * input_length + il);
                const int weight_idx = ((oc * in_channels + ic) * kernel_size + k);
                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }
    
    // Write output
    const int output_idx = ((n * out_channels + oc) * output_length + ol);
    output[output_idx] = sum;
}

DeconvLayer::DeconvLayer(int in_ch, int out_ch, int kernel) 
    : in_channels(in_ch), out_channels(out_ch), kernel_size(kernel) {
    
    // Only allocate weights and bias
    size_t weights_size = out_channels * in_channels * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&weights, weights_size));
    CUDA_CHECK(cudaMalloc(&bias, bias_size));
}

DeconvLayer::~DeconvLayer() {
    if (weights) CUDA_CHECK(cudaFree(weights));
    if (bias) CUDA_CHECK(cudaFree(bias));
}

void DeconvLayer::loadWeights(const float* w, const float* b) {
    size_t weights_size = out_channels * in_channels * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    
    CUDA_CHECK(cudaMemcpy(weights, w, weights_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias, b, bias_size, cudaMemcpyHostToDevice));
}

void DeconvLayer::forward(const Tensor& input, Tensor& output) {
    std::cout << "Deconv Input dimensions: N=" << input.dims[0] 
              << ", C=" << input.dims[1]
              << ", H=" << input.dims[2]
              << ", W=" << input.dims[3] << std::endl;
              
    std::cout << "Deconv Output dimensions: N=" << output.dims[0]
              << ", C=" << output.dims[1]
              << ", H=" << output.dims[2]
              << ", W=" << output.dims[3] << std::endl;
              
    std::cout << "Deconv Filter dimensions: out_channels=" << out_channels
              << ", in_channels=" << in_channels
              << ", kernel_size=" << kernel_size << std::endl;

    // Launch parameters
    const int batch_size = input.dims[0];
    const int input_length = input.dims[2];
    const int output_length = output.dims[2];  // Should be 2x input_length
    const int total_elements = batch_size * out_channels * output_length;
    
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    deconv1dForwardKernel<<<blocks, threads_per_block>>>(
        input.data,
        weights,
        bias,
        output.data,
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size
    );

    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace pvfinder
