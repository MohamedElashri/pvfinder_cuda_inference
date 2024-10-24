#include "layers/deconv.hpp"

namespace pvfinder {

DeconvLayer::DeconvLayer(int in_ch, int out_ch, int kernel) 
    : in_channels(in_ch), out_channels(out_ch), kernel_size(kernel) {
    
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&biasDesc));
    
    // Set up convolution descriptor for transposed convolution
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
        0, 0,           // padding
        2, 1,           // stride (2 for upsampling)
        1, 1,           // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    
    // For transposed convolution, we swap in/out channels in filter
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        in_channels, out_channels,  // swapped for transposed conv
        kernel_size, 1));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, out_channels, 1, 1));
    
    // Allocate weights and bias
    size_t weights_size = in_channels * out_channels * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&weights, weights_size));
    CUDA_CHECK(cudaMalloc(&bias, bias_size));
}

DeconvLayer::~DeconvLayer() {
    if (weights) CUDA_CHECK(cudaFree(weights));
    if (bias) CUDA_CHECK(cudaFree(bias));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(biasDesc));
}

void DeconvLayer::loadWeights(const float* w, const float* b) {
    size_t weights_size = in_channels * out_channels * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    
    CUDA_CHECK(cudaMemcpy(weights, w, weights_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias, b, bias_size, cudaMemcpyHostToDevice));
}

void DeconvLayer::forward(const Tensor& input, Tensor& output) {
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        input.dims[0], input.dims[1], input.dims[2], input.dims[3]));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        output.dims[0], output.dims[1], output.dims[2], output.dims[3]));
    
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
        filterDesc,
        inputDesc,
        convDesc,
        outputDesc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        &workspace_size));
    
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Transposed convolution using backward data algorithm
    CUDNN_CHECK(cudnnConvolutionBackwardData(cudnnHandle,
        &alpha,
        filterDesc,
        weights,
        inputDesc,
        input.data,
        convDesc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        workspace,
        workspace_size,
        &beta,
        outputDesc,
        output.data));
    
    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnnHandle,
        &alpha,
        biasDesc,
        bias,
        &alpha,
        outputDesc,
        output.data));
    
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
}

} // namespace pvfinder