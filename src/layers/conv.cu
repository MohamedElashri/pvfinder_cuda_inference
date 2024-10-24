#include "layers/conv.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

namespace pvfinder {

ConvLayer::ConvLayer(int in_ch, int out_ch, int kernel) 
    : in_channels(in_ch), out_channels(out_ch), kernel_size(kernel) {
    
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&biasDesc));
    
    // Set up convolution descriptor for 1D convolution
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        convDesc,
        kernel_size/2, 0,     // zero padding height, width
        1, 1,                 // stride height, width
        1, 1,                 // dilation height, width
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    
    // Set up filter descriptor
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filterDesc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        out_channels,
        in_channels,
        1,              // filter height = 1
        kernel_size));  // filter width = kernel_size
    
    // Set up bias descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        biasDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, out_channels, 1, 1));
    
    // Allocate weights and bias
    size_t weights_size = out_channels * in_channels * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&weights, weights_size));
    CUDA_CHECK(cudaMalloc(&bias, bias_size));
}

ConvLayer::~ConvLayer() {
    if (weights) CUDA_CHECK(cudaFree(weights));
    if (bias) CUDA_CHECK(cudaFree(bias));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(biasDesc));
}

void ConvLayer::loadWeights(const float* w, const float* b) {
    size_t weights_size = out_channels * in_channels * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    
    CUDA_CHECK(cudaMemcpy(weights, w, weights_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias, b, bias_size, cudaMemcpyHostToDevice));
}

cudnnConvolutionFwdAlgoPerf_t ConvLayer::getBestAlgorithm(
    const cudnnTensorDescriptor_t& inputDesc,
    const cudnnFilterDescriptor_t& filterDesc,
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& outputDesc,
    void* workspace,
    size_t workspaceSize) {
    
    // Query how many algorithms are available
    int algo_count;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &algo_count));
    
    // Get all available algorithms
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perfResults(algo_count);
    int returnedAlgoCount = 0;
    
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        cudnnHandle,
        inputDesc,
        filterDesc,
        convDesc,
        outputDesc,
        algo_count,
        &returnedAlgoCount,
        perfResults.data()));
    
    // Find the first algorithm that works with our workspace constraint
    for (int i = 0; i < returnedAlgoCount; i++) {
        std::cout << "Algorithm " << i << ": status = " << perfResults[i].status 
                << ", memory = " << perfResults[i].memory << std::endl;
        if (perfResults[i].status == CUDNN_STATUS_SUCCESS && perfResults[i].memory <= workspaceSize) {
            return perfResults[i];
        }
    }
    
    throw std::runtime_error("Could not find a suitable convolution algorithm");
}

void ConvLayer::forward(const Tensor& input, Tensor& output) {
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    
    // Debug input dimensions
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

    // Set tensor descriptors with explicit strides
    const int inputStride[4] = { 
        input.dims[1] * input.dims[2] * input.dims[3],  // nStride
        input.dims[2] * input.dims[3],                  // cStride
        input.dims[3],                                  // hStride
        1                                               // wStride
    };
    
    const int outputStride[4] = {
        output.dims[1] * output.dims[2] * output.dims[3],  // nStride
        output.dims[2] * output.dims[3],                   // cStride
        output.dims[3],                                    // hStride
        1                                                  // wStride
    };

    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        inputDesc,
        CUDNN_DATA_FLOAT,
        input.dims[0],    // batch size
        input.dims[1],    // channels
        input.dims[2],    // height (sequence length)
        input.dims[3],    // width
        inputStride[0], inputStride[1], inputStride[2], inputStride[3]
    ));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        outputDesc,
        CUDNN_DATA_FLOAT,
        output.dims[0],   // batch size
        output.dims[1],   // channels
        output.dims[2],   // height (sequence length)
        output.dims[3],   // width
        outputStride[0], outputStride[1], outputStride[2], outputStride[3]
    ));

    // Set filter descriptor
    const int filterDims[4] = {
        out_channels,     // output channels
        in_channels,      // input channels
        kernel_size,      // kernel height (kernel size for 1D)
        1                 // kernel width
    };
    
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(
        filterDesc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        4,                // nbDims
        filterDims
    ));

    // Set convolution descriptor
    int pad[2] = {kernel_size / 2, 0};    // padding height, width
    int stride[2] = {1, 1};               // stride height, width
    int dilation[2] = {1, 1};             // dilation height, width
    
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
        convDesc,
        2,               // nbDims
        pad,            // padding
        stride,         // stride
        dilation,       // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    // Get output dimensions
    int n, c, h, w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        convDesc,
        inputDesc,
        filterDesc,
        &n, &c, &h, &w));

    std::cout << "Calculated output dimensions: N=" << n 
              << ", C=" << c
              << ", H=" << h
              << ", W=" << w << std::endl;

    // Try different algorithms in order of preference
    const cudnnConvolutionFwdAlgo_t algos[] = {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
    };

    // Try each algorithm until one works
    cudnnStatus_t status = CUDNN_STATUS_NOT_SUPPORTED;
    cudnnConvolutionFwdAlgo_t selectedAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_size = 0;

    for (auto algo : algos) {
        status = cudnnGetConvolutionForwardWorkspaceSize(
            cudnnHandle,
            inputDesc,
            filterDesc,
            convDesc,
            outputDesc,
            algo,
            &workspace_size);
            
        if (status == CUDNN_STATUS_SUCCESS) {
            selectedAlgo = algo;
            std::cout << "Selected algorithm: " << algo 
                      << " with workspace size: " << workspace_size << std::endl;
            break;
        }
    }

    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Could not find a suitable convolution algorithm");
    }

    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    float alpha = 1.0f, beta = 0.0f;
    
    // Perform convolution
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnnHandle,
        &alpha,
        inputDesc,
        input.data,
        filterDesc,
        weights,
        convDesc,
        selectedAlgo,
        workspace,
        workspace_size,
        &beta,
        outputDesc,
        output.data));

    // Add bias
    CUDNN_CHECK(cudnnAddTensor(
        cudnnHandle,
        &alpha,
        biasDesc,
        bias,
        &alpha,
        outputDesc,
        output.data));

    // Cleanup
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
}

} // namespace pvfinder