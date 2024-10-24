#pragma once
#include "../common.hpp"
#include "/usr/include/cudnn.h"
#include <vector>

namespace pvfinder {

class ConvLayer {
public:
    ConvLayer(int in_channels, int out_channels, int kernel_size);
    ~ConvLayer();
    
    void loadWeights(const float* w, const float* b);
    void forward(const Tensor& input, Tensor& output);
    
private:
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnTensorDescriptor_t biasDesc;
    
    float *weights{nullptr};
    float *bias{nullptr};
    
    int in_channels;
    int out_channels;
    int kernel_size;

    // Helper function to find the best convolution algorithm
    cudnnConvolutionFwdAlgoPerf_t getBestAlgorithm(
        const cudnnTensorDescriptor_t& inputDesc,
        const cudnnFilterDescriptor_t& filterDesc,
        const cudnnConvolutionDescriptor_t& convDesc,
        const cudnnTensorDescriptor_t& outputDesc,
        void* workspace,
        size_t workspaceSize);
};

} // namespace pvfinder