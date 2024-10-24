#pragma once
#include "../common.hpp"

namespace pvfinder {

class DeconvLayer {
public:
    DeconvLayer(int in_channels, int out_channels, int kernel_size);
    ~DeconvLayer();
    
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
};

} // namespace pvfinder