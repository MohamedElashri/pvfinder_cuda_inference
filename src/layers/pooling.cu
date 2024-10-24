#include "layers/pooling.hpp"

namespace pvfinder {

MaxPoolLayer::MaxPoolLayer(int kernel) : kernel_size(kernel) {
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    
    // Note: For 1D pooling, we use width=1 for other dimensions
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        poolingDesc,
        CUDNN_POOLING_MAX,          // mode
        CUDNN_NOT_PROPAGATE_NAN,    // nanPropagation
        1,              // window height (1 for 1D)
        kernel_size,    // window width (our actual pooling size)
        0,              // vertical padding
        0,              // horizontal padding
        1,              // vertical stride
        kernel_size     // horizontal stride (same as window width)
    ));
}

MaxPoolLayer::~MaxPoolLayer() {
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}

void MaxPoolLayer::forward(const Tensor& input, Tensor& output) {
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    
    // For 1D data, treat it as 2D with width=1
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        inputDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        input.dims[0],    // batch size
        input.dims[1],    // channels
        1,                // height (1 for 1D)
        input.dims[2]     // width (our actual data size)
    ));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        outputDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        output.dims[0],   // batch size
        output.dims[1],   // channels
        1,                // height (1 for 1D)
        output.dims[2]    // width (our actual data size)
    ));
    
    float alpha = 1.0f, beta = 0.0f;
    
    CUDNN_CHECK(cudnnPoolingForward(cudnnHandle,
        poolingDesc,
        &alpha,
        inputDesc,
        input.data,
        &beta,
        outputDesc,
        output.data));
    
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
}

} // namespace pvfinder