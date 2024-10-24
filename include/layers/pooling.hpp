#pragma once
#include "../common.hpp"

namespace pvfinder {

class MaxPoolLayer {
public:
    MaxPoolLayer(int kernel_size);
    ~MaxPoolLayer();
    
    void forward(const Tensor& input, Tensor& output);

private:
    cudnnPoolingDescriptor_t poolingDesc;
    int kernel_size;
};

} // namespace pvfinder