#pragma once
#include "../common.hpp"

namespace pvfinder {

class LinearLayer {
public:
    LinearLayer(int in_features, int out_features);
    ~LinearLayer();
    
    void loadWeights(const float* w, const float* b);
    void forward(const Tensor& input, Tensor& output);

private:
    float *weights{nullptr};
    float *bias{nullptr};
    int in_features;
    int out_features;
};

} // namespace pvfinder