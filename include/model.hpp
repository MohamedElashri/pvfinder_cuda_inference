#pragma once
#include "common.hpp"
#include "layers/conv.hpp"
#include "layers/deconv.hpp"
#include "layers/linear.hpp"
#include "layers/pooling.hpp"
#include "layers/activation.hpp"
#include <memory>
#include <cnpy.h>

namespace pvfinder {

class PVFinderModel {
public:
    PVFinderModel();
    ~PVFinderModel() = default;

    void loadWeights(const std::string& path);
    void forward(const Tensor& input, Tensor& output);

private:
    // Linear layers
    std::unique_ptr<LinearLayer> layer1;
    std::unique_ptr<LinearLayer> layer2;
    std::unique_ptr<LinearLayer> layer3;
    std::unique_ptr<LinearLayer> layer4;
    std::unique_ptr<LinearLayer> layer5;
    std::unique_ptr<LinearLayer> layer6A;

    // U-Net layers
    std::unique_ptr<ConvLayer> rcbn1;
    std::unique_ptr<ConvLayer> rcbn2;
    std::unique_ptr<ConvLayer> rcbn3;
    
    std::unique_ptr<MaxPoolLayer> pool;
    
    std::unique_ptr<DeconvLayer> up1;
    std::unique_ptr<DeconvLayer> up2;
    
    std::unique_ptr<ConvLayer> outIntermediate;
    std::unique_ptr<ConvLayer> outFinal;
};

} // namespace pvfinder