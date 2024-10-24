#include "model.hpp"
#include <stdexcept>

namespace pvfinder {

PVFinderModel::PVFinderModel() {
    // Initialize all layers with proper dimensions
    // Linear layers
    layer1 = std::make_unique<LinearLayer>(9, 20);
    layer2 = std::make_unique<LinearLayer>(20, 20);
    layer3 = std::make_unique<LinearLayer>(20, 20);
    layer4 = std::make_unique<LinearLayer>(20, 20);
    layer5 = std::make_unique<LinearLayer>(20, 20);
    layer6A = std::make_unique<LinearLayer>(20, 800);  // 8 channels * 100 bins

    // UNet layers
    rcbn1 = std::make_unique<ConvLayer>(8, 64, 25);   // latentChannels to n, kernel=25
    rcbn2 = std::make_unique<ConvLayer>(64, 64, 7);   // n to n, kernel=7
    rcbn3 = std::make_unique<ConvLayer>(64, 64, 5);   // n to n, kernel=5
    
    pool = std::make_unique<MaxPoolLayer>(2);
    
    up1 = std::make_unique<DeconvLayer>(64, 64, 2);   // n to n, kernel=2
    up2 = std::make_unique<DeconvLayer>(128, 64, 2);  // 2n to n, kernel=2
    
    outIntermediate = std::make_unique<ConvLayer>(128, 64, 5);
    outFinal = std::make_unique<ConvLayer>(64, 1, 5);
}

void PVFinderModel::forward(const Tensor& input, Tensor& output) {
    // Temporary storage for intermediate results
    Tensor fc1_out(input.dims[0], 20, 1, 1);
    Tensor fc2_out(input.dims[0], 20, 1, 1);
    Tensor fc3_out(input.dims[0], 20, 1, 1);
    Tensor fc4_out(input.dims[0], 20, 1, 1);
    Tensor fc5_out(input.dims[0], 20, 1, 1);
    Tensor fc6_out(input.dims[0], 800, 1, 1);

    // Forward through dense layers
    layer1->forward(input, fc1_out);
    leakyReLU(fc1_out);
    
    layer2->forward(fc1_out, fc2_out);
    leakyReLU(fc2_out);
    
    layer3->forward(fc2_out, fc3_out);
    leakyReLU(fc3_out);
    
    layer4->forward(fc3_out, fc4_out);
    leakyReLU(fc4_out);
    
    layer5->forward(fc4_out, fc5_out);
    leakyReLU(fc5_out);
    
    layer6A->forward(fc5_out, fc6_out);
    leakyReLU(fc6_out);

    // Reshape for UNet
    Tensor unet_input = reshapeToUNet(fc6_out);  // (N, 8, 100, 1)
    
    // Down path
    Tensor down1(unet_input.dims[0], 64, 100, 1);
    rcbn1->forward(unet_input, down1);
    
    Tensor down2(down1.dims[0], 64, 50, 1);
    pool->forward(down1, down2);
    rcbn2->forward(down2, down2);
    
    Tensor down3(down2.dims[0], 64, 25, 1);
    pool->forward(down2, down3);
    rcbn3->forward(down3, down3);
    
    // Up path with skip connections
    Tensor up1_out(down3.dims[0], 64, 50, 1);
    up1->forward(down3, up1_out);
    concatenate(up1_out, down2);
    
    Tensor up2_out(up1_out.dims[0], 64, 100, 1);
    up2->forward(up1_out, up2_out);
    concatenate(up2_out, down1);
    
    // Final convolutions
    Tensor out_inter(up2_out.dims[0], 64, 100, 1);
    outIntermediate->forward(up2_out, out_inter);

    // Debug prints for intermediate values
    #ifdef DEBUG_OUTPUT
    std::vector<float> debug_inter(out_inter.dims[0] * out_inter.dims[1] * out_inter.dims[2]);
    CUDA_CHECK(cudaMemcpy(debug_inter.data(), out_inter.data, 
                         debug_inter.size() * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    std::cout << "Pre-final conv stats:\n";
    float max_inter = *std::max_element(debug_inter.begin(), debug_inter.end());
    float min_inter = *std::min_element(debug_inter.begin(), debug_inter.end());
    std::cout << "  Max: " << max_inter << ", Min: " << min_inter << "\n";
    #endif    

    outFinal->forward(out_inter, output);

     // Debug prints before activation
    #ifdef DEBUG_OUTPUT
    std::vector<float> pre_act(output.size / sizeof(float));
    CUDA_CHECK(cudaMemcpy(pre_act.data(), output.data, 
                         output.size, 
                         cudaMemcpyDeviceToHost));
    std::cout << "Pre-activation stats:\n";
    float max_pre = *std::max_element(pre_act.begin(), pre_act.end());
    float min_pre = *std::min_element(pre_act.begin(), pre_act.end());
    std::cout << "  Max: " << max_pre << ", Min: " << min_pre << "\n";
    #endif

    // Final activation and scaling
    softplus(output);
    scale(output, 0.001f);

    // Final debug prints
    #ifdef DEBUG_OUTPUT
    std::vector<float> final_out(output.size / sizeof(float));
    CUDA_CHECK(cudaMemcpy(final_out.data(), output.data, 
                         output.size, 
                         cudaMemcpyDeviceToHost));
    std::cout << "Final output stats:\n";
    float max_final = *std::max_element(final_out.begin(), final_out.end());
    float min_final = *std::min_element(final_out.begin(), final_out.end());
    std::cout << "  Max: " << max_final << ", Min: " << min_final << "\n";
    #endif
}

void PVFinderModel::loadWeights(const std::string& path) {
    try {
        cnpy::npz_t weights = cnpy::npz_load(path);
        
        auto check_weight = [&](const std::string& name) {
            if (weights.count(name) == 0) {
                throw std::runtime_error("Missing weight: " + name);
            }
        };

        // Load linear layers (these match the current format)
        auto load_linear = [&](std::unique_ptr<LinearLayer>& layer, 
                             const std::string& base_name) {
            std::string w_name = base_name + ".weight";
            std::string b_name = base_name + ".bias";
            
            check_weight(w_name);
            check_weight(b_name);
            
            auto& w_array = weights[w_name];
            auto& b_array = weights[b_name];
            layer->loadWeights(w_array.data<float>(), b_array.data<float>());
        };

        // Load conv/deconv layers with batch norm
        auto load_conv = [&](auto& layer, const std::string& base_name) {
            // Conv weights are in .0.weight and .0.bias
            std::string w_name = base_name + ".0.weight";
            std::string b_name = base_name + ".0.bias";
            
            check_weight(w_name);
            check_weight(b_name);
            
            auto& w_array = weights[w_name];
            auto& b_array = weights[b_name];
            layer->loadWeights(w_array.data<float>(), b_array.data<float>());
        };

        // Load final conv layers without batch norm
        auto load_final_conv = [&](auto& layer, const std::string& base_name) {
            std::string w_name = base_name + ".weight";
            std::string b_name = base_name + ".bias";
            
            check_weight(w_name);
            check_weight(b_name);
            
            auto& w_array = weights[w_name];
            auto& b_array = weights[b_name];
            layer->loadWeights(w_array.data<float>(), b_array.data<float>());
        };

        // Load linear layers
        load_linear(layer1, "layer1");
        load_linear(layer2, "layer2");
        load_linear(layer3, "layer3");
        load_linear(layer4, "layer4");
        load_linear(layer5, "layer5");
        load_linear(layer6A, "layer6A");

        // Load conv layers with batch norm
        load_conv(rcbn1, "rcbn1");
        load_conv(rcbn2, "rcbn2");
        load_conv(rcbn3, "rcbn3");

        // Load deconv layers
        // For up1, the first part is deconv (.0) and second part is conv with batch norm (.1)
        std::string w_name = "up1.0.weight";
        std::string b_name = "up1.0.bias";
        check_weight(w_name);
        check_weight(b_name);
        up1->loadWeights(weights[w_name].data<float>(), weights[b_name].data<float>());

        w_name = "up2.0.weight";
        b_name = "up2.0.bias";
        check_weight(w_name);
        check_weight(b_name);
        up2->loadWeights(weights[w_name].data<float>(), weights[b_name].data<float>());

        // Load final layers
        load_final_conv(outIntermediate, "out_intermediate");
        load_final_conv(outFinal, "outc");

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load weights: " + std::string(e.what()));
    }
}
} // namespace pvfinder