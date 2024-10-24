#include <cuda_runtime.h>
#include "/usr/include/cudnn.h"
#include "model.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <weights_path> <validation_data_path>" << std::endl;
        return 1;
    }

    try {
        // Initialize CUDA
        int deviceId = 1;
        CUDA_CHECK(cudaSetDevice(deviceId));
        
        // Create cuDNN handle
        CUDNN_CHECK(cudnnCreate(&pvfinder::cudnnHandle));

        // Initialize model and load weights
        pvfinder::PVFinderModel model;
        model.loadWeights(argv[1]);

        // Load validation data
        cnpy::NpyArray data = cnpy::npy_load(argv[2]);
        const float* validation_data = data.data<float>();
        
        // Process data in batches
        int batch_size = 64;
        int total_samples = data.shape[0];
        
        for (int batch_start = 0; batch_start < total_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, total_samples - batch_start);
            
            // Create input tensor
            pvfinder::Tensor input(current_batch_size, 9, 1, 1);
            CUDA_CHECK(cudaMemcpy(input.data, 
                                validation_data + batch_start * 9,
                                current_batch_size * 9 * sizeof(float),
                                cudaMemcpyHostToDevice));
            
            // Create output tensor
            pvfinder::Tensor output(current_batch_size, 1, 100, 1);
            
            // Forward pass
            model.forward(input, output);
            
            // Process output as needed
            // Lets just print the first output
            float* output_data = new float[output.size / sizeof(float)];
            CUDA_CHECK(cudaMemcpy(output_data, output.data, output.size, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 100; ++i) {
                std::cout << output_data[i] << " ";
            }
            std::cout << std::endl;
            delete[] output_data;

        }

        // Cleanup
        CUDNN_CHECK(cudnnDestroy(pvfinder::cudnnHandle));
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}