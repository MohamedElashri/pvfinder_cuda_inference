#include "inference.hpp"
#include <cuda_runtime.h>
#include "/usr/include/cudnn.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstring>
#include "model.hpp"
#include <cnpy.h>
#include <limits>


namespace pvfinder {


Inference::Inference(const std::string& weights_path, const std::string& data_path)
    : data_path(data_path) {
    try {
        // Initialize CUDA
        int deviceId = 0;
        CUDA_CHECK(cudaSetDevice(deviceId));
        
        // Create cuDNN handle
        CUDNN_CHECK(cudnnCreate(&cudnnHandle));

        // Load model weights
        model.loadWeights(weights_path);
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Initialization error: " + std::string(e.what()));
    }
}

void Inference::validateOutput(const Tensor& output, int eventStartIdx) {
    // Allocate host memory for output
    int total_size = output.dims[0] * output.dims[1] * output.dims[2] * output.dims[3];
    std::vector<float> host_output(total_size);
    
    // Copy output to host
    CUDA_CHECK(cudaMemcpy(host_output.data(), output.data, 
                         total_size * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    // Validate each batch
    for (int batch = 0; batch < output.dims[0]; batch++) {
        float sum = 0.0f;
        float min_val = std::numeric_limits<float>::max();
        float max_val = -std::numeric_limits<float>::max();
        
        // Calculate statistics for this batch
        for (int i = 0; i < output.dims[2]; i++) {
            float val = host_output[batch * output.dims[2] + i];
            sum += val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        // Print statistics for first few batches
        if (batch < 3) {
            std::cout << "Event " << (eventStartIdx + batch) << " statistics:\n"
                     << "  Sum: " << sum << "\n"
                     << "  Min: " << min_val << "\n"
                     << "  Max: " << max_val << "\n"
                     << "  Mean: " << sum/output.dims[2] << std::endl;
        }
    }
}

void Inference::saveOutputs(const std::vector<float>& output_data, 
                          int eventStartIdx, 
                          int batchSize,
                          std::ofstream& outFile) {
    // Each event has 100 bins
    for (int batch = 0; batch < batchSize; batch++) {
        int eventId = eventStartIdx + batch;
        outFile << eventId;  // Write event ID
        
        // Write all 100 bin values for this event
        for (int i = 0; i < 100; i++) {
            outFile << "," << output_data[batch * 100 + i];
        }
        outFile << "\n";
    }
}

void Inference::run(const std::string& output_filename) {
    try {
        // Create output directory if needed
        size_t last_slash = output_filename.find_last_of('/');
        if (last_slash != std::string::npos) {
            std::string dir = output_filename.substr(0, last_slash);
            std::string mkdir_cmd = "mkdir -p " + dir;
            system(mkdir_cmd.c_str());
        }

        // Load validation data
        cnpy::NpyArray data = cnpy::npy_load(data_path);
        const float* validation_data = data.data<float>();
        
        // Open output file
        std::ofstream outFile(output_filename);
        if (!outFile) {
            throw std::runtime_error("Failed to open output file: " + output_filename);
        }

        // Write header
        outFile << "EventID";
        for (int i = 0; i < 100; i++) {
            outFile << ",bin" << i;
        }
        outFile << "\n";
        
        // Process data in batches
        int batch_size = 64;
        int total_samples = data.shape[0];
        std::vector<float> host_output;  // For storing output data

        for (int batch_start = 0; batch_start < total_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, total_samples - batch_start);
            
            // Create input tensor
            Tensor input(current_batch_size, 9, 1, 1);
            CUDA_CHECK(cudaMemcpy(input.data, 
                                validation_data + batch_start * 9,
                                current_batch_size * 9 * sizeof(float),
                                cudaMemcpyHostToDevice));
            
            // Create output tensor
            Tensor output(current_batch_size, 1, 100, 1);
            
            // Forward pass
            model.forward(input, output);
            
            // Validate output
            validateOutput(output, batch_start);
            
            // Copy output to host
            int output_size = current_batch_size * 100;  // 100 bins per event
            host_output.resize(output_size);
            CUDA_CHECK(cudaMemcpy(host_output.data(), output.data,
                                output_size * sizeof(float),
                                cudaMemcpyDeviceToHost));
            
            // Save to file
            saveOutputs(host_output, batch_start, current_batch_size, outFile);
        }

        outFile.close();
        std::cout << "Output saved to: " << output_filename << std::endl;
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error during inference: " + std::string(e.what()));
    }
}

} // namespace pvfinder


void printHelp(const char* programName) {
    std::cout << "Usage: " << programName << " [options] -w WEIGHTS -d DATA [-o OUTPUT]\n"
              << "\nOptions:\n"
              << "  -w, --weights FILE    Model weights file (.npz)\n"
              << "  -d, --data FILE       Validation data file (.npy)\n"
              << "  -o, --out FILE        Output file [default: ./output.csv]\n"
              << "  -h, --help            Show this help message\n"
              << "\nExamples:\n"
              << "  " << programName << " -w model.npz -d val.npy -o results.csv\n"
              << "  " << programName << " --weights model.npz --data val.npy\n"
              << std::endl;
}


// Main function
int main(int argc, char** argv) {
    if (argc == 1 || (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))) {
        printHelp(argv[0]);
        return 0;
    }

    std::string weights_path;
    std::string data_path;
    std::string output_path = "./output.csv";  // default output path

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-w" || arg == "--weights") {
            if (i + 1 < argc) {
                weights_path = argv[++i];
            } else {
                std::cerr << "Error: -w/--weights requires a file path" << std::endl;
                return 1;
            }
        }
        else if (arg == "-d" || arg == "--data") {
            if (i + 1 < argc) {
                data_path = argv[++i];
            } else {
                std::cerr << "Error: -d/--data requires a file path" << std::endl;
                return 1;
            }
        }
        else if (arg == "-o" || arg == "--out") {
            if (i + 1 < argc) {
                output_path = argv[++i];
            } else {
                std::cerr << "Error: -o/--out requires a file path" << std::endl;
                return 1;
            }
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n"
                     << "Use --help for usage information." << std::endl;
            return 1;
        }
    }

    // Check required arguments
    if (weights_path.empty() || data_path.empty()) {
        std::cerr << "Error: weights (-w) and data (-d) paths are required\n"
                  << "Use --help for usage information." << std::endl;
        return 1;
    }

    try {
        pvfinder::Inference inference(weights_path, data_path);
        inference.run(output_path);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
