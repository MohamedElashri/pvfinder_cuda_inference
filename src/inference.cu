#include "inference.hpp"
#include "pvs.hpp"
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

// Define a global debug key
#define DEBUG

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
    
    // Parameters for PV finding (matching Python implementation)
    float bin_threshold = 0.01f;
    float integral_threshold = 0.50f;
    int min_width = 2;
    
    // Validate each batch
    for (int batch = 0; batch < output.dims[0]; batch++) {
        float sum = 0.0f;
        float min_val = std::numeric_limits<float>::max();
        float max_val = -std::numeric_limits<float>::max();
        
        // Get the start index for this event's bins
        int event_start = batch * output.dims[2];
        
        // Calculate statistics for this batch
        for (int i = 0; i < output.dims[2]; i++) {
            float val = host_output[event_start + i];
            sum += val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        // Print detailed statistics for first few batches
        if (batch < 3) {
            std::cout << "\n==================================\n";
            std::cout << "Event " << (eventStartIdx + batch) << " analysis:\n";
            std::cout << "==================================\n";
            
            // 1. Basic KDE Statistics
            std::cout << "\n1. KDE Statistics:\n";
            std::cout << "   Min value: " << min_val << "\n";
            std::cout << "   Max value: " << max_val << "\n";
            std::cout << "   Sum: " << sum << "\n";
            std::cout << "   Mean: " << sum/output.dims[2] << "\n";
            
            // 2. Value Distribution Analysis
            int num_high = 0;
            int num_low = 0;
            int num_zero = 0;
            int num_neg = 0;
            float thresh_high = 1.0;
            float thresh_low = 1.0;
            
            for (int i = 0; i < output.dims[2]; i++) {
                float val = host_output[event_start + i];
                if (val > thresh_high) num_high++;
                if (val < thresh_low && val > 0) num_low++;
                if (val == 0) num_zero++;
                if (val < 0) num_neg++;
            }
            
            std::cout << "\n2. Value Distribution:\n";
            std::cout << "   Values > " << thresh_high << ": " << num_high << " bins\n";
            std::cout << "   Values < " << thresh_low << ": " << num_low << " bins\n";
            std::cout << "   Zero values: " << num_zero << " bins\n";
            std::cout << "   Negative values: " << num_neg << " bins\n";
            
            // 3. Sample Values
            std::cout << "\n3. First 10 bin values:\n";
            for (int i = 0; i < std::min(10, output.dims[2]); i++) {
                float val = host_output[event_start + i];
                float pos = 0.1f * i - 100.0f;  // Convert bin to mm
                std::cout << "   Bin " << std::setw(2) << i 
                         << " (" << std::setw(6) << std::fixed << std::setprecision(1) << pos 
                         << " mm): " << std::scientific << std::setprecision(4) << val << "\n";
            }

            // Warning for unusually high values
            if (max_val > 10.0) {
                std::cout << "\n*** WARNING: Unusually high values detected! ***\n";
            }

            // 4. PV Finding
            std::vector<float> event_output(host_output.begin() + event_start,
                                          host_output.begin() + event_start + output.dims[2]);
            
            auto pvs_updated = pv_locations_updated_res(
                event_output, bin_threshold, integral_threshold, min_width);
            auto pvs_standard = pv_locations_res(
                event_output, bin_threshold, integral_threshold, min_width);
            
            // Get resolution
            auto predicted_reso = get_reco_resolution(
                pvs_updated, event_output, 1.0f, 1, 0.1f, false);
            
            std::cout << "\n4. PV Finding Results:\n";
            
            // Updated method results
            std::cout << "   4.1 Updated Method (" << pvs_updated.size() << " PVs):\n";
            for (size_t i = 0; i < pvs_updated.size(); i++) {
                float pos_mm = 0.1f * pvs_updated[i] - 100.0f;
                float res_mm = 0.1f * predicted_reso[i];
                std::cout << "      PV at " << std::fixed << std::setprecision(2) 
                         << pos_mm << " ± " << res_mm << " mm\n";
                
                // Show local KDE values around this PV
                int bin = static_cast<int>(pvs_updated[i]);
                std::cout << "      Local KDE values:\n";
                for (int j = std::max(0, bin-2); j <= std::min(bin+2, output.dims[2]-1); j++) {
                    float local_pos = 0.1f * j - 100.0f;
                    std::cout << "        " << local_pos << " mm: " 
                             << event_output[j] << "\n";
                }
            }

            // Standard method results
            std::cout << "\n   4.2 Standard Method (" << pvs_standard.size() << " PVs):\n";
            for (const auto& pv : pvs_standard) {
                float pos_mm = 0.1f * pv - 100.0f;
                std::cout << "      PV at " << std::fixed << std::setprecision(2) 
                         << pos_mm << " mm\n";
            }

            // 5. Quality Checks
            std::cout << "\n5. Quality Checks:\n";
            std::cout << "   Threshold used: " << bin_threshold << "\n";
            std::cout << "   Integral threshold: " << integral_threshold << "\n";
            std::cout << "   Minimum width: " << min_width << " bins\n";
            
            int peaks_above_threshold = 0;
            float max_peak = 0.0f;
            for (const auto& val : event_output) {
                if (val >= bin_threshold) {
                    peaks_above_threshold++;
                    max_peak = std::max(max_peak, val);
                }
            }
            std::cout << "   Bins above threshold: " << peaks_above_threshold << "\n";
            std::cout << "   Maximum peak value: " << max_peak << "\n";
            
            std::cout << "\n==================================\n\n";
        }
    }
}

void Inference::saveOutputs(const std::vector<float>& output_data, 
                          int eventStartIdx, 
                          int batchSize,
                          std::ofstream& outFile) {
    // Parameters for PV finding
    float bin_threshold = 0.01f;
    float integral_threshold = 0.50f;
    int min_width = 2;
    
    for (int batch = 0; batch < batchSize; batch++) {
        int eventId = eventStartIdx + batch;
        outFile << eventId;  // Write event ID
        
        // Write all 100 bin values for this event
        for (int i = 0; i < 100; i++) {
            outFile << "," << output_data[batch * 100 + i];
        }

        // Extract event data for PV finding
        std::vector<float> event_output(output_data.begin() + batch * 100,
                                      output_data.begin() + (batch + 1) * 100);
        
        // Find PVs and get resolution
        auto pvs = pv_locations_updated_res(event_output, bin_threshold, 
                                          integral_threshold, min_width);
        auto resolution = get_reco_resolution(pvs, event_output, 1.0f, 1, 0.1f, false);

        // Add PV information to output
        outFile << " # PVs:";
        for (size_t i = 0; i < pvs.size(); i++) {
            float pos_mm = 0.1f * pvs[i] - 100.0f;
            float res_mm = 0.1f * resolution[i];
            outFile << " " << pos_mm << "±" << res_mm;
        }
        
        outFile << "\n";
    }
}

int getEventFromInterval(int intervalNumber) {
    return intervalNumber / 40;
}

int getLocalInterval(int intervalNumber) {
    return intervalNumber % 40;
}


void Inference::run(const std::string& output_filename) {
    try {
        // Load validation data
        cnpy::NpyArray data = cnpy::npy_load(data_path);
        const float* validation_data = data.data<float>();
        
        std::cout << "Data shape: [";
        for(size_t i = 0; i < data.shape.size(); i++) {
            std::cout << data.shape[i] << (i < data.shape.size()-1 ? ", " : "");
        }
        std::cout << "]\n";

        // Data shape should be [N_intervals, 9, 250]
        if (data.shape.size() != 3 || data.shape[1] != 9) {
            throw std::runtime_error("Invalid data shape. Expected [N_intervals, 9, 250]");
        }

        int total_intervals = data.shape[0];
        int n_features = data.shape[1];    // should be 9
        int n_tracks = data.shape[2];      // should be 250
        int intervals_per_event = 40;       // Fixed number from Python code
        int total_events = total_intervals / intervals_per_event;

        std::cout << "Total intervals: " << total_intervals << "\n";
        std::cout << "Total events: " << total_events << "\n";
        std::cout << "Intervals per event: " << intervals_per_event << "\n";
        std::cout << "Features per track: " << n_features << "\n";
        std::cout << "Tracks per interval: " << n_tracks << "\n";

        // Open output file
        std::ofstream outFile(output_filename);
        if (!outFile) {
            throw std::runtime_error("Failed to open output file: " + output_filename);
        }

        // Write header
        outFile << "EventID,IntervalID";
        for (int i = 0; i < 100; i++) {
            outFile << ",bin" << i;
        }
        outFile << "\n";

        // Process in batches of intervals
        int batch_size = 64;  // batch size in terms of intervals
        std::vector<float> host_output;

        for (int interval_start = 0; interval_start < total_intervals; interval_start += batch_size) {
            int current_batch_size = std::min(batch_size, total_intervals - interval_start);
            
            // Create input tensor [batch_size, n_features, n_tracks]
            Tensor input(current_batch_size, n_features, n_tracks, 1);

            // Copy input data for this batch of intervals
            size_t input_size = current_batch_size * n_features * n_tracks;
            size_t input_offset = interval_start * n_features * n_tracks;
            
            CUDA_CHECK(cudaMemcpy(input.data,
                                validation_data + input_offset,
                                input_size * sizeof(float),
                                cudaMemcpyHostToDevice));
            
            // Create output tensor for the intervals
            Tensor output(current_batch_size, 1, 100, 1);
            
            // Forward pass
            model.forward(input, output);
            
            // Validate output
            validateOutput(output, interval_start);
            
            // Copy output to host
            int output_size = current_batch_size * 100;  // 100 bins per interval
            host_output.resize(output_size);
            CUDA_CHECK(cudaMemcpy(host_output.data(), output.data,
                                output_size * sizeof(float),
                                cudaMemcpyDeviceToHost));
            
            // Save intervals
            for (int i = 0; i < current_batch_size; i++) {
                int global_interval = interval_start + i;
                int event_id = getEventFromInterval(global_interval);
                int interval_id = getLocalInterval(global_interval);

                outFile << event_id << "," << interval_id;
                
                // Write KDE values for this interval
                for (int bin = 0; bin < 100; bin++) {
                    outFile << "," << host_output[i * 100 + bin];
                }
                outFile << "\n";
            }

            if (interval_start % 400 == 0) {  // Print every 10 events worth of intervals
                std::cout << "Processed intervals up to " << interval_start << "/" 
                         << total_intervals 
                         << " (Event " << getEventFromInterval(interval_start) << ")"
                         << std::endl;
            }
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
