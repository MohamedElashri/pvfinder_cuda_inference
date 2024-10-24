#include "data_loader.hpp"
#include "npy_loader.hpp"

namespace pvfinder {

DataLoader::DataLoader(const std::string& filepath, int bs) 
    : batch_size(bs), current_pos(0) {
    std::vector<float> data;
    std::vector<size_t> shape;
    
    if (!NpyLoader::loadFloat32(filepath, data, shape)) {
        throw std::runtime_error("Failed to load NPY file");
    }
    
    // Store the data
    total_samples = shape[0];  // First dimension is number of samples
    data_size = data.size() / total_samples;
    
    // Allocate GPU memory and copy data
    CUDA_CHECK(cudaMalloc(&gpu_data, data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu_data, data.data(), data.size() * sizeof(float), 
                         cudaMemcpyHostToDevice));
}

DataLoader::~DataLoader() {
    if (gpu_data) {
        cudaFree(gpu_data);
    }
}

bool DataLoader::hasNext() const {
    return current_pos < total_samples;
}

Tensor DataLoader::nextBatch() {
    int actual_batch_size = std::min(batch_size, total_samples - current_pos);
    Tensor batch(actual_batch_size, 9, 250, 1);  // Adjust dimensions based on your data
    
    CUDA_CHECK(cudaMemcpy(batch.data,
                         gpu_data + current_pos * data_size,
                         actual_batch_size * data_size * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    current_pos += actual_batch_size;
    return batch;
}

void DataLoader::reset() {
    current_pos = 0;
}

} // namespace pvfinder