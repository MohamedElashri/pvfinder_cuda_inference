#pragma once
#include "common.hpp"
#include <string>

namespace pvfinder {

class DataLoader {
public:
    DataLoader(const std::string& filepath, int batch_size);
    ~DataLoader();
    
    bool hasNext() const;
    Tensor nextBatch();
    void reset();

private:
    int batch_size;
    int current_pos;
    int total_samples;
    size_t data_size;
    float* gpu_data{nullptr};
};

} // namespace pvfinder