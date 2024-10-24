#pragma once
#include "model.hpp"
#include <string>
#include <vector>
#include <fstream>

namespace pvfinder {

class Inference {
public:
    Inference(const std::string& weights_path, const std::string& data_path);
    void run(const std::string& output_filename = "./output.csv");

private:
    void validateOutput(const Tensor& output, int eventStartIdx);
    void saveOutputs(const std::vector<float>& output_data, 
                    int eventStartIdx, 
                    int batchSize,
                    std::ofstream& outFile);

    PVFinderModel model;
    std::string data_path;
};

void printHelp(const char* programName);

} // namespace pvfinder
