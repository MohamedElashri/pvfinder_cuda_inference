#pragma once
#include <fstream>
#include <string>
#include <vector>
#include "common.hpp"

namespace pvfinder {

class NpyLoader {
public:
    static bool loadFloat32(const std::string& filename, std::vector<float>& data, std::vector<size_t>& shape) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // Check magic string
        char magic[6];
        file.read(magic, 6);
        if (std::string(magic, 6) != "\x93NUMPY") {
            throw std::runtime_error("Invalid NPY file format");
        }

        // Read version
        uint8_t major_version, minor_version;
        file.read(reinterpret_cast<char*>(&major_version), 1);
        file.read(reinterpret_cast<char*>(&minor_version), 1);

        // Read header length
        uint16_t header_len;
        if (major_version == 1) {
            file.read(reinterpret_cast<char*>(&header_len), 2);
        } else if (major_version == 2) {
            uint32_t header_len_32;
            file.read(reinterpret_cast<char*>(&header_len_32), 4);
            header_len = header_len_32;
        } else {
            throw std::runtime_error("Unsupported NPY version");
        }

        // Read header
        std::string header(header_len, ' ');
        file.read(&header[0], header_len);

        // Parse shape from header
        shape.clear();
        size_t pos = header.find("'shape': (");
        if (pos == std::string::npos) {
            throw std::runtime_error("Cannot find shape in header");
        }
        pos += 9;
        size_t end_pos = header.find(")", pos);
        std::string shape_str = header.substr(pos, end_pos - pos);
        
        // Parse comma-separated shape values
        size_t start = 0;
        size_t end = shape_str.find(",");
        while (end != std::string::npos) {
            shape.push_back(std::stoul(shape_str.substr(start, end - start)));
            start = end + 1;
            while (start < shape_str.length() && (shape_str[start] == ' ' || shape_str[start] == ',')) {
                start++;
            }
            end = shape_str.find(",", start);
        }
        if (start < shape_str.length()) {
            shape.push_back(std::stoul(shape_str.substr(start)));
        }

        // Calculate total size and read data
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        data.resize(total_size);
        file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));

        return true;
    }
};

} // namespace pvfinder