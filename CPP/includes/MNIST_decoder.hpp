#pragma once

#include <string>
#include <vector>
#include <fstream>

#include "data_structure.hpp"


namespace Digit_OCR::MNIST{
    std::vector<ST_image_1D> decoder_1D(const std::string label_path, const std::string image_path);
    std::vector<ST_image_2D> decoder_2D(const std::string label_path, const std::string image_path);

    inline int decode_1label(int offset);
    inline std::vector<double> decode_1image_1D(int offset);
    inline std::vector<std::vector<double>> decode_1image_2D(int offset);

    int read_big_endian(std::ifstream& file);
}