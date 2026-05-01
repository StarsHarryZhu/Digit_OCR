#pragma once

#include <string>
#include <vector>
#include <fstream>


namespace Digit_OCR::MNIST{
    struct data_1D{
        std::vector<size_t> label;
        std::vector<std::vector<double>> image;

        data_1D(int count, int row, int col){
            label = std::vector<size_t>(count);
            image = std::vector<std::vector<double>>(count, std::vector<double>(row * col));
        }
    };
    struct data_2D{
        std::vector<size_t> label;
        std::vector<std::vector<std::vector<double>>> image;

        data_2D(int count, int row, int col){
            label = std::vector<size_t>(count);
            image = std::vector<std::vector<std::vector<double>>>(count, std::vector<std::vector<double>>(row, std::vector<double>(col)));
        }
    };

    data_1D decoder_1D(const std::string label_path, const std::string image_path);
    data_2D decoder_2D(const std::string label_path, const std::string image_path);

    int read_big_endian(std::ifstream& file);
}