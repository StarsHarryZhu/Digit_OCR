#pragma once

#include <string>
#include <vector>
#include <fstream>


namespace Digit_OCR::MNIST{
    struct MNIST_data_1D{
        std::vector<size_t> label;
        std::vector<std::vector<double>> image;

        MNIST_data_1D(int count, int row, int col){
            label = std::vector<size_t>(count);
            image = std::vector<std::vector<double>>(count, std::vector<double>(row * col));
        }
    };
    struct MNIST_data_2D{
        std::vector<size_t> label;
        std::vector<std::vector<std::vector<double>>> image;

        MNIST_data_2D(int count, int row, int col){
            label = std::vector<size_t>(count);
            image = std::vector<std::vector<std::vector<double>>>(count, std::vector<std::vector<double>>(row, std::vector<double>(col)));
        }
    };

    MNIST_data_1D decoder_1D(const std::string label_path, const std::string image_path);
    MNIST_data_2D decoder_2D(const std::string label_path, const std::string image_path);

    inline int decode_label(int offset);
    inline std::vector<double> decode_image(int offset);

    int read_big_endian(std::ifstream& file);
}