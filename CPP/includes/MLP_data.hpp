#pragma once

#include <vector>
#include <string>

namespace Digit_OCR{

struct MLP_data{
    const int code;
    const std::string file;

    const int input_size;
    const int hidden_size;
    std::vector<std::vector<double>> w1;
    std::vector<double> b1;
    std::vector<std::vector<double>> w2;
    std::vector<double> b2;

    MLP_data(std::string file, int input_size, int hidden_size):
        code(114), 
        file(file), 
        input_size(input_size), 
        hidden_size(hidden_size){}
};

}