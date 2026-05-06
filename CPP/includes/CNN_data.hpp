#pragma once

#include <vector>
#include <string>

namespace Digit_OCR{
struct CNN_data{
    const int code;
    const std::string file;

    const int img_row;
    const int img_col;

    const int filter_count;
    const int filter_size;

    const int pool_ratio;

    const double learning_rate;

    std::vector<std::vector<std::vector<double>>> conv_filters;
    std::vector<double> conv_b;

    std::vector<std::vector<double>> flatten_w;
    std::vector<double> flatten_b;

    CNN_data(std::string file, double learning_rate, int row, int col, int f_count, int f_size, int pool_ratio):
        code(514),
        file(file),
        learning_rate(learning_rate),
        img_row(row),
        img_col(col),
        filter_count(f_count),
        filter_size(f_size),
        pool_ratio(pool_ratio)
        {}
};
}