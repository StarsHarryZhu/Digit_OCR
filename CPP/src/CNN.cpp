#include <string>
#include <random>
#include <cmath>

#include "OCR.hpp"

namespace Digit_OCR{
void digit_OCR::CNN_init(){
    // init the normal distribution for neural cells
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.01);


    CNN.conv_filters.resize(CNN.filter_count, 
        std::vector<std::vector<double>>(CNN.filter_size, 
            std::vector<double>(CNN.filter_size)
        )
    );

    CNN.conv_b.resize(CNN.filter_count, 0.0);

    for (int f = 0; f < CNN.filter_count; f++) {
        for (int i = 0; i < CNN.filter_size; i++) {
            for (int j = 0; j < CNN.filter_size; j++) {
                CNN.conv_filters[f][i][j] = dist(gen);
            }
        }
    }
}

int digit_OCR::CNN_OCR(std::string path){
    auto img = image_loader_2D(path);
    auto filted = CNN_conv_forward(img);
    auto pooled = CNN_maxpool(filted);
}

std::vector<std::vector<std::vector<double>>> digit_OCR::CNN_conv_forward(const std::vector<std::vector<double>>& input){
    const int filted_row = row_default - CNN.filter_size + 1;
    const int filted_col = col_default - CNN.filter_size + 1;

    std::vector<std::vector<std::vector<double>>> filted(
        CNN.filter_count, std::vector<std::vector<double>>(
            filted_row, std::vector<double>(filted_col, 0)
        )
    );

    for(int f = 0; f < CNN.filter_count; f++){
        for(int r = 0; r < filted_row; r++){
            for(int c = 0; c < filted_col; c++){
                double sum = 0;

                for(int i = 0; i < CNN.filter_size; i++)
                    for(int j = 0; j < CNN.filter_size; j++)
                        sum += input[r + i][c + j] * CNN.conv_filters[f][i][j];
                
                filted[f][r][c] = sum + CNN.conv_b[f];
            }
        }
    }

    for(auto& filted_ones : filted)
        for(auto& r : filted_ones)
            for(auto& item : r)
                ReLU(item);
    return filted;
}

std::vector<std::vector<std::vector<double>>> digit_OCR::CNN_maxpool(const std::vector<std::vector<std::vector<double>>>& filted){
    if(filted[0].size() % CNN.pool_ratio != 0 || filted[0][0].size() % CNN.pool_ratio != 0)
        throw new std::runtime_error("pool ratio should able to devide the row and col of filted matrix");
    const int pooled_row = filted[0].size() / CNN.pool_ratio;
    const int pooled_col = filted[0][0].size() / CNN.pool_ratio;

    std::vector<std::vector<std::vector<double>>> pooled(
        CNN.filter_count, std::vector<std::vector<double>>(
            pooled_row, std::vector<double>(pooled_col, 0)
        )
    );

    for(int f = 0; f < CNN.filter_count; f++){
        for(int r = 0; r < pooled_row; r++){
            for(int c = 0; c < pooled_col; c++){
                double maximum = filted[f][r * CNN.pool_ratio][c * CNN.pool_ratio];

                for(int i = 0; i < CNN.pool_ratio; i++)
                    for(int j = 0; j < CNN.pool_ratio; j++)
                        maximum = std::max(maximum, filted[f][r * CNN.pool_ratio + i][c * CNN.pool_ratio + j]);
                
                pooled[f][r][c] = maximum;
            }
        }
    }

    return pooled;
}
}