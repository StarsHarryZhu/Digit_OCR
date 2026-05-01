#include <string>
#include <iostream>
#include <random>

#include "OCR.hpp"

namespace Digit_OCR{
void digit_OCR::MLP_init(){
    // init the normal distribution for neural cells
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.01);

    // init the neural network
    MLP_w1.resize(MLP_input_size, std::vector<double>(MLP_hidden_size));
    MLP_b1.resize(MLP_hidden_size, 0.0);
    MLP_w2.resize(MLP_hidden_size, std::vector<double>(output));
    MLP_b2.resize(output, 0.0);
    // w1 and w2 need normal distribution
    for(int i = 0; i < MLP_input_size; i++)
        for(int j = 0; j < MLP_hidden_size; j++)
            MLP_w1[i][j] = dist(gen);
    
    for(int i = 0; i < MLP_hidden_size; i++)
        for(int j = 0; j < output; j++)
            MLP_w2[i][j] = dist(gen);
}

std::vector<double> digit_OCR::MLP_forward(const std::vector<double>& x){
    // layer 1: input -> hidden
    // origin val
    std::vector<double> z1(MLP_hidden_size);
    // ReLU val
    std::vector<double> a1(MLP_hidden_size);

    // matrix multiply [1 * i][i * h] = [1 * h]
    double sum;
    for(int h = 0; h < MLP_hidden_size; h++){
        sum = MLP_b1[h];
        for(int i = 0; i < MLP_input_size; i++)
            sum += MLP_w1[i][h] * x[i];

        z1[h] = sum;
        a1[h] = ReLU(sum);
    }

    // layer 2: hidden -> output

    // origin val, no need for ReLU
    std::vector<double> z2(output);

    //matrix multiply [1 * h][h * o] = [1 * o]
    for(int o = 0; o < output; o++){
        sum = MLP_b2[o];
        for(int h = 0; h < MLP_hidden_size; h++)
            sum += MLP_w2[h][o] * a1[h];

        z2[o] = sum;
    }

    // normalisze
    softmax(z2);

    return z2;
}
}