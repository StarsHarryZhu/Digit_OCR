#include <string>
#include <iostream>
#include <random>
#include <cmath>

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

std::vector<std::vector<double>> digit_OCR::MLP_forward(const std::vector<double>& x, std::vector<double>& p){
    // layer 1: input -> hidden
    // origin val
    std::vector<double> z1(MLP_hidden_size);
    // ReLU val
    std::vector<double> a1(MLP_hidden_size);

    // matrix multiply [1 * i][i * h] = [1 * h]
    for(int h = 0; h < MLP_hidden_size; h++){
        double sum = MLP_b1[h];
        for(int i = 0; i < MLP_input_size; i++)
            sum += MLP_w1[i][h] * x[i];

        z1[h] = sum;
        a1[h] = ReLU(sum);
    }

    // layer 2: hidden -> output
    // origin val, no need for ReLU
    std::vector<double> z2(output);
    p.resize(output);

    //matrix multiply [1 * h][h * o] = [1 * o]
    for(int o = 0; o < output; o++){
        double sum = MLP_b2[o];
        for(int h = 0; h < MLP_hidden_size; h++)
            sum += MLP_w2[h][o] * a1[h];

        z2[o] = sum;
        p[o] = sum;
    }

    // store the origin data
    std::vector<std::vector<double>> z;
    z.push_back(z1);
    z.push_back(z2);
    z.push_back(a1);
    z.push_back(p);

    // normalisze
    softmax(p);
    return z;
}

void digit_OCR::MLP_train_once(const int& label, const std::vector<double>& image){
    // forward/probability and z1&z2
    std::vector<double> p;
    auto z = MLP_forward(image, p);

    // target
    auto target_p = target_generator(label);

    // predict error(derivative z2)
    std::vector<double> dz2(output);
    for(int i = 0; i < output; i++)
        dz2[i] = p[i] - target_p[i];

    // hidden error(derivative z1)
    std::vector<double> dz1(MLP_hidden_size);
    for(int h = 0; h < MLP_hidden_size; h++){
        double sum = 0;
        for (int o = 0; o < output; o++) {
            sum += dz2[o] * MLP_w2[h][o];
        }

        dz1[h] = sum * ReLU_derivative(z[0][h]);
    }
    
    // update w2 and b2 
    // *** must earlier than w1 and b1 ***
    for(int h = 0; h < MLP_hidden_size; h++)
        for(int o = 0; o < output; o++)
            MLP_w2[h][o] -= learning_rate * z[2][h] * dz2[o];

    for (int h = 0; h < MLP_hidden_size; h++)
        MLP_b1[h] -= learning_rate * dz1[h];

    // update w1 and b1
    for(int i = 0; i < MLP_input_size; i++)
        for(int h = 0; h < MLP_hidden_size; h++)
            MLP_w1[i][h] -= learning_rate * image[i] * dz1[h];

    for (int o = 0; o < output; o++)
        MLP_b2[o] -= learning_rate * dz2[o];
}

void digit_OCR::MLP_train(const MNIST::data_1D& train, const MNIST::data_1D& test, int epochs){
    std::cout << "start training, target epochs: " << epochs*10 << std::endl;
    for(int i = 0; i < epochs*10; i++){
        for(int j = 0; j < train.label.size(); j++)
            MLP_train_once(train.label[j], train.image[j]);
        if((i+1) % (epochs) == 0){
            auto test_result = MLP_test(test);
            std::printf("Epoch: %d\nAccuracy: %.1f%%\nLoss: %.3f\n\n", i, test_result[0], test_result[1]);
        }
    }
}


std::vector<double> digit_OCR::MLP_test(const MNIST::data_1D& test){
    std::vector<double> ans(2, 0);
    double accurate = 0;
    double loss_sum = 0;
    for(int i = 0; i < test.label.size(); i++){
        // probility
        std::vector<double> p;
        MLP_forward(test.image[i], p);

        // accuracy
        size_t predict_num = predict(p);
        if(predict_num == test.label[i])
            accurate++;

        // loss, avoiding log(0)
        loss_sum += -std::log(p[test.label[i]] + (1e-15));
    }
    ans[0] = accurate * 100.0 / test.label.size();
    ans[1] = loss_sum / test.label.size();
    return ans;
}
}