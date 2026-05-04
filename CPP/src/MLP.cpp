#include <string>
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
    MLP.w1.resize(MLP.input_size, std::vector<double>(MLP.hidden_size));
    MLP.b1.resize(MLP.hidden_size, 0.0);
    MLP.w2.resize(MLP.hidden_size, std::vector<double>(output));
    MLP.b2.resize(output, 0.0);
    // w1 and w2 need normal distribution
    for(int i = 0; i < MLP.input_size; i++)
        for(int j = 0; j < MLP.hidden_size; j++)
            MLP.w1[i][j] = dist(gen);
    
    for(int i = 0; i < MLP.hidden_size; i++)
        for(int j = 0; j < output; j++)
            MLP.w2[i][j] = dist(gen);
}

std::vector<double> digit_OCR::MLP_forward(const std::vector<double>& input, std::vector<std::vector<double>>* config){
    // layer 1: input -> hidden
    // origin val
    std::vector<double> z1(MLP.hidden_size);
    // ReLU val
    std::vector<double> a1(MLP.hidden_size);

    // matrix multiply [1 * i][i * h] = [1 * h]
    for(int h = 0; h < MLP.hidden_size; h++){
        double sum = MLP.b1[h];
        for(int i = 0; i < MLP.input_size; i++)
            sum += MLP.w1[i][h] * input[i];

        z1[h] = sum;
        a1[h] = ReLU(sum);
    }

    // layer 2: hidden -> output
    // origin val, no need for ReLU
    std::vector<double> z2(output);

    //matrix multiply [1 * h][h * o] = [1 * o]
    for(int o = 0; o < output; o++){
        double sum = MLP.b2[o];
        for(int h = 0; h < MLP.hidden_size; h++)
            sum += MLP.w2[h][o] * a1[h];

        z2[o] = sum;
    }

    // store the origin data
    if(config != nullptr){
        (*config).resize(3);
        (*config)[0] = z1;
        (*config)[1] = a1;
        (*config)[2] = z2;
    }

    // normalisze
    softmax(z2);
    return z2;
}

void digit_OCR::MLP_train_once(const int& label, const std::vector<double>& image){
    // forward/probability and z1&z2
    std::vector<std::vector<double>> config;
    auto p = MLP_forward(image, &config);

    // target
    auto target_p = target_generator(label);

    // predict error(derivative z2)
    std::vector<double> dz2(output);
    for(int i = 0; i < output; i++)
        dz2[i] = p[i] - target_p[i];

    // hidden error(derivative z1)
    std::vector<double> dz1(MLP.hidden_size);
    for(int h = 0; h < MLP.hidden_size; h++){
        double sum = 0;
        for (int o = 0; o < output; o++) {
            sum += dz2[o] * MLP.w2[h][o];
        }

        dz1[h] = sum * ReLU_derivative(config[0][h]);
    }
    
    // update w2 and b2 
    // *** must earlier than w1 and b1 ***
    for(int h = 0; h < MLP.hidden_size; h++)
        for(int o = 0; o < output; o++)
            MLP.w2[h][o] -= learning_rate * config[1][h] * dz2[o];
    for (int o = 0; o < output; o++)
        MLP.b2[o] -= learning_rate * dz2[o];

    // update w1 and b1
    for(int i = 0; i < MLP.input_size; i++)
        for(int h = 0; h < MLP.hidden_size; h++)
            MLP.w1[i][h] -= learning_rate * image[i] * dz1[h];
    for (int h = 0; h < MLP.hidden_size; h++)
        MLP.b1[h] -= learning_rate * dz1[h];
}

void digit_OCR::MLP_train(const MNIST::data_1D& train, const MNIST::data_1D& test, int epochs){
    std::cout << "Start MLP training, target epochs: " << epochs << std::endl;
    for(int i = 0; i < epochs; i++){
        for(int j = 0; j < train.label.size(); j++)
            MLP_train_once(train.label[j], train.image[j]);
        
        auto test_result = MLP_test(test);
        std::printf("Epoch: %d\nAccuracy: %.1f%%\nLoss: %.3f\n\n", i+1, test_result[0], test_result[1]);
    }

    std::cout << "MLP is trained successfully." << std::endl;

    // save after trained
    save_MLP();
    std::cout << "MLP parameters are saved." << std::endl;
}


std::vector<double> digit_OCR::MLP_test(const MNIST::data_1D& test){
    std::vector<double> ans(2, 0);
    double accurate = 0;
    double loss_sum = 0;
    for(int i = 0; i < test.label.size(); i++){
        // probility
        auto p = MLP_forward(test.image[i], nullptr);

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