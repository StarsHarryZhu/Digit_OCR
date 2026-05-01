#pragma once

#include <string>
#include <vector>

#include "MNIST_decoder.hpp"

namespace Digit_OCR{
class digit_OCR{

public:
    static digit_OCR& get_instance();
    void init();
    int OCR(std::string path);
    void quit();

//private:
    double ReLU(const double& x);
    double ReLU_derivative(const double& x);
    void softmax(std::vector<double>& z);
    int predict(const std::vector<double>& p);
    std::vector<double> target_generator(int n);

    void MLP_init();
    // return the raw vals, if want get probility, please use p at the second argument
    std::vector<std::vector<double>> MLP_forward(const std::vector<double>& x, std::vector<double>& p);
    void MLP_train_once(const int& label, const std::vector<double>& image);
    // report every 10% train result: accuracy and loss
    // the epochs will be multiplied by 10
    void MLP_train(const MNIST::data_1D& train, const MNIST::data_1D& test, int epochs);
    // return {accurcy(%), loss}
    std::vector<double> MLP_test(const MNIST::data_1D& test);

    void CNN_init();


    digit_OCR() = default;
    ~digit_OCR();
    digit_OCR operator= (digit_OCR other){
        return get_instance();
    };

public:

private:
    inline static digit_OCR* pinstance = nullptr;

    const std::string train_label = "MNIST/Train/train-labels-idx1-ubyte";
    const std::string train_image = "MNIST/Train/train-images-idx3-ubyte";
    const std::string test_label = "MNIST/Test/t10k-labels-idx1-ubyte";
    const std::string test_image = "MNIST/Test/t10k-images-idx3-ubyte";


    const int MLP_input_size = 784;
    const int MLP_hidden_size = 128;
    std::vector<std::vector<double>> MLP_w1;
    std::vector<double> MLP_b1;
    std::vector<std::vector<double>> MLP_w2;
    std::vector<double> MLP_b2;

    const double learning_rate = 0.01;
    const int output = 10;

};
}