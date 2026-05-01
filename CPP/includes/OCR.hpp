#pragma once

#include <string>
#include <vector>

namespace Digit_OCR{
class digit_OCR{

public:
    static digit_OCR& get_instance();
    void init();
    int OCR(std::string path);
    void quit();

//private:
    double ReLU(const double& x);
    void softmax(std::vector<double>& z);

    void MLP_init();
    std::vector<double> MLP_forward(const std::vector<double>& x);

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


    const int MLP_input_size = 768;
    const int MLP_hidden_size = 128;
    std::vector<std::vector<double>> MLP_w1;
    std::vector<double> MLP_b1;
    std::vector<std::vector<double>> MLP_w2;
    std::vector<double> MLP_b2;

    const int output = 10;

};
}