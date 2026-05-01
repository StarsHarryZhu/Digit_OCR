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
    std::vector<std::vector<double>> MLP_forward(const std::vector<double>& x);
    void MLP_train_once(const int& label, const std::vector<double>& image);
    // every epoch will fully trained with all train data set.
    void MLP_train(const MNIST::data_1D& train, const MNIST::data_1D& test, int epochs);
    // return {accurcy(%), loss}
    std::vector<double> MLP_test(const MNIST::data_1D& test);

    void CNN_init();


    // save and load for model parameters
    void save();
    void load();

    // save helper
    void save_MLP();
    void save_CNN();
    inline void save_single(std::ofstream& file, const size_t& val);
    inline void save_vector(std::ofstream& file, const std::vector<double>& v);
    inline void save_vector_1D(std::ofstream& file, const std::vector<double>& v);
    inline void save_vector_2D(std::ofstream& file, const std::vector<std::vector<double>>& v);

    // load helper
    void load_MLP();
    void load_CNN();
    inline size_t load_single(std::ifstream& file);
    inline void load_vector(std::ifstream& file, std::vector<double>& v);
    inline void load_vector_1D(std::ifstream& file, std::vector<double>& v);
    inline void load_vector_2D(std::ifstream& file, std::vector<std::vector<double>>& v);


    digit_OCR() = default;
    ~digit_OCR();
    digit_OCR operator= (digit_OCR other){
        return get_instance();
    };

public:

private:
    inline static digit_OCR* pinstance = nullptr;

    // save & load location
    const int MLP_code = 114;
    const int CNN_code = 514;
    const std::string MLP_file = "model/MLP.bin";
    const std::string CNN_file = "model/CNN.bin";

    // train and test assets location
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