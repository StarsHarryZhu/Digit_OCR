#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "MNIST_decoder.hpp"
#include "MLP_data.hpp"
#include "CNN_data.hpp"

namespace Digit_OCR{
class digit_OCR{

public:
    static digit_OCR& get_instance();
    void init();
    int MLP_OCR(std::string path);
    int CNN_OCR(std::string path);
    int mixed_OCR(std::string path);
    void quit();

private:
    // model
    double ReLU(const double& x);
    double ReLU_derivative(const double& x);
    void softmax(std::vector<double>& z);
    int predict(const std::vector<double>& p);
    std::vector<double> target_generator(int n);


    // MLP
    void MLP_init();
    // config is the origin z1 a1 z2
    std::vector<double> MLP_forward(const std::vector<double>& input, std::vector<std::vector<double>>* config);
    void MLP_train_once(const int& label, const std::vector<double>& image);
    // every epoch will fully trained with all train data set.
    void MLP_train(const MNIST::data_1D& train, const MNIST::data_1D& test, int epochs);
    // return {accurcy(%), loss}
    std::vector<double> MLP_test(const MNIST::data_1D& test);

    void save_MLP();
    void load_MLP();




    // CNN
    void CNN_init();
    std::vector<std::vector<std::vector<double>>> CNN_conv_forward(const std::vector<std::vector<double>>& input);
    std::vector<std::vector<std::vector<double>>> CNN_maxpool(const std::vector<std::vector<std::vector<double>>>& filted);


    void load_CNN();
    void save_CNN();






    // save helper
    inline void save_single(std::ofstream& file, const size_t& val);
    inline void save_vector(std::ofstream& file, const std::vector<double>& v);
    inline void save_vector_1D(std::ofstream& file, const std::vector<double>& v);
    inline void save_vector_2D(std::ofstream& file, const std::vector<std::vector<double>>& v);

    // load helper
    inline size_t load_single(std::ifstream& file);
    inline void load_vector(std::ifstream& file, std::vector<double>& v);
    inline void load_vector_1D(std::ifstream& file, std::vector<double>& v);
    inline void load_vector_2D(std::ifstream& file, std::vector<std::vector<double>>& v);


    // image loader
    std::vector<double> image_loader_1D(const std::string& path);
    std::vector<std::vector<double>> image_loader_2D(const std::string& path);
    // pre-process the image and find if there is digit(return true)
    bool enhance_image(cv::Mat& img);


    digit_OCR();
    ~digit_OCR();
    digit_OCR operator= (digit_OCR other){
        return get_instance();
    };

public:

private:
    inline static digit_OCR* pinstance = nullptr;
    const int row_default = 28;
    const int col_default = 28;

    // train and test assets location
    const std::string train_label = "MNIST/Train/train-labels-idx1-ubyte";
    const std::string train_image = "MNIST/Train/train-images-idx3-ubyte";
    const std::string test_label = "MNIST/Test/t10k-labels-idx1-ubyte";
    const std::string test_image = "MNIST/Test/t10k-images-idx3-ubyte";


    MLP_data MLP;
    CNN_data CNN;

    const double learning_rate = 0.01;
    const int output = 10;

};
}