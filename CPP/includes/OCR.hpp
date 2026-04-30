#pragma once

#include <string>

namespace Digit_OCR{
class digit_OCR{

public:
    static digit_OCR& get_instance();
    void init();
    int OCR(std::string path);
    void quit();

private:
digit_OCR() = default;
~digit_OCR();
digit_OCR operator= (digit_OCR other){};

public:

private:
inline static digit_OCR* pinstance = nullptr;

const std::string train_label = "MNIST/Train/train-labels-idx1-ubyte";
const std::string train_image = "MNIST/Train/train-images-idx3-ubyte";
const std::string test_label = "MNIST/Test/t10k-labels-idx1-ubyte";
const std::string test_image = "MNIST/Test/t10k-images-idx3-ubyte";

};
}