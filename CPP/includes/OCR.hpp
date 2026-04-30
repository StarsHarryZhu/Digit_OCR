#pragma once

#include <string>

#include "data_structure.hpp"

namespace Digit_OCR{
class CL_digit_OCR{

public:
    static CL_digit_OCR& get_instance();
    void init();
    int OCR(std::string path);
    void quit();

private:
CL_digit_OCR() = default;
~CL_digit_OCR();
CL_digit_OCR operator= (CL_digit_OCR other){};

public:

private:
inline static CL_digit_OCR* pinstance = nullptr;

const std::string train_label = "MNIST/Train/train-labels-idx1-ubyte";
const std::string train_image = "MNIST/Train/train-images-idx3-ubyte";
const std::string test_label = "MNIST/Test/t10k-labels-idx1-ubyte";
const std::string test_image = "MNIST/Test/t10k-images-idx3-ubyte";

};
}