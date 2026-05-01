#include <iostream>
#include <string>
#include <fstream>

#include "OCR.hpp"
#include "MNIST_decoder.hpp"

int main(){
    const std::string label_path = "MNIST/Train/train-labels-idx1-ubyte";
    const std::string image_path = "MNIST/Train/train-images-idx3-ubyte";
    auto data_1D = Digit_OCR::MNIST::decoder_1D(label_path, image_path);
    auto data_2D = Digit_OCR::MNIST::decoder_2D(label_path, image_path);

    auto instance = Digit_OCR::digit_OCR::get_instance();
    auto a1 = instance.MLP_forward(data_1D.image[0]);

    for(const auto& item : a1)
        std::cout << item << std::endl;


    return EXIT_SUCCESS;
}