#include <iostream>
#include <string>
#include <fstream>

#include "OCR.hpp"
#include "MNIST_decoder.hpp"

int main(){
    const std::string train_label = "MNIST/Train/train-labels-idx1-ubyte";
    const std::string train_image = "MNIST/Train/train-images-idx3-ubyte";
    auto train_1D = Digit_OCR::MNIST::decoder_1D(train_label, train_image);
    auto train_2D = Digit_OCR::MNIST::decoder_2D(train_label, train_image);

    const std::string test_label = "MNIST/Test/t10k-labels-idx1-ubyte";
    const std::string test_image = "MNIST/Test/t10k-images-idx3-ubyte";
    auto test_1D = Digit_OCR::MNIST::decoder_1D(test_label, test_image);
    auto test_2D = Digit_OCR::MNIST::decoder_2D(test_label, test_image);

    auto instance = Digit_OCR::digit_OCR::get_instance();

    const std::string img_path = "assets/3.png";
    auto resized = instance.image_loader_1D(img_path);

    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
            if(resized[i*28 + j] == 0)
                std::cout << ".";
            else
                std::cout << "#";
        }
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}