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
    std::cout << data_1D.label[0] << std::endl;
    for(int i = 0; i < data_1D.image[0].size(); i++){
        if(i != 0 && i % 28 == 0)
            std::cout << std::endl;
        if(data_1D.image[0][i] == 0)
            std::cout << ".";
        else
            std::cout << "#";
    }

    std::cout << std::endl << std::endl;
    std::cout << data_2D.label[0] << std::endl;
    for(int i = 0; i < data_2D.image[0].size(); i++){
        for(int j = 0; j < data_2D.image[0][i].size(); j++){
            if(data_2D.image[0][i][j] == 0)
                std::cout << ".";
            else
                std::cout << "#";
        }
        std::cout << std::endl;
    }
    return EXIT_SUCCESS;
}