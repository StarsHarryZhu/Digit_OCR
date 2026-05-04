#include <iostream>
#include <string>

#include "OCR.hpp"

int main(){
    const std::string train_label = "MNIST/Train/train-labels-idx1-ubyte";
    const std::string train_image = "MNIST/Train/train-images-idx3-ubyte";
    const std::string test_label = "MNIST/Test/t10k-labels-idx1-ubyte";
    const std::string test_image = "MNIST/Test/t10k-images-idx3-ubyte";

    const std::string img3 = "assets/3.png";

    auto train_2d = Digit_OCR::MNIST::decoder_2D(train_label, train_image);
    auto test_2d = Digit_OCR::MNIST::decoder_2D(test_label, test_image);
    
    auto& i = Digit_OCR::digit_OCR::get_instance();

    i.CNN_init();
    auto p = i.CNN_forward(train_2d.image[0]);

    std::cout << p.size() << std::endl;

    double sum = 0;
    for(int i = 0; i < p.size(); i++)
        sum += p[i];
    
    std::cout << sum;
    
    // std::cout << "Predict answer: " << i.MLP_OCR(img3) << std::endl;
    return EXIT_SUCCESS;
}