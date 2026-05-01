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

    std::cout << "train and test assets loaded successfully" << std::endl;


    auto instance = Digit_OCR::digit_OCR::get_instance();
    instance.MLP_train(train_1D, test_1D, 1);


    std::cout << "start save file" << std::endl;
    instance.save();
    std::cout << "file saved, start init and load file for test" << std::endl;
    instance.init();
    instance.load();
    std::cout << "file is loaded successfully" << std::endl;
    auto test_result = instance.MLP_test(test_1D);
    std::printf("Accuracy: %.1f%%\nLoss: %.3f\n\n", test_result[0], test_result[1]);

    return EXIT_SUCCESS;
}