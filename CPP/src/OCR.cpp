#include <fstream>

#include "OCR.hpp"
#include "MNIST_decoder.hpp"

namespace Digit_OCR{
digit_OCR& digit_OCR::get_instance(){
    if(!pinstance){
        pinstance = new digit_OCR();
        pinstance->init();
    }
    return *pinstance;
}

void digit_OCR::init(){
    if(std::filesystem::exists(MLP.file)){
        std::cout << "Found MLP file." << std::endl;
        load_MLP();
    } else {
        std::cout << "Didnt found MLP file, start training (default 3 epochs)." << std::endl;
        MLP_init();
        auto train = MNIST::decoder_1D(train_label, train_image);
        auto test = MNIST::decoder_1D(test_label, test_image);
        MLP_train(train, test, 3);
    }

    // if(std::filesystem::exists(CNN_file)){
    //     std::cout << "Found CNN file." << std::endl;
    //     load_MLP();
    // } else {
    //     std::cout << "Didnt found CNN file, start training (default 3 epochs)." << std::endl;
    //     CNN_init();
    //     auto train = MNIST::decoder_2D(train_label, train_image);
    //     auto test = MNIST::decoder_2D(test_label, test_image);
    //     CNN_train(train, test, 3);
    // }
}

int digit_OCR::MLP_OCR(std::string path){
    auto img = image_loader_1D(path);
    auto p = MLP_forward(img, nullptr);
    return predict(p);
}

int digit_OCR::CNN_OCR(std::string path){
    auto img = image_loader_2D(path);
    auto p = CNN_forward(img);
    return predict(p);
}

int digit_OCR::mixed_OCR(std::string path){

}

void digit_OCR::quit(){
    delete pinstance;
    pinstance = nullptr;
}

digit_OCR::digit_OCR():
    MLP("model/MLP.bin", row_default*col_default, 128),
    CNN("model/CNN.bin", row_default, col_default, 8, 3, 2)
    {}

digit_OCR::~digit_OCR(){
    
}

}