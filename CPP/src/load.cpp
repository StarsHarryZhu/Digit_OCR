#include <fstream>
#include <iostream>

#include "OCR.hpp"

namespace Digit_OCR{
void digit_OCR::load(){
    load_MLP();
    load_CNN();
}

void digit_OCR::load_MLP(){
    std::ifstream file(MLP_file, std::ios::binary);
    if(!file.is_open()){
        std::cout << "cannot open the MLP file to load data." << std::endl;
        MLP_init();
        return ;
    }

    if(load_single(file) != MLP_code){
        std::cout << "The file is not MLP file." << std::endl;
        MLP_init();
        return ;
    }

    load_vector_2D(file, MLP_w1);
    if(MLP_w1.size() != MLP_input_size || MLP_w1[0].size() != MLP_hidden_size){
        std::cout << "fail to load MLP file." << std::endl;
        MLP_init();
        return ;
    }

    load_vector_1D(file, MLP_b1);
    if(MLP_b1.size() != MLP_hidden_size){
        std::cout << "fail to load MLP file." << std::endl;
        MLP_init();
        return ;
    }

    load_vector_2D(file, MLP_w2);
    if(MLP_w2.size() != MLP_hidden_size || MLP_w2[0].size() != output){
        std::cout << "fail to load MLP file." << std::endl;
        MLP_init();
        return ;
    }

    load_vector_1D(file, MLP_b2);
    if(MLP_b2.size() != output){
        std::cout << "fail to load MLP file." << std::endl;
        MLP_init();
        return ;
    }

}

void digit_OCR::load_CNN(){

}

inline size_t digit_OCR::load_single(std::ifstream& file){
    size_t val;
    file.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

inline void digit_OCR::load_vector(std::ifstream& file, std::vector<double>& v){
    file.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(double));
}

inline void digit_OCR::load_vector_1D(std::ifstream& file, std::vector<double>& v){
    size_t size = load_single(file);
    v.resize(size);
    load_vector(file, v);
}

inline void digit_OCR::load_vector_2D(std::ifstream& file, std::vector<std::vector<double>>& v){
    size_t row = load_single(file);
    size_t col = load_single(file);
    v.resize(row);
    for(auto& item : v){
        item.resize(col);
        load_vector(file, item);
    }
}

}