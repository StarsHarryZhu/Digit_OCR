#include <fstream>

#include "OCR.hpp"

namespace Digit_OCR{

void digit_OCR::save(){
    save_MLP();
    save_CNN();
}


void digit_OCR::save_MLP(){
    std::filesystem::create_directories(
        std::filesystem::path(MLP_file).parent_path()
    );
    std::ofstream file(MLP_file, std::ios::binary);
    if(!file.is_open())
        throw std::runtime_error("cannot open the MLP file to save data");
    
    save_single(file, MLP_code);
    save_vector_2D(file, MLP_w1);
    save_vector_1D(file, MLP_b1);
    save_vector_2D(file, MLP_w2);
    save_vector_1D(file, MLP_b2);

    file.close();
}

void digit_OCR::save_CNN(){

}





void digit_OCR::save_single(std::ofstream& file, const size_t& val){
    file.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

void digit_OCR::save_vector(std::ofstream& file, const std::vector<double>& v){
    file.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(double));
}

void digit_OCR::save_vector_1D(std::ofstream& file, const std::vector<double>& v){
    save_single(file, v.size());
    save_vector(file, v);
}

void digit_OCR::save_vector_2D(std::ofstream& file, const std::vector<std::vector<double>>& v){
    save_single(file, v.size());
    save_single(file, v[0].size());
    for(const auto& item : v)
        save_vector(file, item);
}
}