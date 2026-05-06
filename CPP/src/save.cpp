#include <fstream>

#include "OCR.hpp"

namespace Digit_OCR{
void digit_OCR::save_MLP(){
    std::filesystem::create_directories(
        std::filesystem::path(MLP.file).parent_path()
    );
    std::ofstream file(MLP.file, std::ios::binary);
    if(!file.is_open())
        throw std::runtime_error("cannot open the MLP file to save data");
    
    save_single(file, MLP.code);
    save_vector_2D(file, MLP.w1);
    save_vector_1D(file, MLP.b1);
    save_vector_2D(file, MLP.w2);
    save_vector_1D(file, MLP.b2);

    file.close();
}

void digit_OCR::save_CNN(){
    std::filesystem::create_directories(
        std::filesystem::path(CNN.file).parent_path()
    );
    std::ofstream file(CNN.file, std::ios::binary);
    if(!file.is_open())
        throw std::runtime_error("cannot open the MLP file to save data");
    
    save_single(file, CNN.code);
    save_vector_3D(file, CNN.conv_filters);
    save_vector_1D(file, CNN.conv_b);
    save_vector_2D(file, CNN.flatten_w);
    save_vector_1D(file, CNN.flatten_b);

    file.close();
}





inline void digit_OCR::save_single(std::ofstream& file, const size_t& val){
    file.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

inline void digit_OCR::save_vector(std::ofstream& file, const std::vector<double>& v){
    file.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(double));
}

inline void digit_OCR::save_vector_1D(std::ofstream& file, const std::vector<double>& v){
    save_single(file, v.size());
    save_vector(file, v);
}

inline void digit_OCR::save_vector_2D(std::ofstream& file, const std::vector<std::vector<double>>& v){
    save_single(file, v.size());
    save_single(file, v[0].size());
    for(const auto& item : v)
        save_vector(file, item);
}

inline void digit_OCR::save_vector_3D(std::ofstream& file, const std::vector<std::vector<std::vector<double>>>& v){
    save_single(file, v.size());
    save_single(file, v[0].size());
    save_single(file, v[0][0].size());
    for(const auto& item : v)
        save_vector_2D(file, item);
}
}