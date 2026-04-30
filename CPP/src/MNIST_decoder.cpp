#include "MNIST_decoder.hpp"

#include <stdexcept>

namespace Digit_OCR::MNIST{
    std::vector<ST_image_1D> decoder_1D(const std::string label_path, const std::string image_path){

    }

    std::vector<ST_image_2D> decoder_2D(const std::string label_path, const std::string image_path){

    }



    inline int decode_1label(int offset){

    }

    inline std::vector<double> decode_1image_1D(int offset){

    }

    inline std::vector<std::vector<double>> decode_1image_2D(int offset){

    }



    int read_big_endian(std::ifstream& file){
        unsigned char bytes[4];
        file.read(reinterpret_cast<char*>(bytes), 4);
        if (!file) {
            throw std::runtime_error("Failed to read int from file.");
        }

        return (bytes[0] << 24) |
            (bytes[1] << 16) |
            (bytes[2] << 8)  |
            (bytes[3]);
    }
}