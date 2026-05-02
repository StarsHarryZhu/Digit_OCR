#include "MNIST_decoder.hpp"

#include <stdexcept>

namespace Digit_OCR::MNIST{
    data_1D decoder_1D(const std::string& label_path, const std::string& image_path){
        std::ifstream label_file(label_path, std::ios::binary);
        if(read_big_endian(label_file) != 2049)
            throw std::runtime_error("cannot read label file: " + label_path);
        
        std::ifstream image_file(image_path, std::ios::binary);
        if(read_big_endian(image_file) != 2051)
            throw std::runtime_error("cannot read image file: " + image_path);

        int count = read_big_endian(label_file);
        if(count != read_big_endian(image_file))
            std::runtime_error("the number of label and image are not same");
        int row = read_big_endian(image_file);
        int col = read_big_endian(image_file);

        
        data_1D data(count, row, col);
        for(int i = 0; i < count; i++){
            unsigned char byte;
            label_file.read(reinterpret_cast<char*>(&byte), 1);
            data.label[i] = byte;

            for(int j = 0; j < row * col; j++){
                image_file.read(reinterpret_cast<char*>(&byte), 1);
                data.image[i][j] = byte / 255.0;
            }
        }

        return data;
    }

    data_2D decoder_2D(const std::string& label_path, const std::string& image_path){
        std::ifstream label_file(label_path, std::ios::binary);
        if(read_big_endian(label_file) != 2049)
            throw std::runtime_error("cannot read label file: " + label_path);
        
        std::ifstream image_file(image_path, std::ios::binary);
        if(read_big_endian(image_file) != 2051)
            throw std::runtime_error("cannot read image file: " + image_path);

        int count = read_big_endian(label_file);
        if(count != read_big_endian(image_file))
            std::runtime_error("the number of label and image are not same");
        int row = read_big_endian(image_file);
        int col = read_big_endian(image_file);

        data_2D data(count, row, col);
        for(int i = 0; i < count; i++){
            unsigned char byte;
            label_file.read(reinterpret_cast<char*>(&byte), 1);
            data.label[i] = byte;

            for(int j = 0; j < row; j++){
                for(int k = 0; k < col; k++){
                    image_file.read(reinterpret_cast<char*>(&byte), 1);
                    data.image[i][j][k] = byte / 255.0;
                }
            }
        }

        return data;
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