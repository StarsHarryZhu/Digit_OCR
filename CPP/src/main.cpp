#include <iostream>
#include <string>

#include "OCR.hpp"

int main(){
    const std::string img3 = "assets/3.png";
    
    auto& i = Digit_OCR::digit_OCR::get_instance();

    std::cout << i.CNN_OCR(img3);
    
    // std::cout << "Predict answer: " << i.MLP_OCR(img3) << std::endl;
    return EXIT_SUCCESS;
}