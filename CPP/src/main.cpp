#include <iostream>
#include <string>

#include "OCR.hpp"

int main(){
    const std::string img3 = "assets/3.png";
    
    auto& instance = Digit_OCR::digit_OCR::get_instance();
    
    std::cout << "Predict answer: " << instance.MLP_OCR(img3) << std::endl;
    return EXIT_SUCCESS;
}