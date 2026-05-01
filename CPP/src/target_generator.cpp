#include "OCR.hpp"

namespace Digit_OCR{
std::vector<double> digit_OCR::target_generator(int n){
    std::vector<double> target(output, 0.0);
    target[n] = 1.0;
    return target;
}
}