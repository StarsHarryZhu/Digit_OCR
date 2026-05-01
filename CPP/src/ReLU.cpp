#include "OCR.hpp"

namespace Digit_OCR{
double digit_OCR::ReLU(const double& x){
    return x > 0 ? x : 0;
}
double digit_OCR::ReLU_derivative(const double& x){
    return x > 0 ? 1 : 0;
}
}