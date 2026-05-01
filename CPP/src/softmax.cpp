#include <vector>
#include <cmath>

#include "OCR.hpp"

namespace Digit_OCR{
void digit_OCR::softmax(std::vector<double>& z){
    //find the max val
    double max_val = z[0];
    for(const auto& item : z)
        max_val = max_val > item ? max_val : item;

    // get the sum
    double sum = 0;
    for(int i = 0; i < z.size(); i++){
        z[i] = exp(z[i] - max_val);
        sum += z[i];
    }
    
    // normalize
    for(int i = 0; i < z.size(); i++)
        z[i] /= sum;
}
}