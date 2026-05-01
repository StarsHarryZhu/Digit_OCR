#include "OCR.hpp"

namespace Digit_OCR{
int digit_OCR::predict(const std::vector<double>& p){
    int ans = 0;
    double max_val = p[0];
    for(int i = 1; i < p.size(); i++){
        if(max_val < p[i]){
            ans = i;
            max_val = p[i];
        }
    }
    return ans;
}
}