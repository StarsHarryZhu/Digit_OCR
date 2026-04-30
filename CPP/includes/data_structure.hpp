#pragma once

#include <vector>

namespace Digit_OCR{
    struct ST_image_1D{
        size_t label;
        std::vector<double> image;
    };
    struct ST_image_2D{
        size_t label;
        std::vector<std::vector<double>> image;
    };
}