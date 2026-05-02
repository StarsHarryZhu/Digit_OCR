#include <opencv2/opencv.hpp>

#include "OCR.hpp"

namespace Digit_OCR{
std::vector<double> digit_OCR::image_loader_1D(const std::string path){
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if(img.empty())
        throw std::runtime_error("cannot load image: " + path);

    if(!enhance_image(img)){
        std::cout << "there is no digit found." << std::endl;
    }

    std::vector<double> result(row_default * col_default);
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            unsigned char pixel = img.at<unsigned char>(r, c);
            result[r * 28 + c] = pixel / 255.0;
        }
    }

    return result;
}

std::vector<std::vector<double>> digit_OCR::image_loader_2D(const std::string path){
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if(img.empty())
        throw std::runtime_error("cannot load image: " + path);

    if(!enhance_image(img)){
        std::cout << "there is no digit found." << std::endl;
    }

    std::vector<std::vector<double>> result(row_default, std::vector<double>(col_default));
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            unsigned char pixel = img.at<unsigned char>(r, c);
            result[r][c] = pixel / 255.0;
        }
    }

    return result;
}

bool digit_OCR::enhance_image(cv::Mat& img){
    cv::threshold(img, img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // if white background with black digit, the color should be reversed
    int white_pixels = cv::countNonZero(img);
    int total_pixels = img.rows * img.cols;
    if (white_pixels > total_pixels / 2)
        cv::bitwise_not(img, img);

    // find and cut the digit region
    std::vector<cv::Point> points;
    cv::findNonZero(img, points);

    if (points.empty()) 
        return false;

    // cut the digit region, max 20 * 20 for 28 * 28 canvas
    cv::Rect box = cv::boundingRect(points);
    cv::Mat digit = img(box);
    
    double scale = std::min(0.9 * row_default / digit.rows, 0.9 * col_default / digit.cols);


    int r = digit.rows * scale;
    int c = digit.cols * scale;

    cv::Mat resized;
    cv::resize(digit, resized, cv::Size(c, r), 0, 0, cv::INTER_AREA);

    // canvas
    img = cv::Mat::zeros(28, 28, CV_8UC1);

    // put the digit onto the canvas

    int x_offset = (col_default - c) / 2;
    int y_offset = (row_default - r) / 2;

    resized.copyTo(img(cv::Rect(x_offset, y_offset, c, r)));

    return true;
}

}