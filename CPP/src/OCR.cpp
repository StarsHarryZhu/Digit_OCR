#include "OCR.hpp"

namespace Digit_OCR{
digit_OCR& digit_OCR::get_instance(){
    if(!pinstance){
        pinstance = new digit_OCR();
        pinstance->init();
    }
    return *pinstance;
}

void digit_OCR::init(){

}

int digit_OCR::OCR(std::string path){

}

void digit_OCR::quit(){
    delete pinstance;
    pinstance = nullptr;
}

digit_OCR::~digit_OCR(){
    
}

}