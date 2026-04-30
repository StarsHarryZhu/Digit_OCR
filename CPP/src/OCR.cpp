#include "OCR.hpp"

namespace Digit_OCR{
CL_digit_OCR& CL_digit_OCR::get_instance(){
    if(!pinstance){
        pinstance = new CL_digit_OCR();
        pinstance->init();
    }
    return *pinstance;
}

void CL_digit_OCR::init(){

}

int CL_digit_OCR::OCR(std::string path){

}

void CL_digit_OCR::quit(){
    delete pinstance;
    pinstance = nullptr;
}

CL_digit_OCR::~CL_digit_OCR(){
    
}

}