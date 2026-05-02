cmake_minimum_required(VERSION 3.20)

project(Digit_OCR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")
include(OpenCV)

file(GLOB_RECURSE DIGIT_OCR_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/CPP/*.cpp"
)

if(NOT DIGIT_OCR_SOURCES)
    message(FATAL_ERROR "No C++ source files found under CPP/")
endif()

add_executable(Digit_OCR ${DIGIT_OCR_SOURCES})

target_include_directories(Digit_OCR PRIVATE
    "${CMAKE_SOURCE_DIR}/CPP/includes"
)
target_link_libraries(Digit_OCR PRIVATE digit_ocr::opencv)

set_target_properties(Digit_OCR PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

if(CMAKE_CONFIGURATION_TYPES)
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${config}" config_upper)
        set_target_properties(Digit_OCR PROPERTIES
            "RUNTIME_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_SOURCE_DIR}"
        )
    endforeach()
endif()
