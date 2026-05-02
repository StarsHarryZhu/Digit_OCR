include_guard(GLOBAL)

set(
    DIGIT_OCR_OPENCV_COMPONENTS
    "core;imgproc;imgcodecs;highgui"
    CACHE STRING
    "OpenCV components required by Digit_OCR"
)

if(APPLE AND NOT OpenCV_DIR)
    set(_digit_ocr_opencv_candidates
        "/opt/homebrew/opt/opencv/lib/cmake/opencv4/OpenCVConfig.cmake"
        "/usr/local/opt/opencv/lib/cmake/opencv4/OpenCVConfig.cmake"
    )

    foreach(_digit_ocr_opencv_config ${_digit_ocr_opencv_candidates})
        if(EXISTS "${_digit_ocr_opencv_config}")
            get_filename_component(
                _digit_ocr_opencv_dir
                "${_digit_ocr_opencv_config}"
                DIRECTORY
            )
            set(
                OpenCV_DIR
                "${_digit_ocr_opencv_dir}"
                CACHE PATH
                "Path to OpenCVConfig.cmake"
                FORCE
            )
            break()
        endif()
    endforeach()

    unset(_digit_ocr_opencv_dir)
    unset(_digit_ocr_opencv_candidates)
    unset(_digit_ocr_opencv_config)
endif()

find_package(OpenCV REQUIRED COMPONENTS ${DIGIT_OCR_OPENCV_COMPONENTS})

add_library(digit_ocr_opencv INTERFACE)
add_library(digit_ocr::opencv ALIAS digit_ocr_opencv)

target_include_directories(digit_ocr_opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(digit_ocr_opencv INTERFACE ${OpenCV_LIBS})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
