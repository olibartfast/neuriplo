cmake_minimum_required(VERSION 3.10)
project(InferenceEngines)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
message(STATUS "InferenceEngines Cmake module path: ${CMAKE_MODULE_PATH}")

# Define the default backend if not set from the command line
if(NOT DEFINED DEFAULT_BACKEND)
    unset(DEFAULT_BACKEND CACHE)
    set(DEFAULT_BACKEND "LIBTORCH" CACHE STRING "Default backend for inference" FORCE)
endif()

# Define the supported backends
set(SUPPORTED_BACKENDS "ONNX_RUNTIME" "LIBTORCH" "LIBTENSORFLOW" "OPENCV_DNN" "TENSORRT" "OPENVINO")

# Check if the specified backend is supported
list(FIND SUPPORTED_BACKENDS ${DEFAULT_BACKEND} SUPPORTED_BACKEND_INDEX)
if (SUPPORTED_BACKEND_INDEX EQUAL -1)
    message(STATUS "Unsupported default backend: ${DEFAULT_BACKEND}")
    set(DEFAULT_BACKEND "OPENCV_DNN" CACHE STRING "Default backend for inference" FORCE)
endif()

message(STATUS "Default backend: ${DEFAULT_BACKEND}")

set(INFER_ROOT ${CMAKE_CURRENT_LIST_DIR}/src)


include(SelectBackend)

# Add source files for inference engines
set(INFERENCE_SOURCES src/InferenceInterface.cpp)

# Create the inference engines library
add_library(${PROJECT_NAME} SHARED ${INFERENCE_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}    
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
)
