cmake_minimum_required(VERSION 3.10)
project(InferenceEngines)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

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

set(INFER_ROOT ${CMAKE_CURRENT_LIST_DIR}/backends)

# Add source files for inference engines
set(SOURCES ${CMAKE_CURRENT_LIST_DIR}/backends/src/InferenceInterface.cpp)

include(SelectBackend)

# Create the inference engines library
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/backends/src
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
)

include(LinkBackend)
include(SetCompilerFlags)

# Add GoogleTest
# Include directories for tests
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
unset(BUILD_INFERENCE_ENGINE_TESTS CACHE)
option(BUILD_INFERENCE_ENGINE_TESTS "Build tests (optional)" ON)
if(BUILD_INFERENCE_ENGINE_TESTS) 
include(SetupTests) 
endif(BUILD_INFERENCE_ENGINE_TESTS)  
