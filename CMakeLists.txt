cmake_minimum_required(VERSION 3.10)
project(InferenceEngines)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "InferenceEngines Cmake module path: ${CMAKE_MODULE_PATH}")

# Read DEFAULT_BACKEND from the cache. If not set, provide a default.
if(NOT DEFINED DEFAULT_BACKEND)
    set(DEFAULT_BACKEND "OPENCV_DNN" CACHE STRING "Default backend for inference")
endif()

# Define the supported backends
set(SUPPORTED_BACKENDS "ONNX_RUNTIME" "LIBTORCH" "LIBTENSORFLOW" "OPENCV_DNN" "TENSORRT" "OPENVINO")

# Check if the specified backend is supported
list(FIND SUPPORTED_BACKENDS ${DEFAULT_BACKEND} SUPPORTED_BACKEND_INDEX)
if(SUPPORTED_BACKEND_INDEX EQUAL -1)
    message(FATAL_ERROR "Unsupported default backend: ${DEFAULT_BACKEND}")
endif()

message(STATUS "Default backend: ${DEFAULT_BACKEND}")

set(INFER_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/backends)

# Add source files for inference engines
set(SOURCES ${CMAKE_CURRENT_LIST_DIR}/backends/src/InferenceInterface.cpp ${CMAKE_CURRENT_LIST_DIR}/backends/src/ModelInfo.cpp ${CMAKE_CURRENT_LIST_DIR}/src/InferenceBackendSetup.cpp)

include(SelectBackend)

# Create the inference engines library
add_library(InferenceEngines SHARED ${SOURCES})
target_include_directories(InferenceEngines PUBLIC     
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>   
)
target_include_directories(InferenceEngines PRIVATE
    ${INFER_ROOT}/src
)

target_link_libraries(InferenceEngines PRIVATE
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
)

include(LinkBackend)
include(SetCompilerFlags)

# Add GoogleTest
# Include directories for tests
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${gtest_SOURCE_DIR}/include)
option(BUILD_INFERENCE_ENGINE_TESTS "Build tests (optional)" OFF)
if(BUILD_INFERENCE_ENGINE_TESTS) 
include(SetupTests) 
endif(BUILD_INFERENCE_ENGINE_TESTS)