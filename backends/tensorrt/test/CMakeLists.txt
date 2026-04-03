set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/TensorRTInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(TensorRTInferTest ${TEST_SOURCES})

find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer libnvinfer
    PATHS
        ${TENSORRT_DIR}/lib
        ${TENSORRT_DIR}/lib/Release
        ${TENSORRT_DIR}/lib/Debug
    NO_DEFAULT_PATH
)
find_library(TENSORRT_NVONNXPARSER_LIBRARY
    NAMES nvonnxparser libnvonnxparser
    PATHS
        ${TENSORRT_DIR}/lib
        ${TENSORRT_DIR}/lib/Release
        ${TENSORRT_DIR}/lib/Debug
    NO_DEFAULT_PATH
)
if(NOT TENSORRT_NVINFER_LIBRARY OR NOT TENSORRT_NVONNXPARSER_LIBRARY)
    message(FATAL_ERROR "Unable to locate the TensorRT test libraries under ${TENSORRT_DIR}")
endif()

target_include_directories(TensorRTInferTest SYSTEM PRIVATE
    ${TENSORRT_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_include_directories(TensorRTInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/tensorrt/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(TensorRTInferTest PRIVATE
    neuriplo
    ${TENSORRT_NVINFER_LIBRARY}
    ${TENSORRT_NVONNXPARSER_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    glog::glog
    gtest
    gtest_main
)

add_test(NAME TensorRTInferTest COMMAND TensorRTInferTest)

# Copy scripts to the build directory
configure_file(${CMAKE_CURRENT_LIST_DIR}/generate_trt_engine.sh  ${CMAKE_CURRENT_BINARY_DIR}/generate_trt_engine.sh FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
