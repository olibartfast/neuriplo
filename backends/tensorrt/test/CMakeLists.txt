set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/TensorRTInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(CUDA REQUIRED)

add_executable(TensorRTInferTest ${TEST_SOURCES})

target_include_directories(TensorRTInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/tensorrt/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_link_directories(TensorRTInferTest PRIVATE
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

target_link_libraries(TensorRTInferTest PRIVATE
    InferenceEngines
    nvinfer
    nvonnxparser
    cudart
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    glog::glog
    gtest
    gtest_main
)

add_test(NAME TensorRTInferTest COMMAND TensorRTInferTest)

# Copy scripts to the build directory
configure_file(${CMAKE_CURRENT_LIST_DIR}/generate_trt_engine.sh  ${CMAKE_CURRENT_BINARY_DIR}/generate_trt_engine.sh FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
