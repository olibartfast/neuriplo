set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ONNXRuntimeInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

add_executable(ONNXRuntimeInferTest ${TEST_SOURCES})

target_include_directories(ONNXRuntimeInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/onnx-runtime/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${ONNX_RUNTIME_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_link_libraries(ONNXRuntimeInferTest PRIVATE
    neuriplo
    ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so
    ${OpenCV_LIBS}
    gtest
    gtest_main
    ${GLOG_LIBRARIES}
    glog::glog
)

add_test(NAME ONNXRuntimeInferTest COMMAND ONNXRuntimeInferTest)

# Copy scripts to the build directory
configure_file(${CMAKE_CURRENT_LIST_DIR}/generate_model.sh  ${CMAKE_CURRENT_BINARY_DIR}/generate_model.sh FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
configure_file(${CMAKE_CURRENT_LIST_DIR}/export_torchvision_classifier.py  ${CMAKE_CURRENT_BINARY_DIR}/export_torchvision_classifier.py COPYONLY)
