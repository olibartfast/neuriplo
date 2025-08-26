set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/OpenVINOInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(OpenVINOInferTest ${TEST_SOURCES})

target_include_directories(OpenVINOInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/openvino/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_link_libraries(OpenVINOInferTest PRIVATE
    neuriplo
    openvino::runtime
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    glog::glog
    gtest
    gtest_main
)

add_test(NAME OpenVINOInferTest COMMAND OpenVINOInferTest)

# Copy scripts to the build directory
configure_file(${CMAKE_CURRENT_LIST_DIR}/generate_openvino_ir.sh  ${CMAKE_CURRENT_BINARY_DIR}/generate_openvino_ir.sh FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
configure_file(${CMAKE_CURRENT_LIST_DIR}/export_torchvision_classifier.py  ${CMAKE_CURRENT_BINARY_DIR}/export_torchvision_classifier.py COPYONLY)
