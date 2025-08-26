
set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/OCVDNNInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

add_executable(OCVDNNInferTest ${TEST_SOURCES})

target_include_directories(OCVDNNInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/opencv-dnn/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)


target_link_libraries(OCVDNNInferTest PRIVATE
    neuriplo
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    gtest
    gtest_main
    glog::glog
)

add_test(NAME OCVDNNInferTest COMMAND OCVDNNInferTest)


# Copy scripts to the build directory
configure_file(${CMAKE_CURRENT_LIST_DIR}/generate_model.sh  ${CMAKE_CURRENT_BINARY_DIR}/generate_model.sh FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
configure_file(${CMAKE_CURRENT_LIST_DIR}/export_torchvision_classifier.py  ${CMAKE_CURRENT_BINARY_DIR}/export_torchvision_classifier.py COPYONLY)

