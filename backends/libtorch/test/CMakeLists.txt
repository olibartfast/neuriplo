set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/LibtorchInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

add_executable(LibtorchInferTest ${TEST_SOURCES})

target_include_directories(LibtorchInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/libtorch/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)


target_link_libraries(LibtorchInferTest PRIVATE
    neuriplo
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    gtest
    gtest_main
    ${GLOG_LIBRARIES}
    glog::glog
)

add_test(NAME LibtorchInferTest COMMAND LibtorchInferTest)


# Copy model_downloader.py script to the build directory
configure_file(${CMAKE_SOURCE_DIR}/scripts/model_downloader.py  ${CMAKE_CURRENT_BINARY_DIR}/model_downloader.py COPYONLY)

