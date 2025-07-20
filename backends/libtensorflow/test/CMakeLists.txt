set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/TensorFlowInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

# Add cmake modules path to find our custom FindTensorFlow.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(TensorFlow REQUIRED)

add_executable(TensorFlowInferTest ${TEST_SOURCES})

target_include_directories(TensorFlowInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/libtensorflow/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${TensorFlow_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_link_libraries(TensorFlowInferTest PRIVATE
    InferenceEngines
    ${TensorFlow_CC_LIBRARY}
    ${TensorFlow_FRAMEWORK_LIBRARY}
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    glog::glog
    gtest
    gtest_main
)

add_test(NAME TensorFlowInferTest COMMAND TensorFlowInferTest)

# Copy model_downloader.py script to the build directory
configure_file(${CMAKE_SOURCE_DIR}/scripts/model_downloader.py  ${CMAKE_CURRENT_BINARY_DIR}/model_downloader.py COPYONLY)
