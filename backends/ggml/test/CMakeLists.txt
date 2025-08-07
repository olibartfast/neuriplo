set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/GGMLInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

add_executable(GGMLInferTest ${TEST_SOURCES})

target_include_directories(GGMLInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/ggml/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${GGML_DIR}/include
)

target_link_libraries(GGMLInferTest PRIVATE
    InferenceEngines
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    ${GGML_DIR}/lib/libggml.so
    gtest
    gtest_main
    glog::glog
)

add_test(NAME GGMLInferTest COMMAND GGMLInferTest)
