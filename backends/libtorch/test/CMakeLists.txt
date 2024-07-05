
set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/LibtorchInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

add_executable(LibtorchInferTest ${TEST_SOURCES})

target_include_directories(LibtorchInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/libtorch/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}   
)


target_link_libraries(LibtorchInferTest PRIVATE
    InferenceEngines
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    gtest
    gtest_main
)

add_test(NAME LibtorchInferTest COMMAND LibtorchInferTest)
