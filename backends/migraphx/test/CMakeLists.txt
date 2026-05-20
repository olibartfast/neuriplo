set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/MIGraphXInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

# MIGraphX ships its CMake config under /opt/rocm; ensure it is found
list(APPEND CMAKE_PREFIX_PATH "${MIGRAPHX_ROOT}")
find_package(migraphx REQUIRED)

add_executable(MIGraphXInferTest ${TEST_SOURCES})

target_include_directories(MIGraphXInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/migraphx/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_link_libraries(MIGraphXInferTest PRIVATE
    neuriplo
    migraphx::c
    ${OpenCV_LIBS}
    gtest
    gtest_main
    glog::glog
)

add_test(NAME MIGraphXInferTest COMMAND MIGraphXInferTest)

configure_file(${CMAKE_CURRENT_LIST_DIR}/generate_model.sh
    ${CMAKE_CURRENT_BINARY_DIR}/generate_model.sh FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
configure_file(${CMAKE_CURRENT_LIST_DIR}/export_onnx_classifier.py
    ${CMAKE_CURRENT_BINARY_DIR}/export_onnx_classifier.py COPYONLY)
