set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/LibtorchInferTest.cpp
    ${CMAKE_CURRENT_LIST_DIR}/LibtorchRuntimeFactoryTest.cpp
    ${CMAKE_CURRENT_LIST_DIR}/LibtorchInputMetadataTest.cpp
)

find_package(Glog REQUIRED)

add_executable(LibtorchInferTest ${TEST_SOURCES})

target_include_directories(LibtorchInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/libtorch/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${GLOG_INCLUDE_DIRS}
)


target_link_libraries(LibtorchInferTest PRIVATE
    neuriplo
    ${TORCH_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
    ${GLOG_LIBRARIES}
    glog::glog
)

add_test(NAME LibtorchInferTest COMMAND LibtorchInferTest)


# Copy model_downloader.py script to the build directory
configure_file(${CMAKE_SOURCE_DIR}/scripts/model_downloader.py  ${CMAKE_CURRENT_BINARY_DIR}/model_downloader.py COPYONLY)


# model_downloader.py's typed export helper is part of the staged input set.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
configure_file(${CMAKE_SOURCE_DIR}/scripts/lib/torchscript_metadata.py
    ${CMAKE_CURRENT_BINARY_DIR}/lib/torchscript_metadata.py COPYONLY)

# The exporter tests invoke this synthetic component probe explicitly. It does
# not need weights, but AOTI interoperability needs the pinned Python compiler.
add_executable(LibtorchInputContractProbe LibtorchInputContractProbe.cpp)
target_include_directories(LibtorchInputContractProbe PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/libtorch/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(LibtorchInputContractProbe PRIVATE neuriplo ${TORCH_LIBRARIES} glog::glog)
