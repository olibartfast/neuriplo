set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/TVMInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

add_executable(TVMInferTest ${TEST_SOURCES})

find_library(TVM_RUNTIME_LIBRARY
    NAMES tvm_runtime libtvm_runtime
    PATHS
        ${TVM_DIR}/build
        ${TVM_DIR}/build/Release
        ${TVM_DIR}/build/Debug
    NO_DEFAULT_PATH
)
if(NOT TVM_RUNTIME_LIBRARY)
    message(FATAL_ERROR "Unable to locate the TVM runtime test library under ${TVM_DIR}")
endif()

target_include_directories(TVMInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/tvm/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${TVM_DIR}/include
    ${TVM_DIR}/3rdparty/dmlc-core/include
    ${TVM_DIR}/3rdparty/dlpack/include
    ${TVM_DIR}/3rdparty/dlpack
    ${TVM_DIR}/3rdparty/tvm-ffi/3rdparty/dlpack/include
    ${TVM_DIR}/3rdparty/tvm-ffi/include
    ${TVM_DIR}/3rdparty
)

target_link_libraries(TVMInferTest PRIVATE
    neuriplo
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    ${TVM_RUNTIME_LIBRARY}
    gtest
    gtest_main
    glog::glog
)

# Suppress macro redefinition warnings between glog and DMLC
target_compile_options(TVMInferTest PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:CXX>:-w>)

add_test(NAME TVMInferTest COMMAND TVMInferTest)
