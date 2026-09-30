# NATIVE adapter tests. Wired like ONNXRuntimeInferTest: the shared test setup
# add_subdirectory()s every enabled backend's test directory, so this target is
# built only when NATIVE is an enabled backend.

set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/NativeInferTest.cpp
)

add_executable(NativeInferTest ${TEST_SOURCES})
target_compile_features(NativeInferTest PRIVATE cxx_std_17)
target_compile_options(NativeInferTest PRIVATE -Wall -Wextra -Wpedantic)

target_include_directories(NativeInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/native/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(NativeInferTest PRIVATE
    neuriplo
    neuriplo_engine
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME NativeInferTest COMMAND NativeInferTest)

# Fixture resolvable via env var or CMake cache variable of the same name, with
# the well-known local export location as a fallback. The tests ASSERT (fail,
# never skip) when the fixture is absent; the environment property below is the
# channel from the cache variable to the test process.
set(NEURIPLO_NATIVE_FIXTURE "" CACHE PATH
    "Path to the opset-18 ResNet-18 ONNX fixture for the NATIVE adapter tests")
if(NOT NEURIPLO_NATIVE_FIXTURE AND EXISTS
    "/tmp/opencode/resnet18_export/resnet18.onnx")
    set(NEURIPLO_NATIVE_FIXTURE
        "/tmp/opencode/resnet18_export/resnet18.onnx")
endif()

# The fixture run takes ~30 s per inference; several cases run it more than once.
set_tests_properties(NativeInferTest PROPERTIES TIMEOUT 1800)

if(NEURIPLO_NATIVE_FIXTURE)
    set_tests_properties(NativeInferTest PROPERTIES
        ENVIRONMENT "NEURIPLO_NATIVE_FIXTURE=${NEURIPLO_NATIVE_FIXTURE}")
endif()
