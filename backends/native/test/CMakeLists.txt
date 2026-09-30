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

# ---------------------------------------------------------------------------
# NATIVE <-> ONNX_RUNTIME parity (T-27a/T-22, [V-6]). Only meaningful when the
# ONNX Runtime backend is also compiled in: the harness runs the identical file
# and input through both adapters and compares them elementwise.
# ---------------------------------------------------------------------------
if("ONNX_RUNTIME" IN_LIST NEURIPLO_ENABLED_BACKENDS)

    find_package(Glog REQUIRED)

    # Mirror the ONNX Runtime test target's library lookup. This directory is
    # added before backends/onnx-runtime/test, so the variable is not defined
    # yet; resolving it here keeps the parity target self-contained.
    find_library(ONNX_RUNTIME_LIBRARY
        NAMES onnxruntime libonnxruntime
        PATHS
            ${ONNX_RUNTIME_DIR}/lib
            ${ONNX_RUNTIME_DIR}/lib/Release
            ${ONNX_RUNTIME_DIR}/lib/Debug
        NO_DEFAULT_PATH
    )
    if(NOT ONNX_RUNTIME_LIBRARY)
        message(FATAL_ERROR "Unable to locate the ONNX Runtime library under ${ONNX_RUNTIME_DIR}")
    endif()

    # Deterministic fixture provisioning: exports the opset-18 ResNet-18 model
    # (never to the old hard-coded /workspace/). Always re-runs.
    find_program(NEURIPLO_PYTHON3_EXECUTABLE NAMES python3 python)
    if(NOT NEURIPLO_PYTHON3_EXECUTABLE)
        message(FATAL_ERROR "native_parity_fixture needs a Python 3 interpreter to provision the fixture")
    endif()

    set(NEURIPLO_PARITY_FIXTURE_DIR "${CMAKE_BINARY_DIR}/native_parity_fixture")
    add_custom_target(native_parity_fixture
        COMMAND ${CMAKE_COMMAND} -E make_directory "${NEURIPLO_PARITY_FIXTURE_DIR}"
        COMMAND ${NEURIPLO_PYTHON3_EXECUTABLE}
                "${CMAKE_CURRENT_LIST_DIR}/provision_parity_fixture.py"
                "${NEURIPLO_PARITY_FIXTURE_DIR}"
        COMMENT "Provisioning the NATIVE<->ONNX_RUNTIME parity fixture (ResNet-18, opset 18)"
        VERBATIM
        USES_TERMINAL
    )

    add_executable(native_parity_test ${CMAKE_CURRENT_LIST_DIR}/ParityTest.cpp)
    target_compile_features(native_parity_test PRIVATE cxx_std_17)
    target_compile_options(native_parity_test PRIVATE -Wall -Wextra -Wpedantic)

    target_include_directories(native_parity_test PRIVATE
        ${CMAKE_SOURCE_DIR}/backends/native/src
        ${CMAKE_SOURCE_DIR}/backends/onnx-runtime/src
        ${CMAKE_SOURCE_DIR}/backends/src
        ${CMAKE_SOURCE_DIR}/include
        ${ONNX_RUNTIME_DIR}/include
        ${GLOG_INCLUDE_DIRS}
    )

    target_link_libraries(native_parity_test PRIVATE
        neuriplo
        neuriplo_engine
        ${ONNX_RUNTIME_LIBRARY}
        GTest::gtest
        GTest::gtest_main
        ${GLOG_LIBRARIES}
        glog::glog
    )

    add_test(NAME native_parity COMMAND native_parity_test)
    set_tests_properties(native_parity PROPERTIES
        ENVIRONMENT "NEURIPLO_PARITY_FIXTURE=${NEURIPLO_PARITY_FIXTURE_DIR}/resnet18.onnx"
        TIMEOUT 1800
    )

endif()
