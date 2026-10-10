# NEURIPLO_ENGINE adapter tests. Wired like ONNXRuntimeInferTest: the shared test setup
# add_subdirectory()s every enabled backend's test directory, so this target is
# built only when NEURIPLO_ENGINE is an enabled backend.

set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/NeuriploEngineInferTest.cpp
)

add_executable(NeuriploEngineInferTest ${TEST_SOURCES})
target_compile_features(NeuriploEngineInferTest PRIVATE cxx_std_17)
target_compile_options(NeuriploEngineInferTest PRIVATE -Wall -Wextra -Wpedantic)

target_include_directories(NeuriploEngineInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/neuriplo-engine/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(NeuriploEngineInferTest PRIVATE
    neuriplo
    neuriplo_engine
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME NeuriploEngineInferTest COMMAND NeuriploEngineInferTest)

# Fixture resolvable via env var or CMake cache variable of the same name, with
# the well-known local export location as a fallback. The tests ASSERT (fail,
# never skip) when the fixture is absent; the environment property below is the
# channel from the cache variable to the test process.
set(NEURIPLO_ENGINE_FIXTURE "" CACHE PATH
    "Path to the opset-18 ResNet-18 ONNX fixture for the NEURIPLO_ENGINE adapter tests")
if(NOT NEURIPLO_ENGINE_FIXTURE AND EXISTS
    "/tmp/opencode/resnet18_export/resnet18.onnx")
    set(NEURIPLO_ENGINE_FIXTURE
        "/tmp/opencode/resnet18_export/resnet18.onnx")
endif()

# The fixture run takes ~30 s per inference; several cases run it more than once.
set_tests_properties(NeuriploEngineInferTest PROPERTIES TIMEOUT 1800)

if(NEURIPLO_ENGINE_FIXTURE)
    set_tests_properties(NeuriploEngineInferTest PROPERTIES
        ENVIRONMENT "NEURIPLO_ENGINE_FIXTURE=${NEURIPLO_ENGINE_FIXTURE}")
endif()

# ---------------------------------------------------------------------------
# NEURIPLO_ENGINE <-> ONNX_RUNTIME parity (T-27a/T-22, [V-6]). Only meaningful when the
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
        message(FATAL_ERROR "neuriplo_engine_parity_fixture needs a Python 3 interpreter to provision the fixture")
    endif()

    set(NEURIPLO_PARITY_FIXTURE_DIR "${CMAKE_BINARY_DIR}/neuriplo_engine_parity_fixture")
    add_custom_target(neuriplo_engine_parity_fixture
        COMMAND ${CMAKE_COMMAND} -E make_directory "${NEURIPLO_PARITY_FIXTURE_DIR}"
        COMMAND ${NEURIPLO_PYTHON3_EXECUTABLE}
                "${CMAKE_CURRENT_LIST_DIR}/provision_parity_fixture.py"
                "${NEURIPLO_PARITY_FIXTURE_DIR}"
        COMMENT "Provisioning the NEURIPLO_ENGINE<->ONNX_RUNTIME parity fixture (ResNet-18, opset 18)"
        VERBATIM
        USES_TERMINAL
    )

    add_executable(neuriplo_engine_parity_test ${CMAKE_CURRENT_LIST_DIR}/ParityTest.cpp)
    target_compile_features(neuriplo_engine_parity_test PRIVATE cxx_std_17)
    target_compile_options(neuriplo_engine_parity_test PRIVATE -Wall -Wextra -Wpedantic)

    target_include_directories(neuriplo_engine_parity_test PRIVATE
        ${CMAKE_SOURCE_DIR}/backends/neuriplo-engine/src
        ${CMAKE_SOURCE_DIR}/backends/onnx-runtime/src
        ${CMAKE_SOURCE_DIR}/backends/src
        ${CMAKE_SOURCE_DIR}/include
        ${ONNX_RUNTIME_DIR}/include
        ${GLOG_INCLUDE_DIRS}
    )

    target_link_libraries(neuriplo_engine_parity_test PRIVATE
        neuriplo
        neuriplo_engine
        ${ONNX_RUNTIME_LIBRARY}
        GTest::gtest
        GTest::gtest_main
        ${GLOG_LIBRARIES}
        glog::glog
    )

    add_test(NAME neuriplo_engine_parity COMMAND neuriplo_engine_parity_test)
    set_tests_properties(neuriplo_engine_parity PROPERTIES
        ENVIRONMENT "NEURIPLO_PARITY_FIXTURE=${NEURIPLO_PARITY_FIXTURE_DIR}/resnet18.onnx"
        TIMEOUT 1800
    )

    # -----------------------------------------------------------------------
    # YOLO NEURIPLO_ENGINE <-> ONNX_RUNTIME parity (packet Y4). The same NMS-embedded
    # yolo26n file through both adapters, compared elementwise.
    # -----------------------------------------------------------------------

    # Deterministic fixture provisioning: exports Ultralytics yolo26n with NMS
    # (never to the old hard-coded /workspace/). Always re-runs.
    set(NEURIPLO_YOLO_PARITY_FIXTURE_DIR "${CMAKE_BINARY_DIR}/neuriplo_engine_yolo_parity_fixture")
    add_custom_target(neuriplo_engine_yolo_parity_fixture
        COMMAND ${CMAKE_COMMAND} -E make_directory "${NEURIPLO_YOLO_PARITY_FIXTURE_DIR}"
        COMMAND ${NEURIPLO_PYTHON3_EXECUTABLE}
                "${CMAKE_CURRENT_LIST_DIR}/provision_yolo_fixture.py"
                "${NEURIPLO_YOLO_PARITY_FIXTURE_DIR}"
        COMMENT "Provisioning the YOLO NEURIPLO_ENGINE<->ONNX_RUNTIME parity fixture (yolo26n NMS, opset 18)"
        VERBATIM
        USES_TERMINAL
    )

    add_executable(neuriplo_engine_yolo_parity_test ${CMAKE_CURRENT_LIST_DIR}/YoloParityTest.cpp)
    target_compile_features(neuriplo_engine_yolo_parity_test PRIVATE cxx_std_17)
    target_compile_options(neuriplo_engine_yolo_parity_test PRIVATE -Wall -Wextra -Wpedantic)

    target_include_directories(neuriplo_engine_yolo_parity_test PRIVATE
        ${CMAKE_SOURCE_DIR}/backends/neuriplo-engine/src
        ${CMAKE_SOURCE_DIR}/backends/onnx-runtime/src
        ${CMAKE_SOURCE_DIR}/backends/src
        ${CMAKE_SOURCE_DIR}/include
        ${ONNX_RUNTIME_DIR}/include
        ${GLOG_INCLUDE_DIRS}
    )

    target_link_libraries(neuriplo_engine_yolo_parity_test PRIVATE
        neuriplo
        neuriplo_engine
        ${ONNX_RUNTIME_LIBRARY}
        GTest::gtest
        GTest::gtest_main
        ${GLOG_LIBRARIES}
        glog::glog
    )

    add_test(NAME neuriplo_engine_yolo_parity COMMAND neuriplo_engine_yolo_parity_test)
    set_tests_properties(neuriplo_engine_yolo_parity PROPERTIES
        ENVIRONMENT "NEURIPLO_YOLO_PARITY_FIXTURE=${NEURIPLO_YOLO_PARITY_FIXTURE_DIR}/yolo26n.onnx;NEURIPLO_YOLO_PARITY_INPUT=${NEURIPLO_YOLO_PARITY_FIXTURE_DIR}/yolo26n_input.bin"
        TIMEOUT 1800
    )

endif()

# ---------------------------------------------------------------------------
# Model-zoo parity (neuriplo-tasks operator coverage packet). Registered only
# when a manifest is configured: the models are large local exports, not
# provisioned fixtures.
# ---------------------------------------------------------------------------
if("ONNX_RUNTIME" IN_LIST NEURIPLO_ENABLED_BACKENDS)
    set(NEURIPLO_MODEL_ZOO_MANIFEST "" CACHE FILEPATH
        "Manifest of ONNX models for the NEURIPLO_ENGINE model-zoo parity test")

    add_executable(neuriplo_engine_model_zoo_parity_test ${CMAKE_CURRENT_LIST_DIR}/ModelZooParityTest.cpp)
    target_compile_features(neuriplo_engine_model_zoo_parity_test PRIVATE cxx_std_17)
    target_compile_options(neuriplo_engine_model_zoo_parity_test PRIVATE -Wall -Wextra -Wpedantic)
    target_include_directories(neuriplo_engine_model_zoo_parity_test PRIVATE
        ${CMAKE_SOURCE_DIR}/backends/neuriplo-engine/src
        ${CMAKE_SOURCE_DIR}/backends/onnx-runtime/src
        ${CMAKE_SOURCE_DIR}/backends/src
        ${CMAKE_SOURCE_DIR}/include
        ${ONNX_RUNTIME_DIR}/include
        ${GLOG_INCLUDE_DIRS}
    )
    target_link_libraries(neuriplo_engine_model_zoo_parity_test PRIVATE
        neuriplo
        neuriplo_engine
        ${ONNX_RUNTIME_LIBRARY}
        GTest::gtest
        GTest::gtest_main
        ${GLOG_LIBRARIES}
        glog::glog
    )

    if(NEURIPLO_MODEL_ZOO_MANIFEST)
        add_test(NAME neuriplo_engine_model_zoo_parity COMMAND neuriplo_engine_model_zoo_parity_test)
        set_tests_properties(neuriplo_engine_model_zoo_parity PROPERTIES
            ENVIRONMENT "NEURIPLO_MODEL_ZOO_MANIFEST=${NEURIPLO_MODEL_ZOO_MANIFEST}"
            TIMEOUT 7200
        )
    endif()
endif()
