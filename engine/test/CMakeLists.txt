# Engine model-loader tests.
#
# Wired via engine/CMakeLists.txt (a guarded add_subdirectory), not the
# backend registry: the engine is standalone-capable and must not depend on
# backends/** or the top-level test harness. GoogleTest is scoped to this
# directory only, and the engine links nothing but the standard library.

find_package(GTest REQUIRED)

add_executable(engine_loader_test LoaderTest.cpp)
target_compile_features(engine_loader_test PRIVATE cxx_std_17)
target_compile_options(engine_loader_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(engine_loader_test PRIVATE
    neuriplo_engine
    GTest::gtest
    GTest::gtest_main)

# Fixture resolvable via env var or CMake cache variable of the same name.
# When unset, fall back to the well-known local export location if it exists;
# the tests themselves ASSERT (fail, never skip) when the fixture is absent.
set(NEURIPLO_NATIVE_FIXTURE "" CACHE PATH
    "Path to the opset-18 ResNet-18 ONNX fixture for the engine loader tests")
if(NOT NEURIPLO_NATIVE_FIXTURE AND EXISTS
    "/tmp/opencode/resnet18_export/resnet18.onnx")
    set(NEURIPLO_NATIVE_FIXTURE
        "/tmp/opencode/resnet18_export/resnet18.onnx")
endif()

add_test(NAME engine_loader
    COMMAND engine_loader_test --gtest_filter=EngineLoader.*)
add_test(NAME engine_loader_negative
    COMMAND engine_loader_test --gtest_filter=EngineLoaderNegative.*)

add_executable(engine_shapes_test ShapeInferenceTest.cpp)
target_compile_features(engine_shapes_test PRIVATE cxx_std_17)
target_compile_options(engine_shapes_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(engine_shapes_test PRIVATE
    neuriplo_engine
    GTest::gtest
    GTest::gtest_main)

add_test(NAME engine_shapes
    COMMAND engine_shapes_test --gtest_filter=EngineShapes.*)
add_test(NAME engine_shapes_negative
    COMMAND engine_shapes_test --gtest_filter=EngineShapesNegative.*)

add_executable(engine_plan_test MemoryPlannerTest.cpp)
target_compile_features(engine_plan_test PRIVATE cxx_std_17)
target_compile_options(engine_plan_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(engine_plan_test PRIVATE
    neuriplo_engine
    GTest::gtest
    GTest::gtest_main)

add_test(NAME engine_plan
    COMMAND engine_plan_test --gtest_filter=EnginePlan.*)
add_test(NAME engine_plan_negative
    COMMAND engine_plan_test --gtest_filter=EnginePlanNegative.*)

add_executable(engine_device_test DeviceTest.cpp)
target_compile_features(engine_device_test PRIVATE cxx_std_17)
target_compile_options(engine_device_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(engine_device_test PRIVATE
    neuriplo_engine
    GTest::gtest
    GTest::gtest_main)

add_test(NAME engine_device
    COMMAND engine_device_test --gtest_filter=EngineDevice.*)

add_executable(engine_kernels_test KernelsTest.cpp)
target_compile_features(engine_kernels_test PRIVATE cxx_std_17)
target_compile_options(engine_kernels_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(engine_kernels_test PRIVATE
    neuriplo_engine
    GTest::gtest
    GTest::gtest_main)

add_test(NAME engine_kernels
    COMMAND engine_kernels_test --gtest_filter=EngineKernels.*)
add_test(NAME engine_kernels_negative
    COMMAND engine_kernels_test --gtest_filter=EngineKernelsNegative.*)

if(NEURIPLO_NATIVE_FIXTURE)
    set_tests_properties(engine_loader engine_loader_negative
        engine_shapes engine_shapes_negative
        engine_plan engine_plan_negative PROPERTIES
        ENVIRONMENT "NEURIPLO_NATIVE_FIXTURE=${NEURIPLO_NATIVE_FIXTURE}")
endif()
