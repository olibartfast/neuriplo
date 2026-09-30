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

if(NEURIPLO_NATIVE_FIXTURE)
    set_tests_properties(engine_loader engine_loader_negative PROPERTIES
        ENVIRONMENT "NEURIPLO_NATIVE_FIXTURE=${NEURIPLO_NATIVE_FIXTURE}")
endif()
