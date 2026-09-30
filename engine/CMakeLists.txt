# First-party inference engine (CPU reference spine).
#
# This directory is deliberately self-contained: it must not link or include
# anything from the backend abstraction layer (include/neuriplo/,
# backends/src/, src/). A configure-time guard below enforces that boundary.

cmake_minimum_required(VERSION 3.10)

# Standalone configure support: when consumed via add_subdirectory() from the
# top-level project, the parent project() and CXX standard already apply.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(neuriplo_engine LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# --- Abstraction-boundary guard ----------------------------------------------
# Engine translation units must not include backend abstraction headers. The
# forbidden patterns are named here in CMake (allowed); the C++ sources
# themselves must stay free of them.
set(ENGINE_FORBIDDEN_PATTERNS
    "include/neuriplo"
    "InferenceInterface"
)
file(GLOB_RECURSE ENGINE_TUS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
foreach(engine_tu IN LISTS ENGINE_TUS)
    file(READ "${engine_tu}" engine_tu_content)
    foreach(pattern IN LISTS ENGINE_FORBIDDEN_PATTERNS)
        if(engine_tu_content MATCHES "${pattern}")
            message(FATAL_ERROR
                "engine/ boundary violation: ${engine_tu} references "
                "'${pattern}'. The engine must not depend on the backend "
                "abstraction layer.")
        endif()
    endforeach()
endforeach()

add_library(neuriplo_engine STATIC
    src/Engine.cpp
    src/WireReader.cpp
    src/ModelLoader.cpp
    src/ShapeInference.cpp
    src/MemoryPlanner.cpp
    src/CpuDevice.cpp
    src/kernels/CpuKernels.cpp
    src/kernels/Relu.cpp
    src/kernels/Add.cpp
    src/kernels/Reshape.cpp
    src/kernels/ReduceMean.cpp
    src/kernels/Gemm.cpp
    src/kernels/MatMul.cpp
    src/kernels/Conv.cpp
    src/kernels/MaxPool.cpp)
target_compile_features(neuriplo_engine PUBLIC cxx_std_17)
target_include_directories(neuriplo_engine
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>")

# Engine tests; GoogleTest stays scoped to engine/test/. The engine directory
# is added during backend selection, before the top-level test setup runs, so
# testing is enabled here for the tests below to register.
if(BUILD_INFERENCE_ENGINE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
