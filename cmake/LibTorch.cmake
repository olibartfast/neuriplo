# LibTorch Configuration

# Set LibTorch directory to the correct location
if(DEFINED DEFAULT_DEPENDENCY_ROOT AND NOT DEFAULT_DEPENDENCY_ROOT STREQUAL "")
    set(_default_libtorch_root "${DEFAULT_DEPENDENCY_ROOT}/libtorch")
else()
    set(_default_libtorch_root "$ENV{HOME}/dependencies/libtorch")
endif()
set(Torch_DIR "${_default_libtorch_root}/share/cmake/Torch" CACHE PATH "Path to libtorch")

# Add LibTorch to CMAKE_PREFIX_PATH to help find_package
list(APPEND CMAKE_PREFIX_PATH "${_default_libtorch_root}")

# Find LibTorch
find_package(Torch REQUIRED)

# LibTorch >= 2.10 headers refuse to compile below C++20 (#error in ATen.h).
# They are included from the registry and the tests as well as the backend
# sources, so raise the standard for every target in a LibTorch build. The
# project default stays C++17 for builds without LibTorch.
if(Torch_VERSION VERSION_GREATER_EQUAL "2.10" AND CMAKE_CXX_STANDARD LESS 20)
    message(STATUS "LibTorch ${Torch_VERSION} requires C++20; raising CMAKE_CXX_STANDARD from ${CMAKE_CXX_STANDARD}")
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()


set(LIBTORCH_SOURCES
    ${INFER_ROOT}/libtorch/src/LibtorchInfer.cpp
    ${INFER_ROOT}/libtorch/src/LibtorchAOTIInfer.cpp
)

# Append LibTorch sources to the main sources
list(APPEND SOURCES ${LIBTORCH_SOURCES})

# Add compile definition to indicate LibTorch usage
add_compile_definitions(USE_LIBTORCH)
