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


set(LIBTORCH_SOURCES
    ${INFER_ROOT}/libtorch/src/LibtorchInfer.cpp
    # Add more LibTorch source files here if needed
)

# Append LibTorch sources to the main sources
list(APPEND SOURCES ${LIBTORCH_SOURCES})

# Add compile definition to indicate LibTorch usage
add_compile_definitions(USE_LIBTORCH)
