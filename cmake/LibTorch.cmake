# LibTorch Configuration

# Set LibTorch directory to the correct location
set(Torch_DIR $ENV{HOME}/dependencies/libtorch/share/cmake/Torch/ CACHE PATH "Path to libtorch")

# Add LibTorch to CMAKE_PREFIX_PATH to help find_package
list(APPEND CMAKE_PREFIX_PATH $ENV{HOME}/dependencies/libtorch)

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
