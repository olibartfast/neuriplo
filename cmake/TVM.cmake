# TVM Configuration
# Uses centralized version management from cmake/versions.cmake

# TVM configuration using centralized version management
message(STATUS "TVM version: ${TVM_VERSION}")

# Set TVM directory
if(NOT DEFINED TVM_DIR)
    set(TVM_DIR "$ENV{HOME}/dependencies/tvm" CACHE PATH "Path to TVM installation")
endif()

message(STATUS "TVM directory: ${TVM_DIR}")

# Check if TVM directory exists
if(NOT EXISTS ${TVM_DIR})
    message(WARNING "TVM directory not found at ${TVM_DIR}")
    message(WARNING "Please install TVM or set TVM_DIR to the correct location")
endif()

# Check for CUDA support
find_package(CUDA QUIET)
if (CUDA_FOUND)
    message(STATUS "✓ CUDA found: ${CUDA_VERSION}")
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
else ()
    message(WARNING "CUDA not found. GPU support will be disabled for TVM.")
endif()

# Define TVM-specific source files
set(TVM_SOURCES
    ${INFER_ROOT}/tvm/src/TVMInfer.cpp
    # Add more TVM source files here if needed
)

# Append TVM sources to the main sources
list(APPEND SOURCES ${TVM_SOURCES})

# Add compile definition to indicate TVM usage
add_compile_definitions(USE_TVM)

# Note: Version management is now handled centrally in cmake/versions.cmake
