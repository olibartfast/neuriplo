# TensorRT Configuration
# Uses centralized version management from cmake/versions.cmake

# TensorRT configuration using centralized version management
message(STATUS "TensorRT version: ${TENSORRT_VERSION}")
message(STATUS "TensorRT directory: ${TENSORRT_DIR}")

# Find CUDA (required for TensorRT)
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA is required for TensorRT backend but not found.")
endif()

# Query GPU compute capabilities
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_COMPUTE_CAP
    RESULT_VARIABLE GPU_COMPUTE_CAP_RESULT
)

if (GPU_COMPUTE_CAP_RESULT EQUAL 0)
    # Split the GPU compute capabilities into a list
    string(REPLACE "\n" ";" GPU_COMPUTE_CAP_LIST ${GPU_COMPUTE_CAP})

    foreach(GPU_CAP ${GPU_COMPUTE_CAP_LIST})
        string(STRIP ${GPU_CAP} GPU_CAP)  # Remove leading/trailing whitespace
        message(STATUS "GPU Compute Capability: ${GPU_CAP}")

        # Extract the major and minor compute capability values
        string(REGEX REPLACE "\\." ";" COMP_CAP_LIST ${GPU_CAP})
        list(GET COMP_CAP_LIST 0 COMPUTE_CAP_MAJOR)
        list(GET COMP_CAP_LIST 1 COMPUTE_CAP_MINOR)

        # Set CUDA flags based on the detected compute capability for each GPU
        set(CUDA_COMPUTE "compute_${COMPUTE_CAP_MAJOR}${COMPUTE_CAP_MINOR}")
        set(CUDA_SM "sm_${COMPUTE_CAP_MAJOR}${COMPUTE_CAP_MINOR}")
        message(STATUS "Setting -gencode;arch=${CUDA_COMPUTE};code=${CUDA_SM} for GPU ${GPU_CAP}")

        # Set CUDA flags for release for each GPU
        set(CUDA_NVCC_FLAGS_RELEASE_${GPU_CAP} ${CUDA_NVCC_FLAGS_RELEASE};-O3;-gencode;arch=${CUDA_COMPUTE};code=${CUDA_SM})

        # Set CUDA flags for debug for each GPU
        set(CUDA_NVCC_FLAGS_DEBUG_${GPU_CAP} ${CUDA_NVCC_FLAGS_DEBUG};-g;-G;-gencode;arch=${CUDA_COMPUTE};code=${CUDA_SM})
    endforeach()
else()
    message(WARNING "Failed to query GPU compute capability. Using default CUDA flags.")
endif()

# Define TensorRT-specific source files
set(TENSORRT_SOURCES
    ${INFER_ROOT}/tensorrt/src/TRTInfer.cpp
    # Add more TensorRT source files here if needed
)
list(APPEND SOURCES ${TENSORRT_SOURCES})

# Add compile definition to indicate TensorRT usage
add_compile_definitions(USE_TENSORRT)

# Note: Version management is now handled centrally in cmake/versions.cmake
