# ONNX Runtime Configuration
# Uses centralized version management from cmake/versions.cmake

# ONNX Runtime configuration using centralized version management
message(STATUS "ONNX Runtime version: ${ONNX_RUNTIME_VERSION}")
message(STATUS "ONNX Runtime directory: ${ONNX_RUNTIME_DIR}")

option(ORT_ENABLE_TENSORRT_EP "Enable explicit ONNX Runtime TensorRT Execution Provider selection" OFF)
option(ORT_ENABLE_OPENVINO_EP "Enable explicit ONNX Runtime OpenVINO Execution Provider selection" OFF)
option(ORT_ENABLE_MIGRAPHX_EP "Enable explicit ONNX Runtime MIGraphX Execution Provider selection" OFF)
option(ORT_ENABLE_QNN_EP "Enable explicit ONNX Runtime Qualcomm QNN Execution Provider selection" OFF)
option(ORT_ENABLE_XNNPACK_EP "Enable explicit ONNX Runtime XNNPACK Execution Provider selection" OFF)
option(ORT_ENABLE_CANN_EP "Enable explicit ONNX Runtime Huawei CANN Execution Provider selection" OFF)
option(ORT_ENABLE_VITISAI_EP "Enable explicit ONNX Runtime Xilinx Vitis AI Execution Provider selection" OFF)

# Check for CUDA support
find_package(CUDA QUIET)
if (CUDA_FOUND)
    message(STATUS "✓ CUDA found: ${CUDA_VERSION}")
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
else ()
    message(STATUS "CUDA not found.")
endif()

# Check for ROCm support
if (EXISTS "/opt/rocm")
    message(STATUS "✓ ROCm found at /opt/rocm")
else ()
    message(STATUS "ROCm not found.")
endif()

if (NOT CUDA_FOUND AND NOT EXISTS "/opt/rocm")
    message(WARNING "Neither CUDA nor ROCm found. GPU support will be disabled for ONNX Runtime.")
endif()

# Define ONNX Runtime-specific source files
set(ONNX_RUNTIME_SOURCES
    ${INFER_ROOT}/onnx-runtime/src/ORTInfer.cpp
    # Add more ONNX Runtime source files here if needed
)

# Append ONNX Runtime sources to the main sources
list(APPEND SOURCES ${ONNX_RUNTIME_SOURCES})

# Add compile definition to indicate ONNX Runtime usage
add_compile_definitions(USE_ONNX_RUNTIME)

if(ORT_ENABLE_TENSORRT_EP)
    add_compile_definitions(ORT_ENABLE_TENSORRT_EP)
endif()

if(ORT_ENABLE_OPENVINO_EP)
    add_compile_definitions(ORT_ENABLE_OPENVINO_EP)
endif()

if(ORT_ENABLE_MIGRAPHX_EP)
    add_compile_definitions(ORT_ENABLE_MIGRAPHX_EP)
endif()

if(ORT_ENABLE_QNN_EP)
    add_compile_definitions(ORT_ENABLE_QNN_EP)
endif()

if(ORT_ENABLE_XNNPACK_EP)
    add_compile_definitions(ORT_ENABLE_XNNPACK_EP)
endif()

if(ORT_ENABLE_CANN_EP)
    add_compile_definitions(ORT_ENABLE_CANN_EP)
endif()

if(ORT_ENABLE_VITISAI_EP)
    add_compile_definitions(ORT_ENABLE_VITISAI_EP)
endif()

# Note: Version management is now handled centrally in cmake/versions.cmake
