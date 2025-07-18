# Centralized version management for InferenceEngines library
# This file should be the single source of truth for all inference backend versions

# Inference Backend Versions
set(ONNX_RUNTIME_VERSION "1.19.2" CACHE STRING "ONNX Runtime version")
set(TENSORRT_VERSION "10.7.0.23" CACHE STRING "TensorRT version")
set(LIBTORCH_VERSION "2.0.0" CACHE STRING "LibTorch version")
set(OPENVINO_VERSION "2025.2.0" CACHE STRING "OpenVINO version")
set(TENSORFLOW_VERSION "2.19.0" CACHE STRING "TensorFlow version")

# CUDA Version (for GPU support)
set(CUDA_VERSION "12.6" CACHE STRING "CUDA version for GPU support")

# System Dependencies (minimum versions)
set(OPENCV_MIN_VERSION "4.6.0" CACHE STRING "Minimum OpenCV version")
set(GLOG_MIN_VERSION "0.6.0" CACHE STRING "Minimum glog version")
set(CMAKE_MIN_VERSION "3.10" CACHE STRING "Minimum CMake version")

# Platform-specific paths (with fallbacks)
if(WIN32)
    set(DEFAULT_DEPENDENCY_ROOT "$ENV{USERPROFILE}/dependencies" CACHE PATH "Default dependency installation root")
else()
    set(DEFAULT_DEPENDENCY_ROOT "$ENV{HOME}/dependencies" CACHE PATH "Default dependency installation root")
endif()

# Dependency-specific paths
set(ONNX_RUNTIME_DIR "${DEFAULT_DEPENDENCY_ROOT}/onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION}" CACHE PATH "ONNX Runtime installation directory")
set(TENSORRT_DIR "${DEFAULT_DEPENDENCY_ROOT}/TensorRT-${TENSORRT_VERSION}" CACHE PATH "TensorRT installation directory")
set(LIBTORCH_DIR "${DEFAULT_DEPENDENCY_ROOT}/libtorch" CACHE PATH "LibTorch installation directory")
set(OPENVINO_DIR "${DEFAULT_DEPENDENCY_ROOT}/openvino_${OPENVINO_VERSION}" CACHE PATH "OpenVINO installation directory")

# Version validation functions
function(validate_version_found found_version required_version component_name)
    if(found_version)
        if(found_version VERSION_LESS required_version)
            message(WARNING "${component_name} version ${found_version} is older than required ${required_version}")
            return()
        endif()
        message(STATUS "${component_name} version ${found_version} meets requirements (>= ${required_version})")
    else()
        message(WARNING "${component_name} version not found, but ${required_version} or higher is required")
    endif()
endfunction()

function(validate_version_exact found_version expected_version component_name)
    if(found_version)
        if(NOT found_version VERSION_EQUAL expected_version)
            message(WARNING "${component_name} version ${found_version} differs from expected ${expected_version}")
            return()
        endif()
        message(STATUS "${component_name} version ${found_version} matches expected version")
    else()
        message(WARNING "${component_name} version not found, expected ${expected_version}")
    endif()
endfunction()

# Print version information for debugging
message(STATUS "=== InferenceEngines Dependency Versions ===")
message(STATUS "ONNX Runtime: ${ONNX_RUNTIME_VERSION}")
message(STATUS "TensorRT: ${TENSORRT_VERSION}")
message(STATUS "LibTorch: ${LIBTORCH_VERSION}")
message(STATUS "OpenVINO: ${OPENVINO_VERSION}")
message(STATUS "TensorFlow: ${TENSORFLOW_VERSION}")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "Dependency Root: ${DEFAULT_DEPENDENCY_ROOT}") 