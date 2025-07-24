# Centralized version management for InferenceEngines library
# This file reads versions from versions.env to maintain consistency

# Function to read versions from versions.env file
function(read_versions_from_env)
    set(VERSIONS_ENV_FILE "${CMAKE_CURRENT_SOURCE_DIR}/versions.env")
    
    if(NOT EXISTS ${VERSIONS_ENV_FILE})
        message(FATAL_ERROR "versions.env file not found at ${VERSIONS_ENV_FILE}")
    endif()
    
    # Read the file and parse each line
    file(READ ${VERSIONS_ENV_FILE} VERSIONS_CONTENT)
    string(REPLACE "\n" ";" VERSIONS_LINES "${VERSIONS_CONTENT}")
    
    foreach(LINE ${VERSIONS_LINES})
        # Skip empty lines and comments
        if(LINE AND NOT LINE MATCHES "^#")
            # Extract variable name and value
            string(REGEX MATCH "^([A-Z_]+)=(.+)$" MATCH "${LINE}")
            if(MATCH)
                set(VAR_NAME "${CMAKE_MATCH_1}")
                set(VAR_VALUE "${CMAKE_MATCH_2}")
                # Remove quotes if present
                string(REGEX REPLACE "^\"(.*)\"$" "\\1" VAR_VALUE "${VAR_VALUE}")
                # Set the variable
                set(${VAR_NAME} "${VAR_VALUE}" PARENT_SCOPE)
            endif()
        endif()
    endforeach()
endfunction()

# Read versions from versions.env
read_versions_from_env()

# Inference Backend Versions (from versions.env)
set(ONNX_RUNTIME_VERSION "${ONNX_RUNTIME_VERSION}" CACHE STRING "ONNX Runtime version")
set(TENSORRT_VERSION "${TENSORRT_VERSION}" CACHE STRING "TensorRT version")
set(LIBTORCH_VERSION "${PYTORCH_VERSION}" CACHE STRING "LibTorch version")
set(OPENVINO_VERSION "${OPENVINO_VERSION}" CACHE STRING "OpenVINO version")
set(TENSORFLOW_VERSION "${TENSORFLOW_VERSION}" CACHE STRING "TensorFlow version")

# CUDA Version (for GPU support)
set(CUDA_VERSION "${CUDA_VERSION}" CACHE STRING "CUDA version for GPU support")

# System Dependencies (minimum versions)
set(OPENCV_MIN_VERSION "${OPENCV_VERSION}" CACHE STRING "Minimum OpenCV version")
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
message(STATUS "=== InferenceEngines Dependency Versions (from versions.env) ===")
message(STATUS "ONNX Runtime: ${ONNX_RUNTIME_VERSION}")
message(STATUS "TensorRT: ${TENSORRT_VERSION}")
message(STATUS "LibTorch: ${LIBTORCH_VERSION}")
message(STATUS "OpenVINO: ${OPENVINO_VERSION}")
message(STATUS "TensorFlow: ${TENSORFLOW_VERSION}")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "OpenCV: ${OPENCV_MIN_VERSION}")
message(STATUS "Dependency Root: ${DEFAULT_DEPENDENCY_ROOT}") 