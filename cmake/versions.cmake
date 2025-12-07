# ==============================================================================
# Centralized Version Management for neuriplo
# ==============================================================================
#
# This file works with versions.env to maintain backend versions:
#
#   1. versions.env         → Defines version numbers (TVM_VERSION=0.18.0)  
#   2. cmake/versions.cmake → Reads versions.env and validates consistency (this file)
#
# WORKFLOW:
# ---------
# 1. versions.env defines the version variables (e.g., TVM_VERSION=0.18.0)
# 2. This file reads versions.env and makes variables available to CMake
# 3. This file validates that every backend has a corresponding version variable
#
# ADDING A NEW BACKEND VERSION:
# ------------------------------
# 1. Add version to versions.env:
#      NEW_BACKEND_VERSION=1.0.0
#
# 2. Add cache variable here (after read_versions_from_env()):
#      set(NEW_BACKEND_VERSION "${NEW_BACKEND_VERSION}" CACHE STRING "New Backend version")
#
# 3. Add to BACKEND_VERSION_MAPPING in this file:
#      Add "NEW_BACKEND:NEW_BACKEND_VERSION" to the mapping list
#
# ==============================================================================

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
set(GGML_VERSION "${GGML_VERSION}" CACHE STRING "GGML version")
set(TVM_VERSION "${TVM_VERSION}" CACHE STRING "TVM version")

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
set(GGML_DIR "${DEFAULT_DEPENDENCY_ROOT}/ggml" CACHE PATH "GGML installation directory")
set(TVM_DIR "${DEFAULT_DEPENDENCY_ROOT}/tvm" CACHE PATH "TVM installation directory")

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

# Function to validate backend-version consistency
# Validates that all backends have corresponding versions in versions.env
function(validate_backend_versions)
    message(STATUS "=== Validating Backend-Version Consistency ===")

    # Define backend to version variable mapping directly
    set(BACKEND_VERSION_MAPPING
        "OPENCV_DNN:OPENCV_VERSION"
        "ONNX_RUNTIME:ONNX_RUNTIME_VERSION" 
        "LIBTORCH:PYTORCH_VERSION"
        "LIBTENSORFLOW:TENSORFLOW_VERSION"
        "TENSORRT:TENSORRT_VERSION"
        "OPENVINO:OPENVINO_VERSION"
        "GGML:GGML_VERSION"
        "TVM:TVM_VERSION"
    )

    set(VALIDATION_FAILED FALSE)
    set(MISSING_VERSIONS "")

    foreach(MAPPING ${BACKEND_VERSION_MAPPING})
        # Split mapping by colon
        string(REPLACE ":" ";" MAPPING_PARTS "${MAPPING}")
        list(GET MAPPING_PARTS 0 BACKEND_NAME)
        list(GET MAPPING_PARTS 1 VERSION_VAR_NAME)

        # Get the version value by variable name
        if(VERSION_VAR_NAME STREQUAL "OPENCV_VERSION")
            set(VERSION_VAR "${OPENCV_MIN_VERSION}")
        elseif(VERSION_VAR_NAME STREQUAL "ONNX_RUNTIME_VERSION")
            set(VERSION_VAR "${ONNX_RUNTIME_VERSION}")
        elseif(VERSION_VAR_NAME STREQUAL "PYTORCH_VERSION")
            set(VERSION_VAR "${LIBTORCH_VERSION}")
        elseif(VERSION_VAR_NAME STREQUAL "TENSORFLOW_VERSION")
            set(VERSION_VAR "${TENSORFLOW_VERSION}")
        elseif(VERSION_VAR_NAME STREQUAL "TENSORRT_VERSION")
            set(VERSION_VAR "${TENSORRT_VERSION}")
        elseif(VERSION_VAR_NAME STREQUAL "OPENVINO_VERSION")
            set(VERSION_VAR "${OPENVINO_VERSION}")
        elseif(VERSION_VAR_NAME STREQUAL "GGML_VERSION")
            set(VERSION_VAR "${GGML_VERSION}")
        elseif(VERSION_VAR_NAME STREQUAL "TVM_VERSION")
            set(VERSION_VAR "${TVM_VERSION}")
        else()
            set(VERSION_VAR "")
        endif()

        # Check if version is defined
        if(NOT VERSION_VAR OR VERSION_VAR STREQUAL "")
            message(WARNING "Backend '${BACKEND_NAME}' is missing version variable '${VERSION_VAR_NAME}' in versions.env")
            list(APPEND MISSING_VERSIONS "${BACKEND_NAME} (${VERSION_VAR_NAME})")
            set(VALIDATION_FAILED TRUE)
        else()
            message(STATUS "  ✓ ${BACKEND_NAME} -> ${VERSION_VAR_NAME} = ${VERSION_VAR}")
        endif()
    endforeach()

    if(VALIDATION_FAILED)
        message(STATUS "")
        message(WARNING "=== Backend-Version Validation FAILED ===")
        message(WARNING "Missing version variables in versions.env:")
        foreach(MISSING ${MISSING_VERSIONS})
            message(WARNING "  - ${MISSING}")
        endforeach()
        message(WARNING "")
        message(WARNING "Please add the missing version variables to versions.env")
    else()
        message(STATUS "=== Backend-Version Validation PASSED ===")
    endif()

    set(VALIDATION_FAILED ${VALIDATION_FAILED} PARENT_SCOPE)
endfunction()

# Print version information for debugging
message(STATUS "=== neuriplo Dependency Versions (from versions.env) ===")
message(STATUS "ONNX Runtime: ${ONNX_RUNTIME_VERSION}")
message(STATUS "TensorRT: ${TENSORRT_VERSION}")
message(STATUS "LibTorch: ${LIBTORCH_VERSION}")
message(STATUS "OpenVINO: ${OPENVINO_VERSION}")
message(STATUS "TensorFlow: ${TENSORFLOW_VERSION}")
message(STATUS "GGML: ${GGML_VERSION}")
message(STATUS "TVM: ${TVM_VERSION}")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "OpenCV: ${OPENCV_MIN_VERSION}")
message(STATUS "Dependency Root: ${DEFAULT_DEPENDENCY_ROOT}") 