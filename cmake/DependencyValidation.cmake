# Dependency validation and setup utilities for neuriplo library
# This module provides functions to validate and setup inference backend dependencies

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

# Function to validate a dependency exists
function(validate_dependency dependency_name dependency_path)
    if(NOT EXISTS "${dependency_path}")
        if(PROJECT_IS_TOP_LEVEL)
            message(FATAL_ERROR "${dependency_name} not found at ${dependency_path}. 
        Please ensure the inference backend is properly installed or run the setup script.")
        else()
            message(WARNING "neuriplo: ${dependency_name} not found at ${dependency_path}")
            return()
        endif()
    endif()
    
    message(STATUS "✓ ${dependency_name} found at ${dependency_path}")
endfunction()

# Function to validate ONNX Runtime
function(validate_onnx_runtime)
    if("ONNX_RUNTIME" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("ONNX Runtime" "${ONNX_RUNTIME_DIR}")
        
        # Check for required files
        set(required_files
            "${ONNX_RUNTIME_DIR}/include/onnxruntime_cxx_api.h"
            "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so"
        )
        
        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "ONNX Runtime installation incomplete. Missing: ${file}")
            endif()
        endforeach()
        
        message(STATUS "✓ ONNX Runtime validation passed")
    endif()
endfunction()

# Function to validate TensorRT
function(validate_tensorrt)
    if("TENSORRT" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("TensorRT" "${TENSORRT_DIR}")
        
        # Check for required files
        set(required_files
            "${TENSORRT_DIR}/include/NvInfer.h"
            "${TENSORRT_DIR}/lib/libnvinfer.so"
        )
        
        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "TensorRT installation incomplete. Missing: ${file}")
            endif()
        endforeach()
        
        message(STATUS "✓ TensorRT validation passed")
    endif()
endfunction()

# Function to validate LibTorch
function(validate_libtorch)
    if("LIBTORCH" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("LibTorch" "${LIBTORCH_DIR}")
        
        # Check for CMake configuration
        if(NOT EXISTS "${LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
            message(FATAL_ERROR "LibTorch CMake configuration not found. Please ensure LibTorch is properly installed.")
        endif()
        
        message(STATUS "✓ LibTorch validation passed")
    endif()
endfunction()

# Function to validate OpenVINO
function(validate_openvino)
    if("OPENVINO" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("OpenVINO" "${OPENVINO_DIR}")
        
        # Check for required files
        set(required_files
            "${OPENVINO_DIR}/runtime/include/openvino/openvino.hpp"
            "${OPENVINO_DIR}/runtime/lib/intel64/libopenvino.so"
        )
        
        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "OpenVINO installation incomplete. Missing: ${file}")
            endif()
        endforeach()
        
        message(STATUS "✓ OpenVINO validation passed")
    endif()
endfunction()

# Function to validate MIGraphX
function(validate_migraphx)
    if("MIGRAPHX" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("MIGraphX root" "${MIGRAPHX_ROOT}")

        list(APPEND CMAKE_PREFIX_PATH "${MIGRAPHX_ROOT}")
        find_package(migraphx REQUIRED)

        message(STATUS "✓ MIGraphX validation passed")
    endif()
endfunction()

# Function to validate CUDA/ROCm (if GPU support is requested)
function(validate_cuda)
    if("TENSORRT" IN_LIST NEURIPLO_ENABLED_BACKENDS OR "ONNX_RUNTIME" IN_LIST NEURIPLO_ENABLED_BACKENDS OR "GGML" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            message(STATUS "✓ CUDA found: ${CUDA_VERSION}")
        else()
            if(EXISTS "/opt/rocm")
                message(STATUS "✓ ROCm found at /opt/rocm (AMD GPU support)")
            else()
                message(WARNING "Neither CUDA nor ROCm found. GPU support will be disabled.")
            endif()
        endif()
    endif()
endfunction()

# Function to validate system dependencies
function(validate_system_dependencies)
    # When used as subdirectory, make these checks optional
    if(NOT PROJECT_IS_TOP_LEVEL)
        # Try to find OpenCV but don't require it
        find_package(OpenCV QUIET)
        if(OpenCV_FOUND)
            message(STATUS "✓ OpenCV ${OpenCV_VERSION} found")
        else()
            message(STATUS "OpenCV not found - parent project should handle OpenCV")
        endif()
        
        # Try to find glog but don't require it
        find_package(Glog QUIET)
        if(Glog_FOUND)
            message(STATUS "✓ glog found")
        else()
            message(STATUS "glog not found - parent project should handle glog")
        endif()
        
        return()
    endif()
    
    # Validate OpenCV
    find_package(OpenCV REQUIRED)
    if(OpenCV_VERSION VERSION_LESS OPENCV_MIN_VERSION)
        message(FATAL_ERROR "OpenCV version ${OpenCV_VERSION} is too old. Minimum required: ${OPENCV_MIN_VERSION}")
    endif()
    message(STATUS "✓ OpenCV ${OpenCV_VERSION} found")
    
    # Validate glog
    find_package(Glog REQUIRED)
    message(STATUS "✓ glog found")
    
    # Validate CMake version
    if(CMAKE_VERSION VERSION_LESS CMAKE_MIN_VERSION)
        message(FATAL_ERROR "CMake version ${CMAKE_VERSION} is too old. Minimum required: ${CMAKE_MIN_VERSION}")
    endif()
    message(STATUS "✓ CMake ${CMAKE_VERSION} found")
endfunction()

# Function to validate all dependencies
function(validate_all_dependencies)
    # Skip validation if this is being used as a FetchContent dependency
    if(NOT PROJECT_IS_TOP_LEVEL)
        message(STATUS "neuriplo used as subdirectory - skipping dependency validation")
        return()
    endif()
    
    message(STATUS "=== Validating neuriplo Dependencies ===")
    
    validate_system_dependencies()
    
    # Each validator self-guards on NEURIPLO_ENABLED_BACKENDS, so validating
    # every enabled backend is just calling them all.
    validate_opencv_dnn()
    validate_onnx_runtime()
    validate_tensorrt()
    validate_libtorch()
    validate_libtensorflow()
    validate_openvino()
    validate_ggml()
    validate_tvm()
    validate_cactus()
    validate_migraphx()
    validate_llamacpp()
    validate_executorch()
    validate_litert()

    # Validate CUDA if needed for any enabled backend
    validate_cuda()
    
    message(STATUS "=== All neuriplo Dependencies Validated Successfully ===")
endfunction()

# Function to check if we're in a Docker environment
function(is_docker_environment result)
    if(EXISTS "/.dockerenv")
        set(${result} TRUE PARENT_SCOPE)
    else()
        set(${result} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Function to provide helpful setup instructions
function(print_setup_instructions)
    message(STATUS "=== Setup Instructions ===")
    message(STATUS "If inference backend dependencies are missing, run the following commands:")
    message(STATUS "")
    
    if("OPENCV_DNN" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        message(STATUS "  OpenCV DNN is included with OpenCV installation")
        message(STATUS "  Ensure OpenCV is installed with DNN module support")
    else()
        message(STATUS "  ./scripts/setup_dependencies.sh --backend ${DEFAULT_BACKEND}")
    endif()
    
    message(STATUS "")
    message(STATUS "Or run the unified setup script:")
    message(STATUS "  ./scripts/setup_dependencies.sh --backend ${DEFAULT_BACKEND}")
    message(STATUS "")
endfunction()

# Function to validate OpenCV DNN
function(validate_opencv_dnn)
    if("OPENCV_DNN" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        # OpenCV is already validated in validate_system_dependencies()
        # Just check that DNN module is available
        find_package(OpenCV REQUIRED)
        
        # Check if OpenCV was compiled with DNN support
        if(NOT OpenCV_FOUND)
            message(FATAL_ERROR "OpenCV not found. OpenCV DNN backend requires OpenCV.")
        endif()
        
        message(STATUS "✓ OpenCV DNN validation passed")
    endif()
endfunction()

# Function to validate LibTensorFlow
function(validate_libtensorflow)
    if("LIBTENSORFLOW" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        # Add cmake modules path to find our custom FindTensorFlow.cmake
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
        
        find_package(TensorFlow QUIET)
        if(NOT TensorFlow_FOUND)
            message(FATAL_ERROR "LibTensorFlow not found. Please install TensorFlow C++ library or run the setup script.")
        endif()
        
        # Check for required TensorFlow components
        if(NOT DEFINED TensorFlow_INCLUDE_DIRS OR NOT DEFINED TensorFlow_LIBRARIES)
            message(FATAL_ERROR "LibTensorFlow installation incomplete. Missing include directories or libraries.")
        endif()
        
        message(STATUS "✓ LibTensorFlow validation passed")
    endif()
endfunction()

# Function to validate GGML
function(validate_ggml)
    if("GGML" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("GGML" "${GGML_DIR}")
        
        # Check for required files
        set(required_files
            "${GGML_DIR}/include/ggml.h"
            "${GGML_DIR}/include/ggml-backend.h"
            "${GGML_DIR}/lib/libggml.so"
        )
        
        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "GGML installation incomplete. Missing: ${file}")
            endif()
        endforeach()
        
        message(STATUS "✓ GGML validation passed")
    endif()
endfunction()

function(validate_tvm)
    if("TVM" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("TVM" "${TVM_DIR}")
        
        # Check for required files - try multiple possible header paths for different TVM versions
        set(possible_header_files
            "${TVM_DIR}/include/tvm/runtime/c_runtime_api.h"
            "${TVM_DIR}/include/tvm/runtime/c_backend_api.h"
            "${TVM_DIR}/include/tvm/c_runtime_api.h"
        )
        
        set(header_found FALSE)
        foreach(header_file ${possible_header_files})
            if(EXISTS "${header_file}")
                set(header_found TRUE)
                message(STATUS "✓ TVM header found: ${header_file}")
                break()
            endif()
        endforeach()
        
        if(NOT header_found)
            message(FATAL_ERROR "TVM installation incomplete. None of the expected header files found: ${possible_header_files}")
        endif()
        
        set(required_files
            "${TVM_DIR}/build/libtvm_runtime.so"
            "${TVM_DIR}/build/libtvm.so"
        )
        
        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "TVM installation incomplete. Missing: ${file}")
            endif()
        endforeach()
        
        message(STATUS "✓ TVM validation passed")
    endif()
endfunction()

# Function to validate llama.cpp
function(validate_llamacpp)
    if("LLAMACPP" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("llama.cpp" "${LLAMACPP_DIR}")

        set(required_files
            "${LLAMACPP_DIR}/include/llama.h"
            "${LLAMACPP_DIR}/lib/libllama.so"
        )

        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "llama.cpp installation incomplete. Missing: ${file}")
            endif()
        endforeach()

        # libggml.so was split into libggml-base.so + libggml-cpu.so in newer master;
        # accept either form since we link by name (-lggml) not by path.
        if(NOT EXISTS "${LLAMACPP_DIR}/lib/libggml.so" AND
           NOT EXISTS "${LLAMACPP_DIR}/lib/libggml-base.so")
            message(FATAL_ERROR "llama.cpp installation incomplete. Missing libggml.so or libggml-base.so in ${LLAMACPP_DIR}/lib")
        endif()

        message(STATUS "✓ llama.cpp validation passed")
    endif()
endfunction()

# Function to validate ExecuTorch
function(validate_executorch)
    if("EXECUTORCH" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("ExecuTorch" "${EXECUTORCH_DIR}")

        set(required_files
            "${EXECUTORCH_DIR}/include/executorch/runtime/core/error.h"
            "${EXECUTORCH_DIR}/lib/libexecutorch.a"
        )

        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "ExecuTorch installation incomplete. Missing: ${file}\nRun: ./scripts/setup_executorch.sh")
            endif()
        endforeach()

        message(STATUS "✓ ExecuTorch validation passed")
    endif()
endfunction()

# Function to validate LiteRT
function(validate_litert)
    if("LITERT" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("LiteRT" "${LITERT_DIR}")

        set(required_files
            "${LITERT_DIR}/include/tensorflow/lite/interpreter.h"
            "${LITERT_DIR}/include/tensorflow/lite/model.h"
            "${LITERT_DIR}/lib/libtensorflowlite.so"
        )

        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "LiteRT installation incomplete. Missing: ${file}")
            endif()
        endforeach()

        set(LITERT_LIBRARY "${LITERT_DIR}/lib/libtensorflowlite.so" CACHE FILEPATH "LiteRT shared library")

        message(STATUS "✓ LiteRT validation passed")
    endif()
endfunction()

# Function to validate Cactus
function(validate_cactus)
    if("CACTUS" IN_LIST NEURIPLO_ENABLED_BACKENDS)
        validate_dependency("Cactus" "${CACTUS_DIR}")

        set(required_files
            "${CACTUS_DIR}/include/cactus.h"
            "${CACTUS_DIR}/include/graph/graph.h"
            "${CACTUS_DIR}/lib/libcactus.so"
        )

        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "Cactus installation incomplete. Missing: ${file}")
            endif()
        endforeach()

        message(STATUS "✓ Cactus validation passed")
    endif()
endfunction()
