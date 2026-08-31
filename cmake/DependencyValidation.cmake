# Dependency validation and setup utilities for neuriplo library
# This module provides functions to validate and setup inference backend dependencies

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

function(require_any_existing_path dependency_name)
    foreach(candidate IN LISTS ARGN)
        if(EXISTS "${candidate}")
            message(STATUS "✓ ${dependency_name} found at ${candidate}")
            return()
        endif()
    endforeach()

    if(PROJECT_IS_TOP_LEVEL)
        string(REPLACE ";" "\n  " formatted_candidates "${ARGN}")
        message(FATAL_ERROR "${dependency_name} installation incomplete. Checked:\n  ${formatted_candidates}")
    else()
        message(WARNING "neuriplo: ${dependency_name} was not found in the expected locations")
    endif()
endfunction()

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
    if("ONNX_RUNTIME" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
        validate_dependency("ONNX Runtime" "${ONNX_RUNTIME_DIR}")
        
        # Check for required files
        set(required_files
            "${ONNX_RUNTIME_DIR}/include/onnxruntime_cxx_api.h"
        )

        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "ONNX Runtime installation incomplete. Missing: ${file}")
            endif()
        endforeach()

        require_any_existing_path(
            "ONNX Runtime library"
            "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so"
            "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.dylib"
            "${ONNX_RUNTIME_DIR}/lib/onnxruntime.lib"
            "${ONNX_RUNTIME_DIR}/lib/onnxruntime.dll"
            "${ONNX_RUNTIME_DIR}/lib/Release/onnxruntime.lib"
            "${ONNX_RUNTIME_DIR}/lib/Release/onnxruntime.dll"
            "${ONNX_RUNTIME_DIR}/bin/Release/onnxruntime.dll"
        )
        
        read_version_from_file(detected_version "${ONNX_RUNTIME_DIR}/VERSION_NUMBER")
        report_version_drift("ONNX Runtime" "${ONNX_RUNTIME_VERSION}" "${detected_version}"
            "${ONNX_RUNTIME_DIR}" "Re-run ./scripts/setup_onnx_runtime.sh to install the declared version.")

        message(STATUS "✓ ONNX Runtime validation passed")
    endif()
endfunction()

# --- Installed-version verification -------------------------------------------
# A dependency directory's name is not proof of what is inside it: several of the
# *_DIR paths carry no version at all, and even a versioned path can be pointed
# elsewhere with -D<DEP>_DIR. Read the version out of the installation and say so
# when it disagrees with versions.env, rather than silently building against
# whatever happens to be on disk.

# Pull the first "<major>.<minor>.<patch>" out of a version stamp file.
function(read_version_from_file out_var file_path)
    set(${out_var} "" PARENT_SCOPE)
    if(NOT EXISTS "${file_path}")
        return()
    endif()
    file(READ "${file_path}" contents)
    if(contents MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endif()
endfunction()

# Compare an installation against its pin. `detected` empty means the version
# could not be read, which is reported but is not itself an error.
function(report_version_drift dependency_name declared detected location remedy)
    if(NOT detected)
        message(STATUS "  (${dependency_name} version could not be read from ${location}; "
                       "declared ${declared})")
        return()
    endif()
    if(detected VERSION_EQUAL declared)
        return()
    endif()
    message(WARNING
        "${dependency_name} version drift: versions.env declares ${declared} but ${location} "
        "contains ${detected}. The build will use ${detected}. ${remedy}")
endfunction()

# The source-built backends ship no version file of their own, so their setup
# scripts write one (scripts/lib/version_stamp.sh). Read it back.
function(read_stamped_version out_var install_dir)
    set(${out_var} "" PARENT_SCOPE)
    set(stamp "${install_dir}/neuriplo-version.txt")
    if(NOT EXISTS "${stamp}")
        return()
    endif()
    file(STRINGS "${stamp}" stamp_lines LIMIT_COUNT 1)
    if(NOT stamp_lines)
        return()
    endif()
    list(GET stamp_lines 0 stamped)
    string(STRIP "${stamped}" stamped)
    set(${out_var} "${stamped}" PARENT_SCOPE)
endfunction()

# Same intent as report_version_drift(), but for pins that are git refs rather
# than version numbers -- v0.11.0, v1.14, b9085. VERSION_EQUAL parses a leading
# non-digit as nothing, so it reports "v1.14" VERSION_EQUAL "v1.15" and "b9085"
# VERSION_EQUAL "b9086" as true -- it would pass every drift these pins can have.
# Compare them as the strings they are.
function(report_tag_drift dependency_name declared detected location remedy)
    if(NOT detected)
        message(STATUS "  (${dependency_name} at ${location} carries no version stamp; "
                       "declared ${declared}. Re-run its setup script to record one.)")
        return()
    endif()
    if(detected STREQUAL declared)
        return()
    endif()
    message(WARNING
        "${dependency_name} version drift: versions.env declares ${declared} but ${location} "
        "was built from ${detected}. The build will use ${detected}. ${remedy}")
endfunction()

# Read the actual TensorRT version from the installed headers. The declared
# TENSORRT_VERSION only names a directory, so without this a build silently
# links against whatever runtime happens to be on disk.
function(detect_tensorrt_version out_var include_dir)
    set(${out_var} "" PARENT_SCOPE)

    set(version_header "${include_dir}/NvInferVersion.h")
    if(NOT EXISTS "${version_header}")
        return()
    endif()

    file(READ "${version_header}" version_header_contents)

    set(detected "")
    foreach(component MAJOR MINOR PATCH BUILD)
        # TensorRT 10.x indirects through TRT_<component>_ENTERPRISE; earlier
        # releases define NV_TENSORRT_<component> as a literal.
        if(version_header_contents MATCHES "define[ \t]+TRT_${component}_ENTERPRISE[ \t]+([0-9]+)")
            set(component_value "${CMAKE_MATCH_1}")
        elseif(version_header_contents MATCHES "define[ \t]+NV_TENSORRT_${component}[ \t]+([0-9]+)")
            set(component_value "${CMAKE_MATCH_1}")
        else()
            return()
        endif()

        if(detected STREQUAL "")
            set(detected "${component_value}")
        else()
            set(detected "${detected}.${component_value}")
        endif()
    endforeach()

    set(${out_var} "${detected}" PARENT_SCOPE)
endfunction()

# Function to validate TensorRT
function(validate_tensorrt)
    if("TENSORRT" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
        validate_dependency("TensorRT" "${TENSORRT_DIR}")
        
        # Check for required files
        set(required_files
            "${TENSORRT_DIR}/include/NvInfer.h"
        )

        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "TensorRT installation incomplete. Missing: ${file}")
            endif()
        endforeach()

        require_any_existing_path(
            "TensorRT library"
            "${TENSORRT_DIR}/lib/libnvinfer.so"
            "${TENSORRT_DIR}/lib/libnvinfer.dylib"
            "${TENSORRT_DIR}/lib/nvinfer.lib"
            "${TENSORRT_DIR}/lib/nvinfer.dll"
            "${TENSORRT_DIR}/lib/Release/nvinfer.lib"
            "${TENSORRT_DIR}/lib/Release/nvinfer.dll"
        )
        
        detect_tensorrt_version(detected_version "${TENSORRT_DIR}/include")
        if(detected_version)
            set(TENSORRT_ACTUAL_VERSION "${detected_version}" CACHE INTERNAL
                "TensorRT version detected from the installed headers")
        endif()
        report_version_drift("TensorRT" "${TENSORRT_VERSION}" "${detected_version}" "${TENSORRT_DIR}"
            "Re-run ./scripts/setup_tensorrt.sh to install the declared version, or point -DTENSORRT_DIR at it.")

        message(STATUS "✓ TensorRT validation passed")
    endif()
endfunction()

# Function to validate LibTorch
function(validate_libtorch)
    if("LIBTORCH" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
        validate_dependency("LibTorch" "${LIBTORCH_DIR}")
        
        # Check for CMake configuration
        if(NOT EXISTS "${LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
            message(FATAL_ERROR "LibTorch CMake configuration not found. Please ensure LibTorch is properly installed.")
        endif()
        
        # LIBTORCH_DIR carries no version, so without this a stale libtorch is
        # completely invisible to the build.
        read_version_from_file(detected_version "${LIBTORCH_DIR}/build-version")
        report_version_drift("LibTorch" "${PYTORCH_VERSION}" "${detected_version}" "${LIBTORCH_DIR}"
            "Re-run ./scripts/setup_libtorch.sh to install the declared version.")

        # build-version also names the compute variant ("2.3.0+cpu",
        # "2.3.0+cu121"). versions.env pins only a version, so the variant has
        # no pin to drift against -- but it decides device placement, and
        # getting it wrong fails silently: with a CPU-only build
        # torch::cuda::is_available() is false, so LibtorchInfer puts every
        # tensor on the CPU however the caller set use_gpu, and the only symptom
        # is that inference is slow. Say so instead.
        set(libtorch_build "")
        if(EXISTS "${LIBTORCH_DIR}/build-version")
            file(STRINGS "${LIBTORCH_DIR}/build-version" libtorch_build_lines LIMIT_COUNT 1)
            if(libtorch_build_lines)
                list(GET libtorch_build_lines 0 libtorch_build)
                string(STRIP "${libtorch_build}" libtorch_build)
            endif()
        endif()
        if(libtorch_build)
            message(STATUS "LibTorch build: ${libtorch_build}")
            if(libtorch_build MATCHES "\\+cpu$")
                find_package(CUDAToolkit QUIET)
                if(CUDAToolkit_FOUND)
                    message(WARNING
                        "LibTorch at ${LIBTORCH_DIR} is a CPU-only build (${libtorch_build}), but CUDA "
                        "${CUDAToolkit_VERSION} is available here. torch::cuda::is_available() will be "
                        "false, so the LibTorch backend will run on the CPU whatever use_gpu is set to. "
                        "Re-run ./scripts/setup_libtorch.sh with FORCE=true to install a CUDA build, or "
                        "set LIBTORCH_VARIANT=cpu to make the CPU-only choice explicit.")
                endif()
            endif()
        endif()

        message(STATUS "✓ LibTorch validation passed")
    endif()
endfunction()

# Function to validate OpenVINO
function(validate_openvino)
    if("OPENVINO" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
        validate_dependency("OpenVINO" "${OPENVINO_DIR}")
        
        # Check for required files
        set(required_files
            "${OPENVINO_DIR}/runtime/include/openvino/openvino.hpp"
        )

        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "OpenVINO installation incomplete. Missing: ${file}")
            endif()
        endforeach()

        require_any_existing_path(
            "OpenVINO runtime library"
            "${OPENVINO_DIR}/runtime/lib/intel64/libopenvino.so"
            "${OPENVINO_DIR}/runtime/lib/intel64/libopenvino.dylib"
            "${OPENVINO_DIR}/runtime/lib/intel64/openvino.lib"
            "${OPENVINO_DIR}/runtime/bin/intel64/Release/openvino.dll"
            "${OPENVINO_DIR}/runtime/lib/intel64/Release/openvino.lib"
            "${OPENVINO_DIR}/runtime/bin/Release/openvino.dll"
        )
        
        read_version_from_file(detected_version "${OPENVINO_DIR}/runtime/version.txt")
        report_version_drift("OpenVINO" "${OPENVINO_VERSION}" "${detected_version}" "${OPENVINO_DIR}"
            "Re-run ./scripts/setup_openvino.sh to install the declared version.")

        message(STATUS "✓ OpenVINO validation passed")
    endif()
endfunction()

# Function to validate MIGraphX
function(validate_migraphx)
    if("MIGRAPHX" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
        validate_dependency("MIGraphX root" "${MIGRAPHX_ROOT}")

        list(APPEND CMAKE_PREFIX_PATH "${MIGRAPHX_ROOT}")
        find_package(migraphx REQUIRED)

        message(STATUS "✓ MIGraphX validation passed")
    endif()
endfunction()

# Function to validate CUDA/ROCm (if GPU support is requested)
function(validate_cuda)
    if("TENSORRT" IN_LIST NEURIPLO_REQUESTED_BACKENDS OR "ONNX_RUNTIME" IN_LIST NEURIPLO_REQUESTED_BACKENDS OR "GGML" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            message(STATUS "✓ CUDA toolkit found (${CUDAToolkit_VERSION})")
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
        # Try to find glog but don't require it
        find_package(Glog QUIET)
        if(Glog_FOUND)
            message(STATUS "✓ glog found")
        else()
            message(STATUS "glog not found - parent project should handle glog")
        endif()
        
        return()
    endif()
    
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
    
    # Each validator self-guards on NEURIPLO_REQUESTED_BACKENDS, so validating
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
    validate_dali()

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
    
    if("OPENCV_DNN" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
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
    if("OPENCV_DNN" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
        # The only OpenCV check left in the tree. It used to live in
        # validate_system_dependencies(), which runs for every build, so a
        # backend that never touches OpenCV still could not configure without
        # it. Guarded on the requested backend, it costs the other 13 nothing.
        find_package(OpenCV REQUIRED)
        # Guarded on OpenCV_FOUND so that a failed find reports only that, and
        # not a second, misleading "version is too old" from an empty version.
        if(OpenCV_FOUND AND OpenCV_VERSION VERSION_LESS OPENCV_MIN_VERSION)
            message(FATAL_ERROR "OpenCV version ${OpenCV_VERSION} is too old. Minimum required: ${OPENCV_MIN_VERSION}")
        endif()
        message(STATUS "✓ OpenCV ${OpenCV_VERSION} found")
        message(STATUS "✓ OpenCV DNN validation passed")
    endif()
endfunction()

# Function to validate LibTensorFlow
function(validate_libtensorflow)
    if("LIBTENSORFLOW" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
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
    if("GGML" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
        validate_dependency("GGML" "${GGML_DIR}")
        
        # Check for required files
        set(required_files
            "${GGML_DIR}/include/ggml.h"
            "${GGML_DIR}/include/ggml-backend.h"
        )

        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "GGML installation incomplete. Missing: ${file}")
            endif()
        endforeach()

        require_any_existing_path(
            "GGML base library"
            "${GGML_DIR}/lib/libggml-base.so"
            "${GGML_DIR}/lib/libggml-base.dylib"
            "${GGML_DIR}/lib/libggml-base.lib"
            "${GGML_DIR}/lib/ggml-base.lib"
            "${GGML_DIR}/lib/Release/libggml-base.lib"
            "${GGML_DIR}/lib/Release/ggml-base.lib"
        )
        require_any_existing_path(
            "GGML CPU library"
            "${GGML_DIR}/lib/libggml-cpu.so"
            "${GGML_DIR}/lib/libggml-cpu.dylib"
            "${GGML_DIR}/lib/libggml-cpu.lib"
            "${GGML_DIR}/lib/ggml-cpu.lib"
            "${GGML_DIR}/lib/Release/libggml-cpu.lib"
            "${GGML_DIR}/lib/Release/ggml-cpu.lib"
        )
        require_any_existing_path(
            "GGML BLAS library"
            "${GGML_DIR}/lib/libggml-blas.so"
            "${GGML_DIR}/lib/libggml-blas.dylib"
            "${GGML_DIR}/lib/libggml-blas.lib"
            "${GGML_DIR}/lib/ggml-blas.lib"
            "${GGML_DIR}/lib/Release/libggml-blas.lib"
            "${GGML_DIR}/lib/Release/ggml-blas.lib"
        )
        
        read_stamped_version(detected_version "${GGML_DIR}")
        report_tag_drift("GGML" "${GGML_VERSION}" "${detected_version}" "${GGML_DIR}"
            "Re-run ./scripts/setup_ggml.sh with FORCE=true to rebuild it from the declared version.")

        message(STATUS "✓ GGML validation passed")
    endif()
endfunction()

function(validate_tvm)
    if("TVM" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
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
        
        require_any_existing_path(
            "TVM runtime library"
            "${TVM_DIR}/build/libtvm_runtime.so"
            "${TVM_DIR}/build/libtvm.so"
            "${TVM_DIR}/build/tvm_runtime.dll"
            "${TVM_DIR}/build/tvm_runtime.lib"
            "${TVM_DIR}/build/tvm.lib"
            "${TVM_DIR}/build/Release/tvm_runtime.lib"
            "${TVM_DIR}/build/Release/tvm.lib"
        )
        
        read_stamped_version(detected_version "${TVM_DIR}")
        report_tag_drift("TVM" "${TVM_VERSION}" "${detected_version}" "${TVM_DIR}"
            "Re-run ./scripts/setup_tvm.sh with FORCE=true to rebuild it from the declared version.")

        message(STATUS "✓ TVM validation passed")
    endif()
endfunction()

# Function to validate llama.cpp
function(validate_llamacpp)
    if("LLAMACPP" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
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

        read_stamped_version(detected_version "${LLAMACPP_DIR}")
        report_tag_drift("llama.cpp" "${LLAMACPP_VERSION}" "${detected_version}" "${LLAMACPP_DIR}"
            "Re-run ./scripts/setup_llamacpp.sh with FORCE=true to rebuild it from the declared version.")

        message(STATUS "✓ llama.cpp validation passed")
    endif()
endfunction()

# Function to validate ExecuTorch
function(validate_executorch)
    if("EXECUTORCH" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
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

        read_stamped_version(detected_version "${EXECUTORCH_DIR}")
        report_tag_drift("ExecuTorch" "${EXECUTORCH_VERSION}" "${detected_version}" "${EXECUTORCH_DIR}"
            "Re-run ./scripts/setup_executorch.sh with FORCE=true to rebuild it from the declared version.")

        message(STATUS "✓ ExecuTorch validation passed")
    endif()
endfunction()

# Function to validate LiteRT
function(validate_litert)
    if("LITERT" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
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

        read_stamped_version(detected_version "${LITERT_DIR}")
        report_tag_drift("LiteRT" "${LITERT_VERSION}" "${detected_version}" "${LITERT_DIR}"
            "Re-run ./scripts/setup_litert.sh with FORCE=true to rebuild it from the declared version.")

        message(STATUS "✓ LiteRT validation passed")
    endif()
endfunction()

# Function to validate DALI
function(validate_dali)
    if("DALI" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
        validate_dependency("DALI" "${DALI_DIR}")

        # Both libraries are required: libdali.so does not pull in the operator
        # library through DT_NEEDED, and without it every pipeline fails at run
        # time with `No schema found for operator "decoders__Image"`.
        set(required_files
            "${DALI_DIR}/include/dali/c_api.h"
            "${DALI_DIR}/libdali.so"
            "${DALI_DIR}/libdali_operators.so"
        )

        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "DALI installation incomplete. Missing: ${file}\n"
                    "NVIDIA publishes no standalone C++ DALI distribution; run "
                    "scripts/setup_dali.sh to extract it from the nvidia-dali wheel.")
            endif()
        endforeach()

        message(STATUS "✓ DALI validation passed")
    endif()
endfunction()

# Function to validate Cactus
function(validate_cactus)
    if("CACTUS" IN_LIST NEURIPLO_REQUESTED_BACKENDS)
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

        read_stamped_version(detected_version "${CACTUS_DIR}")
        report_tag_drift("Cactus" "${CACTUS_VERSION}" "${detected_version}" "${CACTUS_DIR}"
            "Re-run ./scripts/setup_cactus.sh with FORCE=true to rebuild it from the declared version.")

        message(STATUS "✓ Cactus validation passed")
    endif()
endfunction()
