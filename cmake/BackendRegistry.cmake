# Central registry for CMake-visible inference backend metadata.
#
# Keep backend identifiers stable: they are public DEFAULT_BACKEND values used by
# scripts, docs, and downstream CMake consumers.

set(NEURIPLO_BACKEND_IDS
    OPENCV_DNN
    ONNX_RUNTIME
    LIBTORCH
    LIBTENSORFLOW
    TENSORRT
    OPENVINO
    GGML
    TVM
    MIGRAPHX
    CACTUS
    LLAMACPP
    EXECUTORCH
    LITERT
)

set(NEURIPLO_BACKEND_OPENCV_DNN_MODULE OpenCVdnn)
set(NEURIPLO_BACKEND_OPENCV_DNN_TEST_DIR backends/opencv-dnn/test)
set(NEURIPLO_BACKEND_OPENCV_DNN_VERSION_VAR OPENCV_VERSION)

set(NEURIPLO_BACKEND_ONNX_RUNTIME_MODULE ONNXRuntime)
set(NEURIPLO_BACKEND_ONNX_RUNTIME_TEST_DIR backends/onnx-runtime/test)
set(NEURIPLO_BACKEND_ONNX_RUNTIME_VERSION_VAR ONNX_RUNTIME_VERSION)

set(NEURIPLO_BACKEND_LIBTORCH_MODULE LibTorch)
set(NEURIPLO_BACKEND_LIBTORCH_TEST_DIR backends/libtorch/test)
set(NEURIPLO_BACKEND_LIBTORCH_VERSION_VAR PYTORCH_VERSION)

set(NEURIPLO_BACKEND_LIBTENSORFLOW_MODULE LibTensorFlow)
set(NEURIPLO_BACKEND_LIBTENSORFLOW_TEST_DIR backends/libtensorflow/test)
set(NEURIPLO_BACKEND_LIBTENSORFLOW_VERSION_VAR TENSORFLOW_VERSION)

set(NEURIPLO_BACKEND_TENSORRT_MODULE TensorRT)
set(NEURIPLO_BACKEND_TENSORRT_TEST_DIR backends/tensorrt/test)
set(NEURIPLO_BACKEND_TENSORRT_VERSION_VAR TENSORRT_VERSION)

set(NEURIPLO_BACKEND_OPENVINO_MODULE OpenVino)
set(NEURIPLO_BACKEND_OPENVINO_TEST_DIR backends/openvino/test)
set(NEURIPLO_BACKEND_OPENVINO_VERSION_VAR OPENVINO_VERSION)

set(NEURIPLO_BACKEND_GGML_MODULE GGML)
set(NEURIPLO_BACKEND_GGML_TEST_DIR backends/ggml/test)
set(NEURIPLO_BACKEND_GGML_VERSION_VAR GGML_VERSION)

set(NEURIPLO_BACKEND_TVM_MODULE TVM)
set(NEURIPLO_BACKEND_TVM_TEST_DIR backends/tvm/test)
set(NEURIPLO_BACKEND_TVM_VERSION_VAR TVM_VERSION)

set(NEURIPLO_BACKEND_MIGRAPHX_MODULE MIGraphX)
set(NEURIPLO_BACKEND_MIGRAPHX_TEST_DIR backends/migraphx/test)
set(NEURIPLO_BACKEND_MIGRAPHX_VERSION_VAR MIGRAPHX_VERSION)

set(NEURIPLO_BACKEND_CACTUS_MODULE Cactus)
set(NEURIPLO_BACKEND_CACTUS_TEST_DIR backends/cactus/test)
set(NEURIPLO_BACKEND_CACTUS_VERSION_VAR CACTUS_VERSION)

set(NEURIPLO_BACKEND_LLAMACPP_MODULE LlamaCpp)
set(NEURIPLO_BACKEND_LLAMACPP_TEST_DIR backends/llamacpp/test)
set(NEURIPLO_BACKEND_LLAMACPP_VERSION_VAR LLAMACPP_VERSION)

set(NEURIPLO_BACKEND_EXECUTORCH_MODULE ExecuTorch)
set(NEURIPLO_BACKEND_EXECUTORCH_TEST_DIR backends/executorch/test)
set(NEURIPLO_BACKEND_EXECUTORCH_VERSION_VAR EXECUTORCH_VERSION)

set(NEURIPLO_BACKEND_LITERT_MODULE LiteRT)
set(NEURIPLO_BACKEND_LITERT_TEST_DIR backends/litert/test)
set(NEURIPLO_BACKEND_LITERT_VERSION_VAR LITERT_VERSION)

# Plugin-build metadata: source directory under backends/, factory header and
# class for the generated plugin entry point, display name, and GPU forcing.
set(NEURIPLO_BACKEND_OPENCV_DNN_SOURCE_DIR backends/opencv-dnn/src)
set(NEURIPLO_BACKEND_OPENCV_DNN_FACTORY_HEADER OCVDNNRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_OPENCV_DNN_FACTORY_CLASS OCVDNNRuntimeFactory)
set(NEURIPLO_BACKEND_OPENCV_DNN_DISPLAY_NAME "OpenCV DNN")
set(NEURIPLO_BACKEND_OPENCV_DNN_FORCE_GPU 0)
set(NEURIPLO_BACKEND_ONNX_RUNTIME_SOURCE_DIR backends/onnx-runtime/src)
set(NEURIPLO_BACKEND_ONNX_RUNTIME_FACTORY_HEADER ORTRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_ONNX_RUNTIME_FACTORY_CLASS ORTRuntimeFactory)
set(NEURIPLO_BACKEND_ONNX_RUNTIME_DISPLAY_NAME "ONNX Runtime")
set(NEURIPLO_BACKEND_ONNX_RUNTIME_FORCE_GPU 0)
set(NEURIPLO_BACKEND_LIBTORCH_SOURCE_DIR backends/libtorch/src)
set(NEURIPLO_BACKEND_LIBTORCH_FACTORY_HEADER LibtorchRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_LIBTORCH_FACTORY_CLASS LibtorchRuntimeFactory)
set(NEURIPLO_BACKEND_LIBTORCH_DISPLAY_NAME "LibTorch")
set(NEURIPLO_BACKEND_LIBTORCH_FORCE_GPU 0)
set(NEURIPLO_BACKEND_LIBTENSORFLOW_SOURCE_DIR backends/libtensorflow/src)
set(NEURIPLO_BACKEND_LIBTENSORFLOW_FACTORY_HEADER TFRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_LIBTENSORFLOW_FACTORY_CLASS TFRuntimeFactory)
set(NEURIPLO_BACKEND_LIBTENSORFLOW_DISPLAY_NAME "TensorFlow")
set(NEURIPLO_BACKEND_LIBTENSORFLOW_FORCE_GPU 0)
set(NEURIPLO_BACKEND_TENSORRT_SOURCE_DIR backends/tensorrt/src)
set(NEURIPLO_BACKEND_TENSORRT_FACTORY_HEADER TRTRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_TENSORRT_FACTORY_CLASS TRTRuntimeFactory)
set(NEURIPLO_BACKEND_TENSORRT_DISPLAY_NAME "TensorRT")
set(NEURIPLO_BACKEND_TENSORRT_FORCE_GPU 1)
set(NEURIPLO_BACKEND_OPENVINO_SOURCE_DIR backends/openvino/src)
set(NEURIPLO_BACKEND_OPENVINO_FACTORY_HEADER OVRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_OPENVINO_FACTORY_CLASS OVRuntimeFactory)
set(NEURIPLO_BACKEND_OPENVINO_DISPLAY_NAME "OpenVINO")
set(NEURIPLO_BACKEND_OPENVINO_FORCE_GPU 0)
set(NEURIPLO_BACKEND_GGML_SOURCE_DIR backends/ggml/src)
set(NEURIPLO_BACKEND_GGML_FACTORY_HEADER GGMLRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_GGML_FACTORY_CLASS GGMLRuntimeFactory)
set(NEURIPLO_BACKEND_GGML_DISPLAY_NAME "GGML")
set(NEURIPLO_BACKEND_GGML_FORCE_GPU 0)
set(NEURIPLO_BACKEND_TVM_SOURCE_DIR backends/tvm/src)
set(NEURIPLO_BACKEND_TVM_FACTORY_HEADER TVMRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_TVM_FACTORY_CLASS TVMRuntimeFactory)
set(NEURIPLO_BACKEND_TVM_DISPLAY_NAME "TVM")
set(NEURIPLO_BACKEND_TVM_FORCE_GPU 0)
set(NEURIPLO_BACKEND_MIGRAPHX_SOURCE_DIR backends/migraphx/src)
set(NEURIPLO_BACKEND_MIGRAPHX_FACTORY_HEADER MIGraphXRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_MIGRAPHX_FACTORY_CLASS MIGraphXRuntimeFactory)
set(NEURIPLO_BACKEND_MIGRAPHX_DISPLAY_NAME "MIGraphX")
set(NEURIPLO_BACKEND_MIGRAPHX_FORCE_GPU 0)
set(NEURIPLO_BACKEND_CACTUS_SOURCE_DIR backends/cactus/src)
set(NEURIPLO_BACKEND_CACTUS_FACTORY_HEADER CactusRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_CACTUS_FACTORY_CLASS CactusRuntimeFactory)
set(NEURIPLO_BACKEND_CACTUS_DISPLAY_NAME "Cactus")
set(NEURIPLO_BACKEND_CACTUS_FORCE_GPU 0)
set(NEURIPLO_BACKEND_LLAMACPP_SOURCE_DIR backends/llamacpp/src)
set(NEURIPLO_BACKEND_LLAMACPP_FACTORY_HEADER LlamaCppRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_LLAMACPP_FACTORY_CLASS LlamaCppRuntimeFactory)
set(NEURIPLO_BACKEND_LLAMACPP_DISPLAY_NAME "llama.cpp")
set(NEURIPLO_BACKEND_LLAMACPP_FORCE_GPU 0)
set(NEURIPLO_BACKEND_EXECUTORCH_SOURCE_DIR backends/executorch/src)
set(NEURIPLO_BACKEND_EXECUTORCH_FACTORY_HEADER ExecuTorchRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_EXECUTORCH_FACTORY_CLASS ExecuTorchRuntimeFactory)
set(NEURIPLO_BACKEND_EXECUTORCH_DISPLAY_NAME "ExecuTorch")
set(NEURIPLO_BACKEND_EXECUTORCH_FORCE_GPU 0)
set(NEURIPLO_BACKEND_LITERT_SOURCE_DIR backends/litert/src)
set(NEURIPLO_BACKEND_LITERT_FACTORY_HEADER LiteRTRuntimeFactory.hpp)
set(NEURIPLO_BACKEND_LITERT_FACTORY_CLASS LiteRTRuntimeFactory)
set(NEURIPLO_BACKEND_LITERT_DISPLAY_NAME "LiteRT")
set(NEURIPLO_BACKEND_LITERT_FORCE_GPU 0)

# Backend pairs that must not be compiled into the same binary because they
# ship conflicting shared dependencies (LLAMACPP and GGML both bundle libggml*
# at different versions). Format: "A+B".
set(NEURIPLO_BACKEND_CONFLICTS
    "LLAMACPP+GGML"
)

function(neuriplo_check_backend_conflicts backends)
    foreach(conflict IN LISTS NEURIPLO_BACKEND_CONFLICTS)
        string(REPLACE "+" ";" conflict_pair "${conflict}")
        list(GET conflict_pair 0 first)
        list(GET conflict_pair 1 second)
        if(first IN_LIST backends AND second IN_LIST backends)
            message(FATAL_ERROR
                "Backends ${first} and ${second} cannot be enabled together: "
                "they ship conflicting shared dependencies. Pick one, or load "
                "one of them as a plugin once plugin builds are available.")
        endif()
    endforeach()
endfunction()

function(neuriplo_get_supported_backends out_var)
    set(${out_var} ${NEURIPLO_BACKEND_IDS} PARENT_SCOPE)
endfunction()

function(neuriplo_is_supported_backend backend out_var)
    list(FIND NEURIPLO_BACKEND_IDS "${backend}" backend_index)
    if(backend_index EQUAL -1)
        set(${out_var} FALSE PARENT_SCOPE)
    else()
        set(${out_var} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(neuriplo_get_backend_property backend property out_var)
    set(property_var "NEURIPLO_BACKEND_${backend}_${property}")
    if(NOT DEFINED ${property_var})
        message(FATAL_ERROR "Backend '${backend}' does not define registry property '${property}'")
    endif()
    set(${out_var} "${${property_var}}" PARENT_SCOPE)
endfunction()

macro(neuriplo_include_backend_module backend)
    neuriplo_get_backend_property("${backend}" MODULE backend_module)
    include("${backend_module}")
endmacro()

function(neuriplo_add_backend_tests backend)
    neuriplo_get_backend_property("${backend}" TEST_DIR backend_test_dir)
    add_subdirectory("${backend_test_dir}")
endfunction()
