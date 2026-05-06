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
