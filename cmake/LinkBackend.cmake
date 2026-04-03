# Include framework-specific source files and libraries
function(_find_backend_library out_var)
    find_library(${out_var}
        NAMES ${ARGN}
        PATHS
            "${ONNX_RUNTIME_DIR}/lib"
            "${ONNX_RUNTIME_DIR}/lib/Release"
            "${ONNX_RUNTIME_DIR}/lib/Debug"
            "${GGML_DIR}/lib"
            "${GGML_DIR}/lib/Release"
            "${GGML_DIR}/lib/Debug"
            "${TVM_DIR}/build"
            "${TVM_DIR}/build/Release"
            "${TVM_DIR}/build/Debug"
            "${TENSORRT_DIR}/lib"
            "${TENSORRT_DIR}/lib/Release"
            "${TENSORRT_DIR}/lib/Debug"
        NO_DEFAULT_PATH
    )
endfunction()

if (DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/opencv-dnn/src)
elseif (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${ONNX_RUNTIME_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/onnx-runtime/src)
    _find_backend_library(ONNX_RUNTIME_LIBRARY onnxruntime libonnxruntime)
    if(NOT ONNX_RUNTIME_LIBRARY)
        message(FATAL_ERROR "Unable to locate the ONNX Runtime library under ${ONNX_RUNTIME_DIR}/lib")
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_LIBRARY})
elseif (DEFAULT_BACKEND STREQUAL "LIBTORCH")
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/libtorch/src)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE C10_USE_GLOG)
elseif (DEFAULT_BACKEND STREQUAL "TENSORRT")
    find_package(CUDAToolkit QUIET)
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${TENSORRT_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/tensorrt/src)
    if(CUDAToolkit_FOUND)
        target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    _find_backend_library(TENSORRT_NVINFER_LIBRARY nvinfer libnvinfer)
    _find_backend_library(TENSORRT_NVONNXPARSER_LIBRARY nvonnxparser libnvonnxparser)
    if(NOT TENSORRT_NVINFER_LIBRARY OR NOT TENSORRT_NVONNXPARSER_LIBRARY)
        message(FATAL_ERROR "Unable to locate the TensorRT libraries under ${TENSORRT_DIR}/lib")
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${TENSORRT_NVINFER_LIBRARY}
        ${TENSORRT_NVONNXPARSER_LIBRARY}
    )
    if(TARGET CUDA::cudart)
        target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
    else()
        target_link_libraries(${PROJECT_NAME} PRIVATE cudart)
    endif()
elseif(DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW" )
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${TensorFlow_INCLUDE_DIR})
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/libtensorflow/src)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TensorFlow_CC_LIBRARY} ${TensorFlow_FRAMEWORK_LIBRARY})
elseif(DEFAULT_BACKEND STREQUAL "OPENVINO")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${InferenceEngine_INCLUDE_DIRS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/openvino/src)
    target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime )
elseif(DEFAULT_BACKEND STREQUAL "GGML")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${GGML_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/ggml/src)
    _find_backend_library(GGML_BASE_LIBRARY ggml-base libggml-base)
    _find_backend_library(GGML_CPU_LIBRARY ggml-cpu libggml-cpu)
    _find_backend_library(GGML_BLAS_LIBRARY ggml-blas libggml-blas)
    if(NOT GGML_BASE_LIBRARY OR NOT GGML_CPU_LIBRARY OR NOT GGML_BLAS_LIBRARY)
        message(FATAL_ERROR "Unable to locate the required GGML libraries under ${GGML_DIR}/lib")
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${GGML_BASE_LIBRARY}
        ${GGML_CPU_LIBRARY}
        ${GGML_BLAS_LIBRARY}
    )
elseif(DEFAULT_BACKEND STREQUAL "TVM")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE 
        ${TVM_DIR}/include 
        ${TVM_DIR}/3rdparty/dmlc-core/include
        ${TVM_DIR}/3rdparty/dlpack/include
        ${TVM_DIR}/3rdparty/dlpack
        ${TVM_DIR}/3rdparty/tvm-ffi/3rdparty/dlpack/include
        ${TVM_DIR}/3rdparty/tvm-ffi/include
        ${TVM_DIR}/3rdparty
    )
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/tvm/src)
    _find_backend_library(TVM_RUNTIME_LIBRARY tvm_runtime libtvm_runtime)
    if(NOT TVM_RUNTIME_LIBRARY)
        message(FATAL_ERROR "Unable to locate the TVM runtime library under ${TVM_DIR}/build")
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TVM_RUNTIME_LIBRARY})
    
    # Suppress macro redefinition warnings between glog and DMLC
    if(NOT MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-macro-redefined>
            $<$<COMPILE_LANGUAGE:CXX>:-w>)
    endif()
elseif(DEFAULT_BACKEND STREQUAL "MIGRAPHX")
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${INFER_ROOT}/migraphx/src)
    target_link_libraries(${PROJECT_NAME} PRIVATE migraphx::c)
endif()
