# Include framework-specific source files and libraries
if (DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/opencv-dnn/src)
elseif (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${ONNX_RUNTIME_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/onnx-runtime/src)
    target_link_directories(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_DIR}/lib)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so)
elseif (DEFAULT_BACKEND STREQUAL "LIBTORCH")
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/libtorch/src)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE C10_USE_GLOG)
elseif (DEFAULT_BACKEND STREQUAL "TENSORRT")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE /usr/local/cuda/include ${TENSORRT_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/tensorrt/src)
    target_link_directories(${PROJECT_NAME} PRIVATE  /usr/local/cuda/lib64 ${TENSORRT_DIR}/lib)
    target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvonnxparser cudart)
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
    target_link_directories(${PROJECT_NAME} PRIVATE ${GGML_DIR}/lib)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${GGML_DIR}/lib/libggml-base.so
        ${GGML_DIR}/lib/libggml-cpu.so
        ${GGML_DIR}/lib/libggml-blas.so
    )
elseif(DEFAULT_BACKEND STREQUAL "TVM")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE 
        ${TVM_DIR}/include 
        ${TVM_DIR}/3rdparty/dmlc-core/include
        ${TVM_DIR}/3rdparty/dlpack/include
        ${TVM_DIR}/3rdparty/dlpack
        ${TVM_DIR}/3rdparty/tvm-ffi/3rdparty/dlpack/include
        ${TVM_DIR}/3rdparty/tvm-ffi/include
        ${TVM_DIR}/3rdparty)
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/tvm/src)
    target_link_directories(${PROJECT_NAME} PRIVATE ${TVM_DIR}/build)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TVM_DIR}/build/libtvm_runtime.so)
    
    # Suppress macro redefinition warnings between glog and DMLC
    target_compile_options(${PROJECT_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-macro-redefined>
        $<$<COMPILE_LANGUAGE:CXX>:-w>)
elseif(DEFAULT_BACKEND STREQUAL "CACTUS")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${CACTUS_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/cactus/src)
    target_link_directories(${PROJECT_NAME} PRIVATE ${CACTUS_DIR}/lib)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CACTUS_DIR}/lib/libcactus.so)
elseif(DEFAULT_BACKEND STREQUAL "MIGRAPHX")
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${INFER_ROOT}/migraphx/src)
    target_link_libraries(${PROJECT_NAME} PRIVATE migraphx::c)
elseif(DEFAULT_BACKEND STREQUAL "LLAMACPP")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${LLAMACPP_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/llamacpp/src)
    target_link_directories(${PROJECT_NAME} PRIVATE ${LLAMACPP_DIR}/lib)
    # libllama.so has transitive SONAME deps on libggml.so and libggml-base.so;
    # link all present ggml libs so the linker can resolve them.
    find_library(LLAMACPP_GGML_LIB     NAMES ggml     PATHS ${LLAMACPP_DIR}/lib NO_DEFAULT_PATH)
    find_library(LLAMACPP_GGML_BASE_LIB NAMES ggml-base PATHS ${LLAMACPP_DIR}/lib NO_DEFAULT_PATH)
    find_library(LLAMACPP_GGML_CPU_LIB  NAMES ggml-cpu  PATHS ${LLAMACPP_DIR}/lib NO_DEFAULT_PATH)
    find_library(LLAMACPP_MTMD_LIB      NAMES mtmd      PATHS ${LLAMACPP_DIR}/lib NO_DEFAULT_PATH)
    set(_GGML_LIBS "")
    foreach(_lib IN ITEMS LLAMACPP_GGML_LIB LLAMACPP_GGML_BASE_LIB LLAMACPP_GGML_CPU_LIB)
        if(${_lib})
            list(APPEND _GGML_LIBS "${${_lib}}")
        endif()
    endforeach()
    if(NOT _GGML_LIBS)
        message(FATAL_ERROR "No ggml libraries found in ${LLAMACPP_DIR}/lib")
    endif()
    if(NOT LLAMACPP_MTMD_LIB)
        message(FATAL_ERROR "libmtmd not found in ${LLAMACPP_DIR}/lib — rebuild llama.cpp with BUILD_SHARED_LIBS=ON")
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE llama mtmd ${_GGML_LIBS})
    # rpath-link lets the linker resolve SONAME transitive deps (libggml.so.0, libggml-base.so.0)
    # that libllama.so pulls in; rpath embeds the search path in the final binary.
    target_link_options(${PROJECT_NAME} PRIVATE
        "-Wl,-rpath-link,${LLAMACPP_DIR}/lib"
        "-Wl,-rpath,${LLAMACPP_DIR}/lib")
elseif(DEFAULT_BACKEND STREQUAL "EXECUTORCH")
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/executorch/src)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        executorch
        extension_module_static
        extension_tensor
        portable_ops_lib
        portable_kernels
    )
endif()
