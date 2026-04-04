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
elseif(DEFAULT_BACKEND STREQUAL "LLAMACPP")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${LLAMACPP_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/llamacpp/src)
    target_link_directories(${PROJECT_NAME} PRIVATE ${LLAMACPP_DIR}/lib)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${LLAMACPP_DIR}/lib/libllama.so
        ${LLAMACPP_DIR}/lib/libggml.so
    )
endif()
