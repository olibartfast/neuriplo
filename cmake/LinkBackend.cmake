# Include directories and libraries for one backend on an arbitrary target.
# Each branch is scoped to one backend so any subset can be linked into the
# same target (the neuriplo library or a plugin module).
function(neuriplo_link_backend_to target backend)
if (backend STREQUAL "OPENCV_DNN")
    # The only place in the tree that asks for OpenCV. It is REQUIRED here
    # rather than at the top level so that configuring any other backend needs
    # no OpenCV installed at all.
    find_package(OpenCV REQUIRED)
    target_include_directories(${target} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/opencv-dnn/src)
    target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
elseif (backend STREQUAL "ONNX_RUNTIME")
    target_include_directories(${target} SYSTEM PRIVATE ${ONNX_RUNTIME_DIR}/include)
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/onnx-runtime/src)
    target_link_directories(${target} PRIVATE
        ${ONNX_RUNTIME_DIR}/lib
        ${ONNX_RUNTIME_DIR}/lib/Release
        ${ONNX_RUNTIME_DIR}/lib/Debug)
    target_link_libraries(${target} PRIVATE onnxruntime)
elseif (backend STREQUAL "LIBTORCH")
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/libtorch/src)
    target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(${target} PRIVATE C10_USE_GLOG)
elseif (backend STREQUAL "DALI")
    # Both libraries are required. libdali.so pulls in libdali_core.so and
    # libdali_kernels.so through DT_NEEDED, but NOT the operator library --
    # DALI's Python bindings dlopen that one. Without it every pipeline fails at
    # run time with `No schema found for operator "decoders__Image"`.
    find_package(CUDAToolkit QUIET)
    target_include_directories(${target} SYSTEM PRIVATE ${DALI_DIR}/include)
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/dali/src)
    if(CUDAToolkit_FOUND)
        target_include_directories(${target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    target_link_directories(${target} PRIVATE ${DALI_DIR})
    target_link_libraries(${target} PRIVATE dali dali_operators)
elseif (backend STREQUAL "TENSORRT")
    find_package(CUDAToolkit QUIET)
    target_include_directories(${target} SYSTEM PRIVATE ${TENSORRT_DIR}/include)
    if(CUDAToolkit_FOUND)
        target_include_directories(${target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/tensorrt/src)
    target_link_directories(${target} PRIVATE
        ${TENSORRT_DIR}/lib
        ${TENSORRT_DIR}/lib/Release
        ${TENSORRT_DIR}/lib/Debug)
    if(CUDAToolkit_FOUND)
        target_link_directories(${target} PRIVATE ${CUDAToolkit_LIBRARY_DIR})
    endif()
    target_link_libraries(${target} PRIVATE nvinfer nvonnxparser
        $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cudart,cudart>)
elseif(backend STREQUAL "LIBTENSORFLOW" )
    target_include_directories(${target} SYSTEM PRIVATE ${TensorFlow_INCLUDE_DIR})
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/libtensorflow/src)
    target_link_libraries(${target} PRIVATE ${TensorFlow_CC_LIBRARY} ${TensorFlow_FRAMEWORK_LIBRARY})
elseif(backend STREQUAL "OPENVINO")
    target_include_directories(${target} SYSTEM PRIVATE ${InferenceEngine_INCLUDE_DIRS})
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/openvino/src)
    target_link_libraries(${target} PRIVATE openvino::runtime )
elseif(backend STREQUAL "GGML")
    target_include_directories(${target} SYSTEM PRIVATE ${GGML_DIR}/include)
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/ggml/src)
    target_link_directories(${target} PRIVATE
        ${GGML_DIR}/lib
        ${GGML_DIR}/lib/Release
        ${GGML_DIR}/lib/Debug)
    target_link_libraries(${target} PRIVATE ggml-base ggml-cpu ggml-blas)
elseif(backend STREQUAL "TVM")
    target_include_directories(${target} SYSTEM PRIVATE 
        ${TVM_DIR}/include 
        ${TVM_DIR}/3rdparty/dmlc-core/include
        ${TVM_DIR}/3rdparty/dlpack/include
        ${TVM_DIR}/3rdparty/dlpack
        ${TVM_DIR}/3rdparty/tvm-ffi/3rdparty/dlpack/include
        ${TVM_DIR}/3rdparty/tvm-ffi/include
        ${TVM_DIR}/3rdparty)
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/tvm/src)
    target_link_directories(${target} PRIVATE
        ${TVM_DIR}/build
        ${TVM_DIR}/build/Release
        ${TVM_DIR}/build/Debug)
    target_link_libraries(${target} PRIVATE tvm_runtime)
    
    # Suppress macro redefinition warnings between glog and DMLC
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-macro-redefined>
            $<$<COMPILE_LANGUAGE:CXX>:-w>)
    endif()
elseif(backend STREQUAL "CACTUS")
    target_include_directories(${target} SYSTEM PRIVATE ${CACTUS_DIR}/include)
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/cactus/src)
    target_link_directories(${target} PRIVATE ${CACTUS_DIR}/lib)
    target_link_libraries(${target} PRIVATE ${CACTUS_DIR}/lib/libcactus.so)
elseif(backend STREQUAL "MIGRAPHX")
    target_include_directories(${target} PRIVATE
        ${INFER_ROOT}/migraphx/src)
    target_link_libraries(${target} PRIVATE migraphx::c)
elseif(backend STREQUAL "LLAMACPP")
    target_include_directories(${target} SYSTEM PRIVATE ${LLAMACPP_DIR}/include)
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/llamacpp/src)
    target_link_directories(${target} PRIVATE ${LLAMACPP_DIR}/lib)
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
    target_link_libraries(${target} PRIVATE llama mtmd ${_GGML_LIBS})
    # rpath-link lets the linker resolve SONAME transitive deps (libggml.so.0, libggml-base.so.0)
    # that libllama.so pulls in; rpath embeds the search path in the final binary.
    if(NOT MSVC)
        target_link_options(${target} PRIVATE
            "-Wl,-rpath-link,${LLAMACPP_DIR}/lib"
            "-Wl,-rpath,${LLAMACPP_DIR}/lib")
    endif()
elseif(backend STREQUAL "EXECUTORCH")
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/executorch/src)
    set(_ET_LIBS
        executorch
        extension_module_static
        extension_tensor
        portable_ops_lib
        portable_kernels
    )
    if(EXECUTORCH_DELEGATE STREQUAL "xnnpack")
        list(APPEND _ET_LIBS xnnpack_backend)
    endif()
    target_link_libraries(${target} PRIVATE ${_ET_LIBS})
elseif(backend STREQUAL "LITERT")
    target_include_directories(${target} SYSTEM PRIVATE ${LITERT_DIR}/include)
    target_include_directories(${target} PRIVATE ${INFER_ROOT}/litert/src)
    target_link_directories(${target} PRIVATE ${LITERT_DIR}/lib)
    target_link_libraries(${target} PRIVATE ${LITERT_LINK_LIBRARIES})
endif()
endfunction()

foreach(neuriplo_enabled_backend IN LISTS NEURIPLO_ENABLED_BACKENDS)
    neuriplo_link_backend_to(${PROJECT_NAME} "${neuriplo_enabled_backend}")
endforeach()
