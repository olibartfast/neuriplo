set(OPENVINO_SOURCES
${INFER_ROOT}/openvino/src/OVInfer.cpp
# Add more OPENVINO source files here if needed
)

# Set OpenVINO paths
if(DEFINED ENV{OPENVINO_DIR})
    set(OpenVINO_DIR "$ENV{OPENVINO_DIR}/runtime/cmake")
    set(InferenceEngine_DIR "$ENV{OPENVINO_DIR}/runtime/cmake")
    
    # Add include directories
    include_directories("$ENV{OPENVINO_DIR}/runtime/include")
    
    # Add library directories
    link_directories("$ENV{OPENVINO_DIR}/runtime/lib/intel64")
    
    # Add to CMAKE_MODULE_PATH
    list(APPEND CMAKE_MODULE_PATH "$ENV{OPENVINO_DIR}/runtime/cmake")
else()
    # Set default OpenVINO path
    set(OpenVINO_DIR "${HOME}/dependencies/${OPENVINO_VERSION}/runtime/cmake")
    set(InferenceEngine_DIR "${HOME}/dependencies/${OPENVINO_VERSION}/runtime/cmake")
    
    # Add include directories
    include_directories("${HOME}/dependencies/${OPENVINO_VERSION}/runtime/include")
    
    # Add library directories
    link_directories("${HOME}/dependencies/${OPENVINO_VERSION}/runtime/lib/intel64")
    
    # Add to CMAKE_MODULE_PATH
    list(APPEND CMAKE_MODULE_PATH "${HOME}/dependencies/${OPENVINO_VERSION}/runtime/cmake")
endif()

find_package(OpenVINO REQUIRED)

list(APPEND SOURCES ${OPENVINO_SOURCES})

add_compile_definitions(USE_OPENVINO)
