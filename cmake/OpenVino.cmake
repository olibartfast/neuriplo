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
endif()

find_package(OpenVINO REQUIRED)

list(APPEND SOURCES ${OPENVINO_SOURCES})

add_compile_definitions(USE_OPENVINO)