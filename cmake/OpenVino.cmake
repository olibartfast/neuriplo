set(OPENVINO_SOURCES
${INFER_ROOT}/openvino/src/OVInfer.cpp
# Add more OPENVINO source files here if needed
)

# Set OpenVINO paths
if(DEFINED ENV{OPENVINO_DIR})
    set(OpenVINO_DIR "$ENV{OPENVINO_DIR}/runtime/cmake")
    set(InferenceEngine_DIR "$ENV{OPENVINO_DIR}/runtime/cmake")
    
    # Add include directories
    include_directories(SYSTEM "$ENV{OPENVINO_DIR}/runtime/include")
    
    # Add to CMAKE_PREFIX_PATH
    list(APPEND CMAKE_PREFIX_PATH "$ENV{OPENVINO_DIR}")
    
    message(STATUS "Using OPENVINO_DIR from environment: $ENV{OPENVINO_DIR}")
    message(STATUS "OpenVINO_DIR set to: ${OpenVINO_DIR}")
    message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
else()
    # Set default OpenVINO path
    if(DEFINED DEFAULT_DEPENDENCY_ROOT AND NOT DEFAULT_DEPENDENCY_ROOT STREQUAL "")
        set(OPENVINO_INSTALL_DIR "${DEFAULT_DEPENDENCY_ROOT}/openvino_${OPENVINO_VERSION}")
    else()
        set(OPENVINO_INSTALL_DIR "$ENV{HOME}/dependencies/openvino_${OPENVINO_VERSION}")
    endif()
    set(OpenVINO_DIR "${OPENVINO_INSTALL_DIR}/runtime/cmake")
    set(InferenceEngine_DIR "${OPENVINO_INSTALL_DIR}/runtime/cmake")
    
    # Add include directories
    include_directories(SYSTEM "${OPENVINO_INSTALL_DIR}/runtime/include")
    # Add to CMAKE_PREFIX_PATH
    list(APPEND CMAKE_PREFIX_PATH "${OPENVINO_INSTALL_DIR}")
    
    message(STATUS "Using default OpenVINO path")
    message(STATUS "OPENVINO_VERSION: ${OPENVINO_VERSION}")
    message(STATUS "HOME: $ENV{HOME}")
    message(STATUS "OPENVINO_INSTALL_DIR: ${OPENVINO_INSTALL_DIR}")
    message(STATUS "OpenVINO_DIR set to: ${OpenVINO_DIR}")
    message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
endif()

find_package(OpenVINO REQUIRED)

list(APPEND SOURCES ${OPENVINO_SOURCES})

add_compile_definitions(USE_OPENVINO)
