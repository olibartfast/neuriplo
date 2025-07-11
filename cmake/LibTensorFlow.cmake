# TensorFlow Configuration

# Add cmake modules path to find our custom FindTensorFlow.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

# Find TensorFlow
find_package(TensorFlow REQUIRED)

if(NOT TensorFlow_FOUND)
    message(FATAL_ERROR "TensorFlow not found. Please install TensorFlow C++ library.")
endif()

message(STATUS "TensorFlow found: ${TensorFlow_INCLUDE_DIRS}")
message(STATUS "TensorFlow libraries: ${TensorFlow_LIBRARIES}")
message(STATUS "TensorFlow_INCLUDE_DIR: ${TensorFlow_INCLUDE_DIR}")

# Set TensorFlow include directories globally immediately after finding TensorFlow
include_directories(${TensorFlow_INCLUDE_DIR})

set(TensorFlow_SOURCES
    ${INFER_ROOT}/libtensorflow/src/TFDetectionAPI.cpp
)

list(APPEND SOURCES ${TensorFlow_SOURCES})

# Add compile definition to indicate TensorFlow usage
add_compile_definitions(USE_LIBTENSORFLOW)

