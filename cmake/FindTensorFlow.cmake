# FindTensorFlow.cmake
# This module finds TensorFlow C++ library

# Set TensorFlow directory
if(NOT DEFINED TENSORFLOW_DIR)
    set(TENSORFLOW_DIR "$ENV{HOME}/dependencies/tensorflow" CACHE PATH "Path to TensorFlow installation")
endif()

# Check if TensorFlow directory exists
if(NOT EXISTS "${TENSORFLOW_DIR}")
    message(FATAL_ERROR "TensorFlow directory not found at ${TENSORFLOW_DIR}")
endif()

# Set include directories
set(TensorFlow_INCLUDE_DIR "${TENSORFLOW_DIR}/include")
if(NOT EXISTS "${TensorFlow_INCLUDE_DIR}")
    message(FATAL_ERROR "TensorFlow include directory not found at ${TensorFlow_INCLUDE_DIR}")
endif()

# Set library directories
set(TensorFlow_LIBRARY_DIR "${TENSORFLOW_DIR}/lib")
if(NOT EXISTS "${TensorFlow_LIBRARY_DIR}")
    message(FATAL_ERROR "TensorFlow library directory not found at ${TensorFlow_LIBRARY_DIR}")
endif()

# Find TensorFlow libraries
find_library(TensorFlow_CC_LIBRARY
    NAMES libtensorflow_cc.so libtensorflow_cc.so.2
    PATHS "${TensorFlow_LIBRARY_DIR}"
    NO_DEFAULT_PATH
)

find_library(TensorFlow_FRAMEWORK_LIBRARY
    NAMES libtensorflow_framework.so libtensorflow_framework.so.2
    PATHS "${TensorFlow_LIBRARY_DIR}"
    NO_DEFAULT_PATH
)

if(NOT TensorFlow_CC_LIBRARY)
    message(FATAL_ERROR "TensorFlow C++ library not found in ${TensorFlow_LIBRARY_DIR}")
endif()

if(NOT TensorFlow_FRAMEWORK_LIBRARY)
    message(FATAL_ERROR "TensorFlow framework library not found in ${TensorFlow_LIBRARY_DIR}")
endif()

# Set TensorFlow variables
set(TensorFlow_FOUND TRUE)
set(TensorFlow_INCLUDE_DIRS "${TensorFlow_INCLUDE_DIR}")
set(TensorFlow_LIBRARIES "${TensorFlow_CC_LIBRARY}" "${TensorFlow_FRAMEWORK_LIBRARY}")

# Add library directory to link directories
link_directories("${TensorFlow_LIBRARY_DIR}")

message(STATUS "TensorFlow found:")
message(STATUS "  Include dirs: ${TensorFlow_INCLUDE_DIRS}")
message(STATUS "  Libraries: ${TensorFlow_LIBRARIES}") 
