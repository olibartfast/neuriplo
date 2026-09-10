# Guard: the public headers must compile without OpenCV.
#
# OpenCV is a dependency of the opencv-dnn backend alone. Nothing in include/,
# and nothing in the backend-agnostic part of backends/src/, may include an
# <opencv2/...> header or rely on OpenCV to drag in a std header.
#
# This is a standalone project on purpose. It never configures the root
# CMakeLists, so it never runs find_package(OpenCV) and cannot pick up OpenCV's
# include directories by accident.
#
#   cmake -S test/public-headers -B build-headers
#   cmake --build build-headers

cmake_minimum_required(VERSION 3.10)
project(neuriplo_public_headers CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NEURIPLO_PROBE_REQUIRE_OPENCV_ABSENT
       "Fail if <opencv2/dnn.hpp> is reachable without an explicit include path" ON)

find_package(Glog REQUIRED)

set(NEURIPLO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)

# Negative control. Without it this check passes trivially on any machine where
# OpenCV merely happens to be uninstalled, and would keep passing even if a
# header started including OpenCV again.
if(NEURIPLO_PROBE_REQUIRE_OPENCV_ABSENT)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/opencv_reachable.cpp
         "#include <opencv2/dnn.hpp>\nint main() { return 0; }\n")
    try_compile(OPENCV_REACHABLE
        ${CMAKE_CURRENT_BINARY_DIR}/negative_control
        ${CMAKE_CURRENT_BINARY_DIR}/opencv_reachable.cpp
        CXX_STANDARD 17)
    if(OPENCV_REACHABLE)
        message(FATAL_ERROR
            "<opencv2/dnn.hpp> is on the default include path here, so this "
            "guard cannot prove anything. Re-run with "
            "-DNEURIPLO_PROBE_REQUIRE_OPENCV_ABSENT=OFF to skip the check.")
    endif()
    message(STATUS "Negative control: OpenCV is not on the default include path")
endif()

add_library(public_headers_probe STATIC probe.cpp)

target_include_directories(public_headers_probe PRIVATE
    ${NEURIPLO_ROOT}/include
    ${NEURIPLO_ROOT}/backends/src
)

target_link_libraries(public_headers_probe PRIVATE glog::glog)

if(MSVC)
    target_compile_options(public_headers_probe PRIVATE /permissive- /W4)
else()
    target_compile_options(public_headers_probe PRIVATE -Wall -Wextra)
endif()
