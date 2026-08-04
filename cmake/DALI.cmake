# NVIDIA DALI Backend Configuration
#
# GPU preprocessing (nvJPEG decode, resize, normalize) hosted in-process through
# the DALI C API. Not an inference engine: it fills the same backend slot so a
# serving pipeline can chain GPU preprocessing ahead of a model.
#
# NVIDIA publishes no standalone C++ DALI distribution -- the headers and shared
# libraries ship inside a pip wheel whose filename carries an opaque build
# number, so there is nothing stable to download here. Point the build at an
# extracted copy with -DDALI_DIR=<dir> (the directory holding libdali.so and
# include/dali/c_api.h).

set(DALI_DIR "" CACHE PATH "DALI installation root (contains libdali.so and include/dali)")

if(NOT DALI_DIR OR NOT EXISTS "${DALI_DIR}/include/dali/c_api.h")
    message(FATAL_ERROR
        "DALI backend enabled but DALI_DIR does not point at a DALI distribution.\n"
        "Expected ${DALI_DIR}/include/dali/c_api.h.\n"
        "Extract the nvidia-dali wheel (or copy the libraries out of a Triton "
        "container) and pass -DDALI_DIR=<dir>.")
endif()

foreach(_dali_lib dali dali_operators)
    if(NOT EXISTS "${DALI_DIR}/lib${_dali_lib}.so")
        message(FATAL_ERROR "DALI backend requires ${DALI_DIR}/lib${_dali_lib}.so")
    endif()
endforeach()

message(STATUS "DALI version: ${DALI_VERSION}")
message(STATUS "DALI root: ${DALI_DIR}")
message(STATUS "✓ DALI found")

set(DALI_SOURCES
    ${INFER_ROOT}/dali/src/DALIInfer.cpp
)

list(APPEND SOURCES ${DALI_SOURCES})

add_compile_definitions(USE_DALI)
