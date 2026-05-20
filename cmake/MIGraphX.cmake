# MIGraphX Backend Configuration
# MIGraphX ships with ROCm; the canonical way to consume it is find_package(migraphx).
# Set CMAKE_PREFIX_PATH to /opt/rocm (or MIGRAPHX_ROOT) so CMake can find its config.

set(MIGRAPHX_ROOT "/opt/rocm" CACHE PATH "ROCm installation root (contains MIGraphX)")

message(STATUS "MIGraphX version: ${MIGRAPHX_VERSION}")
message(STATUS "MIGraphX root: ${MIGRAPHX_ROOT}")

list(APPEND CMAKE_PREFIX_PATH "${MIGRAPHX_ROOT}")
find_package(migraphx REQUIRED)

message(STATUS "✓ MIGraphX found")

set(MIGRAPHX_SOURCES
    ${INFER_ROOT}/migraphx/src/MIGraphXInfer.cpp
)

list(APPEND SOURCES ${MIGRAPHX_SOURCES})

add_compile_definitions(USE_MIGRAPHX)
