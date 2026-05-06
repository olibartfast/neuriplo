# ExecuTorch Backend Configuration

set(EXECUTORCH_DIR "${EXECUTORCH_DIR}" CACHE PATH "ExecuTorch install prefix or package root")

message(STATUS "ExecuTorch version: ${EXECUTORCH_VERSION}")
message(STATUS "ExecuTorch root: ${EXECUTORCH_DIR}")

list(APPEND CMAKE_PREFIX_PATH "${EXECUTORCH_DIR}")
if(NOT TARGET executorch)
    find_package(executorch REQUIRED CONFIG)
endif()

message(STATUS "✓ ExecuTorch found")

set(EXECUTORCH_SOURCES
    ${INFER_ROOT}/executorch/src/ExecuTorchInfer.cpp
)

list(APPEND SOURCES ${EXECUTORCH_SOURCES})

add_compile_definitions(USE_EXECUTORCH)
