# ExecuTorch Backend Configuration

set(EXECUTORCH_DIR "${EXECUTORCH_DIR}" CACHE PATH "ExecuTorch install prefix or package root")

# ExecuTorch delegates are chosen at .pte export time and consumed by the
# runtime; the C++ side's job is to link the matching backend library so the
# delegate self-registers. `xnnpack` is the default: it ships inside the
# ExecuTorch source tree (no external SDK) and the test model is exported with
# the XNNPACK partitioner. `portable` links only the CPU portable kernels.
# SDK-backed delegates (qnn, vulkan, coreml, mps, ethos-u) are intentionally not
# wired here; each pulls a new external dependency and needs explicit sign-off.
set(EXECUTORCH_DELEGATE "xnnpack" CACHE STRING
    "ExecuTorch delegate backend to link: xnnpack (default) or portable")
set_property(CACHE EXECUTORCH_DELEGATE PROPERTY STRINGS xnnpack portable)

set(_ET_VALID_DELEGATES xnnpack portable)
if(NOT EXECUTORCH_DELEGATE IN_LIST _ET_VALID_DELEGATES)
    message(FATAL_ERROR
        "Unsupported EXECUTORCH_DELEGATE '${EXECUTORCH_DELEGATE}'. "
        "Valid values: ${_ET_VALID_DELEGATES}")
endif()

message(STATUS "ExecuTorch version: ${EXECUTORCH_VERSION}")
message(STATUS "ExecuTorch root: ${EXECUTORCH_DIR}")
message(STATUS "ExecuTorch delegate: ${EXECUTORCH_DELEGATE}")

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

# Pass the configured delegate to the runtime so it can cross-check against the
# backends ExecuTorch actually has registered and emit a clear diagnostic.
add_compile_definitions(NEURIPLO_EXECUTORCH_DELEGATE="${EXECUTORCH_DELEGATE}")
