# NERT backend module: the experimental first-party engine.
#
# The engine lives in its own repository (https://github.com/olibartfast/nert)
# and is consumed with FetchContent at the commit pinned by NERT_VERSION in
# versions.env. The adapter in backends/nert/src is the only place the engine
# and the backend abstraction meet and is compiled into the neuriplo target
# alongside the other backend sources.
#
# To build against a local checkout instead of fetching (no network needed):
#   cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_NERT=/path/to/nert
#
# The adapter's link against nert::nert is attached in cmake/LinkBackend.cmake's
# NERT branch: the neuriplo target is created after this module is included, so
# target_link_libraries() cannot run here.

include(FetchContent)

FetchContent_Declare(nert
    GIT_REPOSITORY https://github.com/olibartfast/nert.git
    GIT_TAG ${NERT_VERSION}
)

# Neuriplo's tests are not NERT's: NERT's own suites run in its repository.
# NERT also honours BUILD_INFERENCE_ENGINE_TESTS, so shadow it with a normal
# variable while NERT is configured and drop the shadow afterwards.
set(NERT_BUILD_TESTS OFF CACHE BOOL "Build the NERT tests" FORCE)

# Opt-in CUDA device layer for the NERT backend: builds NERT with
# NERT_ENABLE_CUDA=ON (needs nvcc) and marks the adapter/tests with
# NEURIPLO_NERT_CUDA so they may select nert::CudaDevice(). OFF keeps the
# CPU-only closure and the previous behavior byte-identical.
option(NEURIPLO_NERT_CUDA "Build the NERT backend with CUDA device support" OFF)
if(NEURIPLO_NERT_CUDA)
    set(NERT_ENABLE_CUDA ON CACHE BOOL "Build NERT with the CUDA device layer" FORCE)
    add_compile_definitions(NEURIPLO_NERT_CUDA)
endif()
if(DEFINED BUILD_INFERENCE_ENGINE_TESTS)
    set(_neuriplo_nert_saved_tests "${BUILD_INFERENCE_ENGINE_TESTS}")
    set(_neuriplo_nert_had_tests TRUE)
else()
    set(_neuriplo_nert_had_tests FALSE)
endif()
set(BUILD_INFERENCE_ENGINE_TESTS OFF)
FetchContent_MakeAvailable(nert)
if(_neuriplo_nert_had_tests)
    set(BUILD_INFERENCE_ENGINE_TESTS "${_neuriplo_nert_saved_tests}")
else()
    unset(BUILD_INFERENCE_ENGINE_TESTS)
endif()
unset(_neuriplo_nert_saved_tests)
unset(_neuriplo_nert_had_tests)

set(NERT_SOURCES
    ${INFER_ROOT}/nert/src/NertInfer.cpp
)

# Append NERT sources to the main sources
list(APPEND SOURCES ${NERT_SOURCES})

# Add compile definition to indicate NERT backend usage. BackendRuntimeRegistry
# uses this to include and register NertRuntimeFactory.
add_compile_definitions(USE_NERT)

# The adapter is compiled into the shared neuriplo target, so the static NERT
# archive it links must be position-independent. Set on the real target, not on
# an alias.
set_target_properties(nert PROPERTIES POSITION_INDEPENDENT_CODE ON)
