# NEURIPLO_ENGINE backend module — the first-party engine, no external SDK.
#
# The engine is built as a standalone static library (neuriplo_engine) so it
# stays decoupled from the backend abstraction layer; the adapter in
# backends/neuriplo-engine/src is the only place the two meet and is compiled into the
# neuriplo target alongside the other backend sources.
#
# The adapter's link against neuriplo_engine is attached in
# cmake/LinkBackend.cmake's NEURIPLO_ENGINE branch: the neuriplo target is created after
# this module is included, so target_link_libraries() cannot run here.

set(NEURIPLO_ENGINE_SOURCES
    ${INFER_ROOT}/neuriplo-engine/src/NeuriploEngineInfer.cpp
)

# Append NEURIPLO_ENGINE sources to the main sources
list(APPEND SOURCES ${NEURIPLO_ENGINE_SOURCES})

# Add compile definition to indicate NEURIPLO_ENGINE backend usage. BackendRuntimeRegistry
# uses this to include and register NeuriploEngineRuntimeFactory.
add_compile_definitions(USE_NEURIPLO_ENGINE)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../engine" engine)

# The adapter is compiled into the shared neuriplo target, so the static engine
# archive it links must be position-independent. Set on the target here rather
# than in engine/CMakeLists.txt: the engine stays standalone-usable, and this
# property only matters when a shared consumer links it.
set_target_properties(neuriplo_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
