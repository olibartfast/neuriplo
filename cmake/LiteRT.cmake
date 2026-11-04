# LiteRT Configuration

message(STATUS "LiteRT version: ${LITERT_VERSION}")
message(STATUS "LiteRT directory: ${LITERT_DIR}")

set(LITERT_LIBRARY "${LITERT_DIR}/lib/libtensorflow-lite.so")
file(GLOB LITERT_DEPENDENCY_LIBRARIES "${LITERT_DIR}/lib/*.so*" "${LITERT_DIR}/lib/*.a")
list(REMOVE_ITEM LITERT_DEPENDENCY_LIBRARIES "${LITERT_LIBRARY}" "${LITERT_DIR}/lib/libtensorflowlite.so")
# Delegate archives refer back to runtime symbols; keep the runtime in the rescan group.
set(LITERT_LINK_LIBRARIES "-Wl,--start-group" "${LITERT_LIBRARY}" ${LITERT_DEPENDENCY_LIBRARIES} "-Wl,--end-group")

set(LITERT_SOURCES
    ${INFER_ROOT}/litert/src/LiteRTInfer.cpp
)

list(APPEND SOURCES ${LITERT_SOURCES})

add_compile_definitions(USE_LITERT)
