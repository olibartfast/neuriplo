set(CACTUS_SOURCES
${INFER_ROOT}/cactus/src/CactusInfer.cpp
)

list(APPEND SOURCES ${CACTUS_SOURCES})

add_compile_definitions(USE_CACTUS)
