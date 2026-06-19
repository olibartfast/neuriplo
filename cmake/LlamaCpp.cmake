set(LLAMACPP_SOURCES
${INFER_ROOT}/llamacpp/src/LlamaCppInfer.cpp
)

list(APPEND SOURCES ${LLAMACPP_SOURCES})

add_compile_definitions(USE_LLAMACPP)
