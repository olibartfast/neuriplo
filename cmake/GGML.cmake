set(GGML_SOURCES
${INFER_ROOT}/ggml/src/GGMLInfer.cpp
# Add more GGML source files here if needed
)

list(APPEND SOURCES ${GGML_SOURCES})

add_compile_definitions(USE_GGML)
