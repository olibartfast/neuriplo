set(GGML_SOURCES
${INFER_ROOT}/ggml/src/GGMLInfer.cpp
${INFER_ROOT}/ggml/src/GGMLOps.cpp
# GGUF-only: no ONNX wrapper (GGMLGraph/GGMLOnnx excluded from the backend).
# Add more GGML source files here if needed
)

list(APPEND SOURCES ${GGML_SOURCES})

add_compile_definitions(USE_GGML)
