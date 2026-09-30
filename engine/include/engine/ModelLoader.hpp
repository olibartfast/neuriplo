#pragma once
// Reads an ONNX model file into the engine's graph.
//
// The single entry point for model loading. The protobuf wire decoding is an
// implementation detail behind this interface, and parsing is hand-written, so
// the engine pulls in no protobuf or ONNX library.
//
// Must not include anything from the backend abstraction layer.

#include <string>

#include "engine/Graph.hpp"

namespace engine {

// Parse the ONNX model at `path` into the Graph IR.
//
// Throws ModelLoadException when the file cannot be read, the wire format is
// malformed, or the model is structurally unusable. Semantic rejections
// (unsupported op, unsupported attribute, unresolvable dynamic shape) name the
// offending node and op type in the message.
Graph LoadGraphFromFile(const std::string& path);

} // namespace engine
