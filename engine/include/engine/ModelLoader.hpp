#pragma once
// ONE isolating interface between the hand-written protobuf wire reader and
// the rest of the engine ([D-7], Group 2a). The definition arrives with
// Group 2b; this header is the contract the rest of the engine depends on.
//
// Boundary contract ([R-1]): no backend-layer includes; the only dependency
// beyond stdlib is nothing — parsing is hand-written ([D-7]).

#include <string>

#include "engine/Graph.hpp"

namespace engine {

// Parse the ONNX model at `path` into the Graph IR.
//
// Throws ModelLoadException when the file cannot be read, the protobuf
// wire format is truncated or malformed, or the model is structurally
// unusable. (Graph-level semantic rejections — unsupported ops, unsupported
// attribute combinations, unresolvable dynamic dimensions — are raised with
// the node name and op type embedded in the message, per [V-3].)
Graph LoadGraphFromFile(const std::string& path);

} // namespace engine
