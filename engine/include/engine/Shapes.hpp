#pragma once
// Static shape inference over a loaded graph.
//
// Takes the graph and a concrete shape for every declared input, and derives
// the type and shape of every tensor the graph references, without mutating
// the graph. This is the shape half of the per-shape seam: calling it again
// with different input dims is the only way a caller changes the resolved
// shapes, so the parser, the IR, and (later) the kernels stay untouched.
//
// Must not include anything from the backend abstraction layer.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "engine/Graph.hpp"

namespace engine {

// Concrete dimensions for the graph inputs, keyed by input name.
using ShapeMap = std::map<std::string, std::vector<int64_t>>;

// The resolved type-and-shape table for every tensor InferShapes reaches.
struct InferredShapes {
    std::map<std::string, TensorInfo> tensors;
};

// Derive a concrete TensorInfo for every graph input, initializer, node output,
// and declared output. Throws ModelLoadException when an input is missing from
// `input_dims` or carries a non-positive dimension, when an operator cannot be
// shaped statically, or when a node input or declared output never resolves.
InferredShapes InferShapes(const Graph& graph, const ShapeMap& input_dims);

} // namespace engine
