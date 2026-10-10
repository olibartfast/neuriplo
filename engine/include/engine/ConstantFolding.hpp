#pragma once
// Load-time constant folding.
//
// Exporters leave subgraphs whose every input is a constant: anchor grids,
// resize targets computed from fixed shapes, scalar arithmetic. Folding runs
// each such node once with the device's kernels and replaces its outputs with
// initializers, so shape inference sees constants instead of expressions it
// cannot evaluate, and the executor never re-runs them.
//
// Must not include anything from the backend abstraction layer.

#include "engine/Device.hpp"
#include "engine/Graph.hpp"

namespace engine {

// Folds, in execution order, every node whose inputs are all initializers and
// whose outputs are float32 or int64 tensors that are not graph outputs. A node
// that cannot be shaped or run at load is left in place, so it fails (or runs)
// exactly as it would have without the pass. Returns the number of folded
// nodes.
int FoldConstants(Graph& graph, const Device& device);

} // namespace engine
