#pragma once
// Arena memory planning over a loaded graph and its inferred shapes.
//
// Liveness-based linear scan: every node output that is neither a graph input
// nor an embedded constant gets a byte range in one contiguous arena. Tensors
// whose live ranges do not intersect may share memory, so the arena is smaller
// than the sum of the tensors it holds. The plan is deterministic and neither
// the graph nor the shapes are mutated.
//
// Must not include anything from the backend abstraction layer.

#include "engine/Graph.hpp"
#include "engine/Shapes.hpp"

#include <cstdint>
#include <map>
#include <string>

namespace engine {

// Where one arena tensor lives: a byte offset and a byte size.
struct BufferAssignment {
    int64_t offset = 0;
    int64_t size = 0;
};

// The whole arena plan: one buffer per live arena tensor, keyed by tensor name,
// and the total number of bytes the arena must reserve.
struct MemoryPlan {
    std::map<std::string, BufferAssignment> buffers;
    int64_t arena_size = 0;
};

// Assign arena offsets to every live node output in `graph`, using the concrete
// shapes in `shapes`. Throws ModelLoadException when a node output has no
// inferred shape, when an inferred size is non-positive, or when the graph has
// no nodes.
MemoryPlan PlanMemory(const Graph& graph, const InferredShapes& shapes);

} // namespace engine
