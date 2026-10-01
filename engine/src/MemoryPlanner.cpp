// Liveness analysis and arena memory planning (T-12).
//
// The planner is the memory half of the per-shape seam: it reads a loaded
// Graph and an InferredShapes value and produces a MemoryPlan, mutating
// neither. A tensor's lifetime is the INCLUSIVE node range [definition, last
// use]: it stays live through the node that consumes it, so a value read by
// node j is still live while node j runs. A graph output's last use is
// nodes.size() - 1. Two lifetimes overlap when their closed intervals
// intersect (a.begin <= b.end && b.begin <= a.end).
//
// This yields the safety invariant the executor relies on: no node output may
// share bytes with any input of its defining node, so a read-then-write kernel
// never has its input overwritten by its own output. Buffers are placed by a
// deterministic linear scan: definitions in ascending node order, names
// ascending within a node, each taking the lowest 64-byte-aligned offset that
// avoids every already-placed tensor whose live range overlaps its own.
// arena_size is the smallest 64-byte multiple covering every assignment.

#include "engine/Plan.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace engine {
namespace {

constexpr int64_t kAlignment = 64;

int64_t align_up(int64_t value, int64_t alignment) {
    const int64_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

int64_t dtype_bytes(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return 4;
    case DataType::Int64:
        return 8;
    case DataType::Bool:
        return 1;
    case DataType::Unknown:
        return 0;
    }
    return 0;
}

// A tensor's live range as a closed node-index interval [begin, end].
struct Interval {
    int64_t begin = 0;
    int64_t end = 0;
};

bool intervals_intersect(const Interval& a, const Interval& b) { return a.begin <= b.end && b.begin <= a.end; }

struct ArenaTensor {
    std::string name;
    int64_t definition = 0;
    Interval live;
    int64_t size = 0;
};

struct PlacedBuffer {
    int64_t offset = 0;
    int64_t size = 0;
    Interval live;
};

bool byte_ranges_overlap(int64_t offset, int64_t size, const PlacedBuffer& other) {
    return offset < other.offset + other.size && other.offset < offset + size;
}

} // namespace

MemoryPlan PlanMemory(const Graph& graph, const InferredShapes& shapes) {
    if (graph.nodes.empty()) {
        throw ModelLoadException("memory plan: cannot plan a graph with no nodes");
    }

    std::set<std::string> external;
    for (const auto& input : graph.inputs) {
        external.insert(input.first);
    }
    for (const auto& initializer : graph.initializers) {
        external.insert(initializer.first);
    }

    std::set<std::string> outputs;
    for (const auto& output : graph.outputs) {
        outputs.insert(output.first);
    }

    // Node outputs that are not external, with the index of the producing node.
    std::map<std::string, int64_t> definitions;
    std::vector<std::string> defined_order;
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        for (const std::string& output : graph.nodes[i].outputs) {
            if (output.empty() || external.count(output) != 0 || definitions.count(output) != 0) {
                continue;
            }
            definitions.emplace(output, static_cast<int64_t>(i));
            defined_order.push_back(output);
        }
    }

    // Greatest later node that consumes each arena tensor.
    std::map<std::string, int64_t> last_uses;
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        for (const std::string& input : graph.nodes[i].inputs) {
            const auto definition = definitions.find(input);
            if (definition == definitions.end()) {
                continue;
            }
            const int64_t index = static_cast<int64_t>(i);
            if (index <= definition->second) {
                continue;
            }
            auto current = last_uses.find(input);
            if (current == last_uses.end() || index > current->second) {
                last_uses[input] = index;
            }
        }
    }

    std::vector<ArenaTensor> tensors;
    for (const std::string& name : defined_order) {
        const auto shape = shapes.tensors.find(name);
        if (shape == shapes.tensors.end()) {
            throw ModelLoadException("memory plan: no inferred shape for tensor '" + name + "'");
        }

        int64_t elements = 1;
        bool non_positive = false;
        for (const int64_t dim : shape->second.dims) {
            if (dim <= 0) {
                non_positive = true;
                break;
            }
            elements *= dim;
        }
        const int64_t size = elements * dtype_bytes(shape->second.dtype);
        if (non_positive || size <= 0) {
            throw ModelLoadException("memory plan: non-positive size for tensor '" + name + "'");
        }

        const int64_t definition = definitions.at(name);
        const bool is_output = outputs.count(name) != 0;
        int64_t end = 0;
        const auto last_use = last_uses.find(name);
        if (is_output) {
            end = static_cast<int64_t>(graph.nodes.size()) - 1;
        } else if (last_use != last_uses.end()) {
            end = last_use->second;
        } else {
            // Never consumed and not an output: nothing to keep alive.
            continue;
        }

        ArenaTensor tensor;
        tensor.name = name;
        tensor.definition = definition;
        tensor.live = Interval{definition, end};
        tensor.size = size;
        tensors.push_back(std::move(tensor));
    }

    std::sort(tensors.begin(), tensors.end(), [](const ArenaTensor& a, const ArenaTensor& b) {
        if (a.definition != b.definition) {
            return a.definition < b.definition;
        }
        return a.name < b.name;
    });

    MemoryPlan plan;
    std::vector<PlacedBuffer> placed;
    int64_t arena_end = 0;
    for (const ArenaTensor& tensor : tensors) {
        int64_t offset = 0;
        for (;;) {
            bool conflict = false;
            for (const PlacedBuffer& other : placed) {
                if (!intervals_intersect(tensor.live, other.live)) {
                    continue;
                }
                if (byte_ranges_overlap(offset, tensor.size, other)) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) {
                break;
            }
            offset += kAlignment;
        }

        plan.buffers[tensor.name] = BufferAssignment{offset, tensor.size};
        placed.push_back(PlacedBuffer{offset, tensor.size, tensor.live});
        arena_end = std::max(arena_end, offset + tensor.size);
    }

    plan.arena_size = align_up(arena_end, kAlignment);
    return plan;
}

} // namespace engine
