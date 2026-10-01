#pragma once
// The sequential graph executor.
//
// A Model binds one loaded graph to one concrete device and one concrete set of
// input shapes. Construction resolves shapes, plans the arena, and reserves it
// once; Run then executes the nodes in order through the device's kernel table,
// binding graph inputs and initializers to their storage and intermediates to
// the single arena. Later runs reuse the same arena.
//
// Must not include or reference the backend abstraction layer.

#include "engine/Device.hpp"
#include "engine/Graph.hpp"
#include "engine/Plan.hpp"
#include "engine/Shapes.hpp"

#include <map>
#include <string>
#include <vector>

namespace engine {

// The caller-visible result of one Run: every declared graph output, flattened
// to a float32 vector and keyed by tensor name.
struct InferenceResult {
    std::map<std::string, std::vector<float>> outputs;
};

// One graph bound to a device and a concrete input shape. The arena is
// allocated exactly once, at construction, and released in the destructor.
class Model {
  public:
    Model(const Graph& graph, const ShapeMap& input_dims, const Device& device);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Graph& graph() const { return graph_; }
    const InferredShapes& shapes() const { return shapes_; }
    const MemoryPlan& plan() const { return plan_; }

    // Execute the graph for `inputs`, keyed by graph input name. Throws
    // InferenceException when an input is missing or mis-sized, when a node's op
    // has no kernel on the device, or when a node output has no arena buffer.
    InferenceResult Run(const std::map<std::string, std::vector<float>>& inputs);

  private:
    const Graph& graph_; // borrowed; must outlive the model
    Device* device_;     // the device chosen once for this graph
    InferredShapes shapes_;
    MemoryPlan plan_;
    void* arena_ = nullptr; // one allocation, shared by every arena tensor
};

} // namespace engine
