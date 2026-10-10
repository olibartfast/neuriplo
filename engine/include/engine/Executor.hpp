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

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace engine {

// One host-side tensor crossing the typed Run boundary: its element type, its
// shape, and its elements as contiguous native-endian bytes. Float32 takes 4
// bytes per element, Int64 8, and Bool 1 (0 or 1; any non-zero byte reads as
// true on input, and outputs are written as 0 or 1).
struct HostTensor {
    DataType dtype = DataType::Float32;
    std::vector<int64_t> dims;
    std::vector<uint8_t> bytes;
};

using HostTensorMap = std::map<std::string, HostTensor>;

// Bytes per element of a graph I/O dtype (Float32, Int64, Bool); 0 otherwise.
size_t HostElementSize(DataType dtype);

// The caller-visible result of one float Run: every declared graph output,
// flattened to a float32 vector and keyed by tensor name.
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
    // Float-only entry point: every graph input and output must be float32; a
    // graph with a non-float input or output throws InferenceException naming
    // the tensor.
    InferenceResult Run(const std::map<std::string, std::vector<float>>& inputs);

    // Typed entry point: each input's dtype and dims must match the graph
    // input's inferred dtype and shape, and its byte count must equal the
    // element count times HostElementSize(dtype). Returns every declared graph
    // output in its own dtype, keyed by name. Throws InferenceException on a
    // missing, mistyped, or mis-sized input, and as Run does otherwise.
    HostTensorMap RunTyped(const HostTensorMap& inputs);

  private:
    // Execute the nodes with the graph inputs already bound in `views`, then
    // bind initializers and node outputs; on return `views` resolves every
    // graph output (arena-backed views stay valid until the next run).
    void Execute(std::map<std::string, TensorView>& views);

    // The view bound to graph output `name` after Execute.
    static const TensorView& OutputView(const std::map<std::string, TensorView>& views, const std::string& name);

    const Graph& graph_; // borrowed; must outlive the model
    Device* device_;     // the device chosen once for this graph
    InferredShapes shapes_;
    MemoryPlan plan_;
    void* arena_ = nullptr; // one allocation, shared by every arena tensor
};

} // namespace engine
