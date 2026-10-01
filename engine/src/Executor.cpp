// The sequential executor (T-16).
//
// Construction runs shape inference and memory planning for the supplied input
// shapes, then reserves the whole arena in one allocation from the device's
// allocator. Run allocates no arena memory: it resolves a TensorView for every
// graph input, initializer, and node output, walks the nodes in order, and
// dispatches each to the device kernel table. The planner's inclusive liveness
// guarantees that no node output shares bytes with any input of its defining
// node, so Run binds each tensor's view directly — no transient copies. Graph
// outputs are copied out of the arena, so a second Run observes the same plan
// and the same arena.

#include "engine/Executor.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace engine {
namespace {

// Product of a shape's dimensions. An empty shape is a scalar: one element.
int64_t ElementCount(const std::vector<int64_t>& dims) {
    int64_t count = 1;
    for (const int64_t dim : dims) {
        count *= dim;
    }
    return count;
}

// The inferred shape for `name`, or a failure naming the tensor.
const TensorInfo& ShapeOf(const InferredShapes& shapes, const std::string& name) {
    const auto it = shapes.tensors.find(name);
    if (it == shapes.tensors.end()) {
        throw InferenceException("executor: tensor '" + name + "' has no inferred shape");
    }
    return it->second;
}

} // namespace

Model::Model(const Graph& graph, const ShapeMap& input_dims, const Device& device)
    : graph_(graph), device_(const_cast<Device*>(&device)), shapes_(InferShapes(graph, input_dims)),
      plan_(PlanMemory(graph, shapes_)) {
    if (plan_.arena_size > 0) {
        arena_ = device_->allocator().allocate(plan_.arena_size);
        if (arena_ == nullptr) {
            throw InferenceException("executor: could not allocate the " + std::to_string(plan_.arena_size) +
                                     "-byte arena");
        }
    }
}

Model::~Model() {
    if (arena_ != nullptr) {
        device_->allocator().release(arena_);
    }
}

InferenceResult Model::Run(const std::map<std::string, std::vector<float>>& inputs) {
    // Every named value resolves to one borrowed view. Graph inputs point at the
    // caller's vectors and initializers at the graph's stored constants; node
    // outputs point into the arena.
    std::map<std::string, TensorView> views;

    for (const auto& entry : graph_.inputs) {
        const std::string& name = entry.first;
        const auto provided = inputs.find(name);
        if (provided == inputs.end()) {
            throw InferenceException("executor: missing input '" + name + "'");
        }
        const TensorInfo& info = ShapeOf(shapes_, name);
        const int64_t expected = ElementCount(info.dims);
        if (static_cast<int64_t>(provided->second.size()) != expected) {
            throw InferenceException("executor: input '" + name + "' has " + std::to_string(provided->second.size()) +
                                     " values, expected " + std::to_string(expected));
        }
        TensorView view;
        view.data = const_cast<float*>(provided->second.data());
        view.dtype = info.dtype;
        view.dims = info.dims;
        views.emplace(name, std::move(view));
    }

    for (const auto& entry : graph_.initializers) {
        const Initializer& init = entry.second;
        TensorView view;
        view.dtype = init.dtype;
        view.dims = init.dims;
        if (init.dtype == DataType::Float32) {
            const auto* values = std::get_if<std::vector<float>>(&init.values);
            if (values == nullptr) {
                throw InferenceException("executor: initializer '" + entry.first + "' has no float values");
            }
            view.data = const_cast<float*>(values->data());
        } else if (init.dtype == DataType::Int64) {
            const auto* values = std::get_if<std::vector<int64_t>>(&init.values);
            if (values == nullptr) {
                throw InferenceException("executor: initializer '" + entry.first + "' has no int64 values");
            }
            view.data = const_cast<int64_t*>(values->data());
        } else {
            throw InferenceException("executor: initializer '" + entry.first + "' has an unsupported dtype");
        }
        views.emplace(entry.first, std::move(view));
    }

    for (const Node& node : graph_.nodes) {
        const KernelFn kernel = device_->kernels().find(node.op_type);
        if (kernel == nullptr) {
            throw InferenceException("executor: no kernel for op '" + node.op_type + "' (node '" + node.name + "')");
        }

        std::vector<TensorView> node_inputs;
        node_inputs.reserve(node.inputs.size());
        for (const std::string& name : node.inputs) {
            if (name.empty()) {
                node_inputs.push_back(TensorView{}); // an omitted optional operand
                continue;
            }
            const auto it = views.find(name);
            if (it == views.end()) {
                throw InferenceException("executor: node '" + node.name + "' input '" + name + "' is not available");
            }
            node_inputs.push_back(it->second);
        }

        std::vector<TensorView> node_outputs;
        node_outputs.reserve(node.outputs.size());
        for (const std::string& name : node.outputs) {
            const auto buffer = plan_.buffers.find(name);
            if (buffer == plan_.buffers.end()) {
                throw InferenceException("executor: no arena buffer for output '" + name + "' of node '" + node.name +
                                         "'");
            }
            const TensorInfo& info = ShapeOf(shapes_, name);
            TensorView view;
            view.data = static_cast<char*>(arena_) + buffer->second.offset;
            view.dtype = info.dtype;
            view.dims = info.dims;
            views[name] = view;
            node_outputs.push_back(std::move(view));
        }

        kernel(node, node_inputs, node_outputs);
    }

    InferenceResult result;
    for (const auto& entry : graph_.outputs) {
        const std::string& name = entry.first;
        const auto it = views.find(name);
        if (it == views.end()) {
            throw InferenceException("executor: graph output '" + name + "' was never produced");
        }
        const TensorView& view = it->second;
        if (view.dtype != DataType::Float32) {
            throw InferenceException("executor: graph output '" + name + "' is not float32");
        }
        const int64_t count = ElementCount(view.dims);
        const float* src = static_cast<const float*>(view.data);
        result.outputs.emplace(name, std::vector<float>(src, src + count));
    }
    return result;
}

} // namespace engine
