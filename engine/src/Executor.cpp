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
#include <cstring>
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

// A dtype's name for error messages.
const char* DataTypeName(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return "float32";
    case DataType::Int64:
        return "int64";
    case DataType::Bool:
        return "bool";
    case DataType::Unknown:
        break;
    }
    return "unknown";
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

size_t HostElementSize(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return sizeof(float);
    case DataType::Int64:
        return sizeof(int64_t);
    case DataType::Bool:
        return sizeof(bool);
    case DataType::Unknown:
        break;
    }
    return 0;
}

InferenceResult Model::Run(const std::map<std::string, std::vector<float>>& inputs) {
    // Graph inputs point at the caller's vectors; Execute binds the rest.
    std::map<std::string, TensorView> views;

    for (const auto& entry : graph_.inputs) {
        const std::string& name = entry.first;
        const TensorInfo& info = ShapeOf(shapes_, name);
        if (info.dtype != DataType::Float32) {
            throw InferenceException("executor: graph input '" + name + "' is " + DataTypeName(info.dtype) +
                                     ", not float32; use RunTyped");
        }
        const auto provided = inputs.find(name);
        if (provided == inputs.end()) {
            throw InferenceException("executor: missing input '" + name + "'");
        }
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

    Execute(views);

    InferenceResult result;
    for (const auto& entry : graph_.outputs) {
        const std::string& name = entry.first;
        const TensorView& view = OutputView(views, name);
        if (view.dtype != DataType::Float32) {
            throw InferenceException("executor: graph output '" + name + "' is not float32");
        }
        const int64_t count = ElementCount(view.dims);
        const float* src = static_cast<const float*>(view.data);
        result.outputs.emplace(name, std::vector<float>(src, src + count));
    }
    return result;
}

HostTensorMap Model::RunTyped(const HostTensorMap& inputs) {
    std::map<std::string, TensorView> views;
    // Bool inputs are normalised to 0/1 bytes so kernels may read them as bool.
    std::map<std::string, std::vector<uint8_t>> bool_inputs;

    for (const auto& entry : graph_.inputs) {
        const std::string& name = entry.first;
        const TensorInfo& info = ShapeOf(shapes_, name);
        const auto provided = inputs.find(name);
        if (provided == inputs.end()) {
            throw InferenceException("executor: missing input '" + name + "'");
        }
        const HostTensor& host = provided->second;
        if (host.dtype != info.dtype) {
            throw InferenceException("executor: input '" + name + "' is " + DataTypeName(host.dtype) + ", expected " +
                                     DataTypeName(info.dtype));
        }
        if (host.dims != info.dims) {
            throw InferenceException("executor: input '" + name + "' has a shape other than the one the model was " +
                                     "built for");
        }
        const size_t expected = static_cast<size_t>(ElementCount(info.dims)) * HostElementSize(info.dtype);
        if (host.bytes.size() != expected) {
            throw InferenceException("executor: input '" + name + "' has " + std::to_string(host.bytes.size()) +
                                     " bytes, expected " + std::to_string(expected));
        }
        TensorView view;
        view.dtype = info.dtype;
        view.dims = info.dims;
        if (info.dtype == DataType::Bool) {
            std::vector<uint8_t>& normalised = bool_inputs[name];
            normalised.resize(host.bytes.size());
            for (size_t i = 0; i < host.bytes.size(); ++i) {
                normalised[i] = host.bytes[i] != 0 ? 1U : 0U;
            }
            view.data = normalised.data();
        } else {
            view.data = const_cast<uint8_t*>(host.bytes.data());
        }
        views.emplace(name, std::move(view));
    }

    Execute(views);

    HostTensorMap result;
    for (const auto& entry : graph_.outputs) {
        const std::string& name = entry.first;
        const TensorView& view = OutputView(views, name);
        const size_t element_size = HostElementSize(view.dtype);
        if (element_size == 0) {
            throw InferenceException("executor: graph output '" + name + "' has no host dtype");
        }
        HostTensor host;
        host.dtype = view.dtype;
        host.dims = view.dims;
        host.bytes.resize(static_cast<size_t>(ElementCount(view.dims)) * element_size);
        if (view.dtype == DataType::Bool) {
            const bool* src = static_cast<const bool*>(view.data);
            for (size_t i = 0; i < host.bytes.size(); ++i) {
                host.bytes[i] = src[i] ? 1U : 0U;
            }
        } else if (!host.bytes.empty()) {
            std::memcpy(host.bytes.data(), view.data, host.bytes.size());
        }
        result.emplace(name, std::move(host));
    }
    return result;
}

const TensorView& Model::OutputView(const std::map<std::string, TensorView>& views, const std::string& name) {
    const auto it = views.find(name);
    if (it == views.end()) {
        throw InferenceException("executor: graph output '" + name + "' was never produced");
    }
    return it->second;
}

void Model::Execute(std::map<std::string, TensorView>& views) {
    for (const auto& entry : graph_.initializers) {
        const Initializer& init = entry.second;
        TensorView view;
        view.dtype = init.dtype;
        view.dims = init.dims;
        view.data = InitializerData(const_cast<Initializer&>(init));
        // An empty initializer (an omitted Resize roi, say) has no storage.
        if (view.data == nullptr && ElementCount(init.dims) != 0) {
            throw InferenceException("executor: initializer '" + entry.first + "' has no values of its dtype");
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
}

} // namespace engine
