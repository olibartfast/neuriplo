// Load-time constant folding: see ConstantFolding.hpp.
//
// Each candidate node is shaped through InferShapes on a one-node graph made
// of the node and its constant inputs, so folding reuses the same shape rules
// as a normal load instead of duplicating them.

#include "engine/ConstantFolding.hpp"

#include "engine/Shapes.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace engine {
namespace {

// Upper bound on one folded tensor, so folding never materializes a large
// broadcast that the executor would otherwise have computed into the arena.
constexpr int64_t kMaxFoldedElements = int64_t{1} << 22;

int64_t ElementCount(const std::vector<int64_t>& dims) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        count *= dim;
    }
    return count;
}

bool FoldNode(Graph& graph, const Node& node, const Device& device) {
    const KernelFn kernel = device.kernels().find(node.op_type);
    if (kernel == nullptr) {
        return false;
    }

    Graph single;
    single.nodes.push_back(node);
    for (const std::string& name : node.inputs) {
        if (!name.empty()) {
            single.initializers[name] = graph.initializers.at(name);
        }
    }
    InferredShapes shapes;
    try {
        shapes = InferShapes(single, {});
    } catch (const ModelLoadException&) {
        return false;
    }

    std::vector<TensorView> inputs;
    for (const std::string& name : node.inputs) {
        TensorView view;
        if (!name.empty()) {
            Initializer& init = single.initializers.at(name);
            view.dtype = init.dtype;
            view.dims = init.dims;
            if (init.dtype == DataType::Float32) {
                view.data = std::get<std::vector<float>>(init.values).data();
            } else {
                view.data = std::get<std::vector<int64_t>>(init.values).data();
            }
        }
        inputs.push_back(std::move(view));
    }

    std::vector<Initializer> results;
    std::vector<TensorView> outputs;
    for (const std::string& name : node.outputs) {
        const TensorInfo& info = shapes.tensors.at(name);
        if (info.dtype != DataType::Float32 && info.dtype != DataType::Int64) {
            return false;
        }
        const int64_t count = ElementCount(info.dims);
        if (count <= 0 || count > kMaxFoldedElements) {
            return false;
        }
        Initializer init;
        init.dtype = info.dtype;
        init.dims = info.dims;
        if (info.dtype == DataType::Float32) {
            init.values = std::vector<float>(static_cast<size_t>(count));
        } else {
            init.values = std::vector<int64_t>(static_cast<size_t>(count));
        }
        results.push_back(std::move(init));
    }
    for (size_t i = 0; i < results.size(); ++i) {
        TensorView view;
        view.dtype = results[i].dtype;
        view.dims = results[i].dims;
        if (results[i].dtype == DataType::Float32) {
            view.data = std::get<std::vector<float>>(results[i].values).data();
        } else {
            view.data = std::get<std::vector<int64_t>>(results[i].values).data();
        }
        outputs.push_back(std::move(view));
    }

    try {
        kernel(node, inputs, outputs);
    } catch (const InferenceException&) {
        return false;
    }
    for (size_t i = 0; i < node.outputs.size(); ++i) {
        graph.initializers[node.outputs[i]] = std::move(results[i]);
    }
    return true;
}

} // namespace

int FoldConstants(Graph& graph, const Device& device) {
    std::set<std::string> graph_outputs;
    for (const auto& output : graph.outputs) {
        graph_outputs.insert(output.first);
    }

    int folded = 0;
    std::vector<Node> kept;
    kept.reserve(graph.nodes.size());
    for (Node& node : graph.nodes) {
        bool candidate = !node.outputs.empty();
        bool any_input = false;
        for (const std::string& name : node.inputs) {
            if (name.empty()) {
                continue;
            }
            any_input = true;
            if (graph.initializers.find(name) == graph.initializers.end()) {
                candidate = false;
                break;
            }
        }
        for (const std::string& name : node.outputs) {
            if (name.empty() || graph_outputs.count(name) != 0) {
                candidate = false;
            }
        }
        if (candidate && any_input && FoldNode(graph, node, device)) {
            ++folded;
            continue;
        }
        kept.push_back(std::move(node));
    }
    graph.nodes = std::move(kept);
    return folded;
}

} // namespace engine
