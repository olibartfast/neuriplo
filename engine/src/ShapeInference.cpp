// Static shape inference: derive a concrete TensorInfo for every tensor a
// graph references, from a concrete shape for each declared input.
//
// The walk trusts nothing but the graph's nodes and initializers. The optional
// value_info table (Graph::tensors) is ignored for intermediates, so a graph
// with no recorded intermediate shapes still infers fully. Compute tensors are
// Float32 throughout; Int64 appears only as the dtype of constant shape/axes
// operands.

#include "engine/Shapes.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace engine {
namespace {

std::string quote(const std::string& s) { return "'" + s + "'"; }

// A node-local shape failure: names the node and op type, per the loader's
// rejection contract.
[[noreturn]] void fail(const Node& node, const std::string& why)
{
    throw ModelLoadException("shape inference on node " +
        (node.name.empty() ? std::string("<unnamed>") : quote(node.name)) +
        " op_type " + quote(node.op_type) + ": " + why);
}

const Attribute* find_attr(const Node& node, const std::string& name)
{
    for (const Attribute& a : node.attributes) {
        if (a.name == name) {
            return &a;
        }
    }
    return nullptr;
}

int64_t attr_int(const Node& node, const std::string& name, int64_t fallback)
{
    const Attribute* a = find_attr(node, name);
    if (a == nullptr) {
        return fallback;
    }
    const int64_t* v = std::get_if<int64_t>(&a->value);
    if (v == nullptr) {
        fail(node, "attribute " + quote(name) + " is not an int");
    }
    return *v;
}

std::string attr_str(const Node& node, const std::string& name,
    const std::string& fallback)
{
    const Attribute* a = find_attr(node, name);
    if (a == nullptr) {
        return fallback;
    }
    const std::string* v = std::get_if<std::string>(&a->value);
    if (v == nullptr) {
        fail(node, "attribute " + quote(name) + " is not a string");
    }
    return *v;
}

std::vector<int64_t> attr_ints(const Node& node, const std::string& name)
{
    const Attribute* a = find_attr(node, name);
    if (a == nullptr) {
        return {};
    }
    const std::vector<int64_t>* v = std::get_if<std::vector<int64_t>>(&a->value);
    if (v == nullptr) {
        fail(node, "attribute " + quote(name) + " is not an int list");
    }
    return *v;
}

int64_t element_count(const std::vector<int64_t>& dims)
{
    int64_t count = 1;
    for (const int64_t d : dims) {
        count *= d;
    }
    return count;
}

// NumPy multidirectional broadcast: right-align, every axis equal or one.
std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
    const std::vector<int64_t>& b, const Node& node)
{
    const size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t ad = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        const int64_t bd = (i < b.size()) ? b[b.size() - 1 - i] : 1;
        if (ad != bd && ad != 1 && bd != 1) {
            fail(node, "operands are not broadcast-compatible");
        }
        out[rank - 1 - i] = std::max(ad, bd);
    }
    return out;
}

// ONNX MatMul, including 1-D operand promotion and batch broadcasting.
std::vector<int64_t> matmul_shape(std::vector<int64_t> a,
    std::vector<int64_t> b, const Node& node)
{
    if (a.empty() || b.empty()) {
        fail(node, "MatMul operands must have rank >= 1");
    }
    const bool a_was_1d = a.size() == 1;
    const bool b_was_1d = b.size() == 1;
    if (a_was_1d) {
        a.insert(a.begin(), 1);
    }
    if (b_was_1d) {
        b.push_back(1);
    }
    const int64_t m = a[a.size() - 2];
    const int64_t k_lhs = a[a.size() - 1];
    const int64_t k_rhs = b[b.size() - 2];
    const int64_t n = b[b.size() - 1];
    if (k_lhs != k_rhs) {
        fail(node, "MatMul inner dimensions differ");
    }
    const std::vector<int64_t> batch_lhs(a.begin(), a.end() - 2);
    const std::vector<int64_t> batch_rhs(b.begin(), b.end() - 2);
    std::vector<int64_t> out = broadcast_shape(batch_lhs, batch_rhs, node);
    out.push_back(m);
    out.push_back(n);
    if (a_was_1d) {
        out.erase(out.end() - 2);
    }
    if (b_was_1d) {
        out.pop_back();
    }
    return out;
}

std::vector<int64_t> gemm_shape(const TensorInfo& a, const TensorInfo& b,
    const Node& node)
{
    if (a.dims.size() != 2 || b.dims.size() != 2) {
        fail(node, "Gemm operands must be rank 2");
    }
    std::vector<int64_t> lhs = a.dims;
    std::vector<int64_t> rhs = b.dims;
    if (attr_int(node, "transA", 0) != 0) {
        std::swap(lhs[0], lhs[1]);
    }
    if (attr_int(node, "transB", 0) != 0) {
        std::swap(rhs[0], rhs[1]);
    }
    if (lhs[1] != rhs[0]) {
        fail(node, "Gemm inner dimensions differ");
    }
    return {lhs[0], rhs[1]};
}

// One spatial axis of a pooling/convolution window: the ONNX output formula,
// with ceil instead of floor when `ceil_mode` is set.
int64_t window_output(int64_t in, int64_t kernel, int64_t dilation,
    int64_t pad_begin, int64_t pad_end, int64_t stride, bool ceil_mode,
    const Node& node, const char* what)
{
    const int64_t effective = dilation * (kernel - 1) + 1;
    const int64_t numer = in + pad_begin + pad_end - effective;
    if (numer < 0) {
        fail(node, std::string(what) + " window is larger than padded input");
    }
    const int64_t out =
        (ceil_mode ? (numer + stride - 1) / stride : numer / stride) + 1;
    if (out <= 0) {
        fail(node, std::string(what) + " produces a non-positive output dim");
    }
    return out;
}

// Resolve per-axis begin/end padding from auto_pad, or the explicit pads list.
void resolve_pads(const Node& node, const std::string& auto_pad,
    const std::vector<int64_t>& in_spatial,
    const std::vector<int64_t>& kernel,
    const std::vector<int64_t>& dilations,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads_attr, std::vector<int64_t>& pad_begin,
    std::vector<int64_t>& pad_end)
{
    const size_t rank = in_spatial.size();
    pad_begin.assign(rank, 0);
    pad_end.assign(rank, 0);

    if (auto_pad == "VALID") {
        return;
    }
    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        for (size_t i = 0; i < rank; ++i) {
            const int64_t out = (in_spatial[i] + strides[i] - 1) / strides[i];
            const int64_t effective = dilations[i] * (kernel[i] - 1) + 1;
            int64_t total = (out - 1) * strides[i] + effective - in_spatial[i];
            if (total < 0) {
                total = 0;
            }
            if (auto_pad == "SAME_UPPER") {
                pad_begin[i] = total / 2;
                pad_end[i] = total - pad_begin[i];
            } else {
                pad_end[i] = total / 2;
                pad_begin[i] = total - pad_end[i];
            }
        }
        return;
    }

    // NOTSET (or an empty value): use the explicit pads, defaulting to zero.
    if (pads_attr.empty()) {
        return;
    }
    if (pads_attr.size() != 2 * rank) {
        fail(node, "pads must have two entries per spatial dimension");
    }
    for (size_t i = 0; i < rank; ++i) {
        pad_begin[i] = pads_attr[i];
        pad_end[i] = pads_attr[rank + i];
    }
}

TensorInfo conv_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.size() < 2 || inputs[0] == nullptr || inputs[1] == nullptr) {
        fail(node, "Conv requires an input and a weight");
    }
    const std::vector<int64_t>& x = inputs[0]->dims;
    const std::vector<int64_t>& w = inputs[1]->dims;
    if (x.size() != 4 || w.size() != 4) {
        fail(node, "Conv input and weight must be rank 4");
    }
    const int64_t group = attr_int(node, "group", 1);
    if (group <= 0) {
        fail(node, "Conv group must be positive");
    }
    if (x[1] % group != 0) {
        fail(node, "Conv input channels are not divisible by group");
    }
    if (w[1] != x[1] / group) {
        fail(node, "Conv weight channel dim does not match input/group");
    }

    const std::vector<int64_t> in_spatial = {x[2], x[3]};
    const std::vector<int64_t> kernel = {w[2], w[3]};
    std::vector<int64_t> strides = attr_ints(node, "strides");
    std::vector<int64_t> dilations = attr_ints(node, "dilations");
    if (strides.empty()) {
        strides = {1, 1};
    }
    if (dilations.empty()) {
        dilations = {1, 1};
    }
    if (strides.size() != 2 || dilations.size() != 2) {
        fail(node, "Conv stride/dilation must have two spatial entries");
    }
    std::vector<int64_t> pad_begin;
    std::vector<int64_t> pad_end;
    resolve_pads(node, attr_str(node, "auto_pad", "NOTSET"), in_spatial,
        kernel, dilations, strides, attr_ints(node, "pads"), pad_begin, pad_end);

    std::vector<int64_t> out = {x[0], w[0]};
    for (size_t i = 0; i < 2; ++i) {
        out.push_back(window_output(in_spatial[i], kernel[i], dilations[i],
            pad_begin[i], pad_end[i], strides[i], false, node, "Conv"));
    }

    if (inputs.size() > 2 && inputs[2] != nullptr) {
        const std::vector<int64_t>& bias = inputs[2]->dims;
        if (bias.size() != 1 || bias[0] != w[0]) {
            fail(node, "Conv bias must be rank-1 of length M");
        }
    }
    return TensorInfo{DataType::Float32, out};
}

TensorInfo maxpool_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "MaxPool requires an input");
    }
    const std::vector<int64_t>& x = inputs[0]->dims;
    if (x.size() != 4) {
        fail(node, "MaxPool input must be rank 4 (NCHW)");
    }
    const std::vector<int64_t> kernel = attr_ints(node, "kernel_shape");
    if (kernel.empty()) {
        fail(node, "MaxPool kernel_shape is required");
    }
    if (kernel.size() != 2) {
        fail(node, "MaxPool kernel_shape must have two spatial entries");
    }
    const std::vector<int64_t> in_spatial = {x[2], x[3]};
    std::vector<int64_t> strides = attr_ints(node, "strides");
    std::vector<int64_t> dilations = attr_ints(node, "dilations");
    if (strides.empty()) {
        strides = {1, 1};
    }
    if (dilations.empty()) {
        dilations = {1, 1};
    }
    if (strides.size() != 2 || dilations.size() != 2) {
        fail(node, "MaxPool stride/dilation must have two spatial entries");
    }
    const bool ceil_mode = attr_int(node, "ceil_mode", 0) != 0;
    std::vector<int64_t> pad_begin;
    std::vector<int64_t> pad_end;
    resolve_pads(node, attr_str(node, "auto_pad", "NOTSET"), in_spatial,
        kernel, dilations, strides, attr_ints(node, "pads"), pad_begin, pad_end);

    std::vector<int64_t> out = {x[0], x[1]};
    for (size_t i = 0; i < 2; ++i) {
        out.push_back(window_output(in_spatial[i], kernel[i], dilations[i],
            pad_begin[i], pad_end[i], strides[i], ceil_mode, node, "MaxPool"));
    }
    return TensorInfo{DataType::Float32, out};
}

// Read an Int64 constant operand (a shape or axes table). Anything else cannot
// be resolved statically.
std::vector<int64_t> int64_constant(const Graph& graph, const Node& node,
    const std::string& name)
{
    const auto it = graph.initializers.find(name);
    if (it == graph.initializers.end() ||
        it->second.dtype != DataType::Int64) {
        fail(node, "input " + quote(name) + " must be an Int64 constant");
    }
    const std::vector<int64_t>* values =
        std::get_if<std::vector<int64_t>>(&it->second.values);
    if (values == nullptr) {
        fail(node, "input " + quote(name) + " is not an Int64 constant");
    }
    return *values;
}

TensorInfo reducemean_shape(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "ReduceMean requires an input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    const bool keepdims = attr_int(node, "keepdims", 1) != 0;
    const bool noop_empty = attr_int(node, "noop_with_empty_axes", 0) != 0;

    std::vector<int64_t> axes;
    const bool axes_supplied =
        node.inputs.size() > 1 && !node.inputs[1].empty();
    if (axes_supplied) {
        axes = int64_constant(graph, node, node.inputs[1]);
    }

    std::set<int64_t> seen;
    for (int64_t& axis : axes) {
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            fail(node, "ReduceMean axis is out of range");
        }
        if (!seen.insert(axis).second) {
            fail(node, "ReduceMean axes must be unique");
        }
    }

    std::vector<int64_t> out = in;
    if (axes.empty()) {
        if (noop_empty) {
            return TensorInfo{DataType::Float32, out};
        }
        for (int64_t i = 0; i < rank; ++i) {
            axes.push_back(i);
        }
    }
    std::sort(axes.begin(), axes.end());
    if (keepdims) {
        for (const int64_t axis : axes) {
            out[static_cast<size_t>(axis)] = 1;
        }
    } else {
        for (auto it = axes.rbegin(); it != axes.rend(); ++it) {
            out.erase(out.begin() + *it);
        }
    }
    return TensorInfo{DataType::Float32, out};
}

TensorInfo reshape_shape(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Reshape requires an input");
    }
    if (node.inputs.size() < 2 || node.inputs[1].empty()) {
        fail(node, "Reshape requires a shape input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const std::vector<int64_t> target =
        int64_constant(graph, node, node.inputs[1]);
    const bool allowzero = attr_int(node, "allowzero", 0) != 0;

    std::vector<int64_t> out(target.size(), 0);
    int64_t inferred = -1;
    for (size_t i = 0; i < target.size(); ++i) {
        const int64_t v = target[i];
        if (v == 0 && !allowzero) {
            if (i >= in.size()) {
                fail(node, "Reshape 0 has no matching input dimension");
            }
            out[i] = in[i];
        } else if (v == -1) {
            if (inferred >= 0) {
                fail(node, "Reshape allows at most one -1");
            }
            inferred = static_cast<int64_t>(i);
        } else if (v < 0) {
            fail(node, "Reshape target dimensions must be positive");
        } else {
            out[i] = v;
        }
    }

    const int64_t in_count = element_count(in);
    int64_t known = 1;
    for (size_t i = 0; i < out.size(); ++i) {
        if (static_cast<int64_t>(i) == inferred) {
            continue;
        }
        known *= out[i];
    }
    if (inferred >= 0) {
        if (known == 0 || in_count % known != 0) {
            fail(node, "Reshape target product does not match input elements");
        }
        out[static_cast<size_t>(inferred)] = in_count / known;
    }

    if (element_count(out) != in_count) {
        fail(node, "Reshape target product does not match input elements");
    }
    return TensorInfo{DataType::Float32, out};
}

std::vector<TensorInfo> compute_outputs(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    const std::string& op = node.op_type;
    auto need = [&node, &inputs](size_t index) -> const TensorInfo& {
        if (index >= inputs.size() || inputs[index] == nullptr) {
            fail(node, "missing required input");
        }
        return *inputs[index];
    };

    if (op == "Add") {
        return {TensorInfo{DataType::Float32,
            broadcast_shape(need(0).dims, need(1).dims, node)}};
    }
    if (op == "Relu") {
        return {TensorInfo{DataType::Float32, need(0).dims}};
    }
    if (op == "MatMul") {
        return {TensorInfo{DataType::Float32,
            matmul_shape(need(0).dims, need(1).dims, node)}};
    }
    if (op == "Gemm") {
        return {TensorInfo{DataType::Float32, gemm_shape(need(0), need(1), node)}};
    }
    if (op == "Conv") {
        return {conv_shape(node, inputs)};
    }
    if (op == "MaxPool") {
        return {maxpool_shape(node, inputs)};
    }
    if (op == "ReduceMean") {
        return {reducemean_shape(graph, node, inputs)};
    }
    if (op == "Reshape") {
        return {reshape_shape(graph, node, inputs)};
    }
    fail(node, "unsupported op_type");
}

} // namespace

InferredShapes InferShapes(const Graph& graph, const ShapeMap& input_dims)
{
    InferredShapes result;

    for (const auto& entry : graph.inputs) {
        const std::string& name = entry.first;
        const auto it = input_dims.find(name);
        if (it == input_dims.end()) {
            throw ModelLoadException(
                "shape inference: input " + quote(name) +
                " is missing from the provided input shapes");
        }
        for (const int64_t dim : it->second) {
            if (dim <= 0) {
                throw ModelLoadException(
                    "shape inference: input " + quote(name) +
                    " has non-positive dimension " + std::to_string(dim));
            }
        }
        result.tensors[name] = TensorInfo{DataType::Float32, it->second};
    }

    for (const auto& init : graph.initializers) {
        result.tensors[init.first] =
            TensorInfo{init.second.dtype, init.second.dims};
    }

    for (const Node& node : graph.nodes) {
        std::vector<const TensorInfo*> inputs;
        inputs.reserve(node.inputs.size());
        for (const std::string& name : node.inputs) {
            if (name.empty()) {
                inputs.push_back(nullptr);
                continue;
            }
            const auto it = result.tensors.find(name);
            if (it == result.tensors.end()) {
                fail(node, "input " + quote(name) + " has no inferred shape");
            }
            inputs.push_back(&it->second);
        }

        const std::vector<TensorInfo> outputs =
            compute_outputs(graph, node, inputs);
        if (outputs.size() != node.outputs.size()) {
            fail(node, "output arity mismatch");
        }
        for (size_t i = 0; i < node.outputs.size(); ++i) {
            result.tensors[node.outputs[i]] = outputs[i];
        }
    }

    for (const auto& entry : graph.outputs) {
        if (result.tensors.find(entry.first) == result.tensors.end()) {
            throw ModelLoadException("shape inference: graph output " +
                quote(entry.first) + " has no inferred shape");
        }
    }

    return result;
}

} // namespace engine
