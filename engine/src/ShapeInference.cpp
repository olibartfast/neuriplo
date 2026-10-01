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
#include <cmath>
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

// Read a Float32 constant operand (Resize scales). Anything else cannot be
// resolved statically.
std::vector<float> float32_constant(const Graph& graph, const Node& node,
    const std::string& name)
{
    const auto it = graph.initializers.find(name);
    if (it == graph.initializers.end() ||
        it->second.dtype != DataType::Float32) {
        fail(node, "input " + quote(name) + " must be a Float32 constant");
    }
    const std::vector<float>* values =
        std::get_if<std::vector<float>>(&it->second.values);
    if (values == nullptr) {
        fail(node, "input " + quote(name) + " is not a Float32 constant");
    }
    return *values;
}

// Normalize an axis in [-rank, rank) to [0, rank).
int64_t normalize_axis(const Node& node, int64_t axis, int64_t rank,
    const char* what)
{
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        fail(node, std::string(what) + " axis is out of range");
    }
    return axis;
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

TensorInfo flatten_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Flatten requires an input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    int64_t axis = attr_int(node, "axis", 1);
    if (axis < 0) {
        axis += rank + 1;
    }
    if (axis < 0 || axis > rank) {
        fail(node, "Flatten axis is out of range");
    }
    int64_t first = 1;
    int64_t second = 1;
    for (int64_t i = 0; i < axis; ++i) {
        first *= in[static_cast<size_t>(i)];
    }
    for (size_t i = static_cast<size_t>(axis); i < in.size(); ++i) {
        second *= in[i];
    }
    return TensorInfo{DataType::Float32, {first, second}};
}

TensorInfo concat_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (node.outputs.size() != 1) {
        fail(node, "Concat must produce exactly one output");
    }
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Concat requires at least one input");
    }
    const int64_t rank = static_cast<int64_t>(inputs[0]->dims.size());
    const int64_t axis =
        normalize_axis(node, attr_int(node, "axis", 1), rank, "Concat");
    std::vector<int64_t> out = inputs[0]->dims;
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i] == nullptr) {
            fail(node, "Concat requires all inputs");
        }
        const std::vector<int64_t>& dims = inputs[i]->dims;
        if (static_cast<int64_t>(dims.size()) != rank) {
            fail(node, "Concat inputs must share rank");
        }
        for (int64_t j = 0; j < rank; ++j) {
            if (j == axis) {
                continue;
            }
            if (dims[static_cast<size_t>(j)] != out[static_cast<size_t>(j)]) {
                fail(node, "Concat inputs differ off the concat axis");
            }
        }
        out[static_cast<size_t>(axis)] += dims[static_cast<size_t>(axis)];
    }
    return TensorInfo{DataType::Float32, out};
}

std::vector<TensorInfo> split_shapes(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Split requires an input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    const int64_t axis =
        normalize_axis(node, attr_int(node, "axis", 0), rank, "Split");
    const size_t count = node.outputs.size();
    if (count == 0) {
        fail(node, "Split must produce at least one output");
    }
    std::vector<int64_t> parts;
    if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
        parts = int64_constant(graph, node, node.inputs[1]);
        if (parts.size() != count) {
            fail(node, "Split split sizes must match the output count");
        }
        int64_t sum = 0;
        for (const int64_t v : parts) {
            if (v < 0) {
                fail(node, "Split split sizes must be non-negative");
            }
            sum += v;
        }
        if (sum != in[static_cast<size_t>(axis)]) {
            fail(node, "Split split sizes must sum to the axis dim");
        }
    } else {
        if (in[static_cast<size_t>(axis)] % static_cast<int64_t>(count) != 0) {
            fail(node,
                "Split axis dim is not divisible by the output count");
        }
        parts.assign(count,
            in[static_cast<size_t>(axis)] / static_cast<int64_t>(count));
    }
    std::vector<TensorInfo> outputs;
    outputs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        std::vector<int64_t> dims = in;
        dims[static_cast<size_t>(axis)] = parts[i];
        outputs.push_back(TensorInfo{DataType::Float32, dims});
    }
    return outputs;
}

TensorInfo unsqueeze_shape(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Unsqueeze requires an input");
    }
    if (node.inputs.size() < 2 || node.inputs[1].empty()) {
        fail(node, "Unsqueeze requires an axes input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    std::vector<int64_t> axes = int64_constant(graph, node, node.inputs[1]);
    const int64_t rank_out =
        static_cast<int64_t>(in.size() + axes.size());
    std::set<int64_t> seen;
    for (int64_t& axis : axes) {
        if (axis < 0) {
            axis += rank_out;
        }
        if (axis < 0 || axis >= rank_out) {
            fail(node, "Unsqueeze axis is out of range");
        }
        if (!seen.insert(axis).second) {
            fail(node, "Unsqueeze axes must be unique");
        }
    }
    std::vector<int64_t> out(static_cast<size_t>(rank_out), 0);
    size_t data_pos = 0;
    for (int64_t i = 0; i < rank_out; ++i) {
        if (seen.count(i) != 0) {
            out[static_cast<size_t>(i)] = 1;
        } else {
            out[static_cast<size_t>(i)] = in[data_pos++];
        }
    }
    return TensorInfo{DataType::Float32, out};
}

TensorInfo expand_shape(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Expand requires an input");
    }
    if (node.inputs.size() < 2 || node.inputs[1].empty()) {
        fail(node, "Expand requires a shape input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const std::vector<int64_t> target =
        int64_constant(graph, node, node.inputs[1]);
    const size_t rank = std::max(in.size(), target.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t di =
            static_cast<int64_t>(in.size()) - static_cast<int64_t>(rank) +
            static_cast<int64_t>(i);
        const int64_t si = static_cast<int64_t>(target.size()) -
            static_cast<int64_t>(rank) + static_cast<int64_t>(i);
        const int64_t d = di < 0 ? 1 : in[static_cast<size_t>(di)];
        if (si < 0) {
            out[i] = d;
            continue;
        }
        const int64_t s = target[static_cast<size_t>(si)];
        if (s < 1) {
            fail(node, "Expand target shape must be positive");
        }
        if (d == s || di < 0) {
            out[i] = s;
        } else if (d == 1) {
            out[i] = s;
        } else {
            fail(node, "Expand input is not broadcastable to the target shape");
        }
    }
    return TensorInfo{DataType::Float32, out};
}

TensorInfo transpose_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Transpose requires an input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    std::vector<int64_t> perm = attr_ints(node, "perm");
    if (perm.empty()) {
        std::vector<int64_t> out(in.rbegin(), in.rend());
        return TensorInfo{DataType::Float32, out};
    }
    if (static_cast<int64_t>(perm.size()) != rank) {
        fail(node, "Transpose perm rank must match the input rank");
    }
    std::vector<int64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (int64_t i = 0; i < rank; ++i) {
        if (sorted[static_cast<size_t>(i)] != i) {
            fail(node, "Transpose perm must be a permutation");
        }
    }
    std::vector<int64_t> out(static_cast<size_t>(rank), 0);
    for (int64_t i = 0; i < rank; ++i) {
        out[static_cast<size_t>(i)] = in[static_cast<size_t>(perm[i])];
    }
    return TensorInfo{DataType::Float32, out};
}

TensorInfo gatherelements_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.size() < 2 || inputs[0] == nullptr || inputs[1] == nullptr) {
        fail(node, "GatherElements requires data and indices inputs");
    }
    const int64_t rank = static_cast<int64_t>(inputs[0]->dims.size());
    (void)normalize_axis(node, attr_int(node, "axis", 1), rank,
        "GatherElements");
    if (static_cast<int64_t>(inputs[1]->dims.size()) != rank) {
        fail(node, "GatherElements data and indices must share rank");
    }
    return TensorInfo{DataType::Float32, inputs[1]->dims};
}

TensorInfo gather_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.size() < 2 || inputs[0] == nullptr || inputs[1] == nullptr) {
        fail(node, "Gather requires data and indices inputs");
    }
    const std::vector<int64_t>& data = inputs[0]->dims;
    const std::vector<int64_t>& indices = inputs[1]->dims;
    const int64_t rank = static_cast<int64_t>(data.size());
    const int64_t axis =
        normalize_axis(node, attr_int(node, "axis", 0), rank, "Gather");
    std::vector<int64_t> out;
    for (int64_t i = 0; i < axis; ++i) {
        out.push_back(data[static_cast<size_t>(i)]);
    }
    out.insert(out.end(), indices.begin(), indices.end());
    for (size_t i = static_cast<size_t>(axis) + 1; i < data.size(); ++i) {
        out.push_back(data[i]);
    }
    return TensorInfo{DataType::Float32, out};
}

TensorInfo cast_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Cast requires an input");
    }
    const int64_t to = attr_int(node, "to", 1);
    if (to == 1) {
        return TensorInfo{DataType::Float32, inputs[0]->dims};
    }
    if (to == 7) {
        return TensorInfo{DataType::Int64, inputs[0]->dims};
    }
    fail(node, "Cast 'to' names an unsupported element type");
}

TensorInfo resize_shape(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Resize requires an input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    const bool has_scales = node.inputs.size() > 2 && !node.inputs[2].empty();
    const bool has_sizes = node.inputs.size() > 3 && !node.inputs[3].empty();
    if (has_scales && has_sizes) {
        fail(node, "Resize takes scales or sizes, not both");
    }
    if (has_sizes) {
        const std::vector<int64_t> sizes =
            int64_constant(graph, node, node.inputs[3]);
        if (static_cast<int64_t>(sizes.size()) != rank) {
            fail(node, "Resize sizes rank must match the input rank");
        }
        for (const int64_t v : sizes) {
            if (v <= 0) {
                fail(node, "Resize sizes must be positive");
            }
        }
        return TensorInfo{DataType::Float32, sizes};
    }
    if (has_scales) {
        const std::vector<float> scales =
            float32_constant(graph, node, node.inputs[2]);
        if (static_cast<int64_t>(scales.size()) != rank) {
            fail(node, "Resize scales rank must match the input rank");
        }
        std::vector<int64_t> out(static_cast<size_t>(rank), 0);
        for (int64_t i = 0; i < rank; ++i) {
            const int64_t dim = static_cast<int64_t>(std::floor(
                static_cast<double>(in[static_cast<size_t>(i)]) *
                static_cast<double>(scales[static_cast<size_t>(i)])));
            if (dim <= 0) {
                fail(node, "Resize produces a non-positive output dim");
            }
            out[static_cast<size_t>(i)] = dim;
        }
        return TensorInfo{DataType::Float32, out};
    }
    fail(node, "Resize requires a scales or sizes input");
}

TensorInfo slice_shape(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Slice requires a data input");
    }
    if (node.inputs.size() < 3 || node.inputs[1].empty() ||
        node.inputs[2].empty()) {
        fail(node, "Slice requires data, starts, and ends inputs");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    const std::vector<int64_t> starts =
        int64_constant(graph, node, node.inputs[1]);
    const std::vector<int64_t> ends =
        int64_constant(graph, node, node.inputs[2]);
    if (starts.size() != ends.size()) {
        fail(node, "Slice starts and ends must have the same length");
    }
    const size_t count = starts.size();
    std::vector<int64_t> axes(count, 0);
    for (size_t i = 0; i < count; ++i) {
        axes[i] = static_cast<int64_t>(i);
    }
    if (node.inputs.size() > 3 && !node.inputs[3].empty()) {
        axes = int64_constant(graph, node, node.inputs[3]);
        if (axes.size() != count) {
            fail(node, "Slice axes must match starts in length");
        }
    }
    std::vector<int64_t> steps(count, 1);
    if (node.inputs.size() > 4 && !node.inputs[4].empty()) {
        steps = int64_constant(graph, node, node.inputs[4]);
        if (steps.size() != count) {
            fail(node, "Slice steps must match starts in length");
        }
    }
    std::vector<int64_t> out = in;
    for (size_t i = 0; i < count; ++i) {
        if (steps[i] <= 0) {
            fail(node, "Slice steps must be positive");
        }
        const int64_t axis = normalize_axis(node, axes[i], rank, "Slice");
        const int64_t dim = in[static_cast<size_t>(axis)];
        int64_t begin = starts[i];
        int64_t end = ends[i];
        if (begin < 0) {
            begin += dim;
        }
        if (end < 0) {
            end += dim;
        }
        begin = std::max<int64_t>(0, std::min(begin, dim));
        end = std::max<int64_t>(0, std::min(end, dim));
        out[static_cast<size_t>(axis)] =
            end > begin ? (end - begin + steps[i] - 1) / steps[i] : 0;
    }
    return TensorInfo{DataType::Float32, out};
}

std::vector<TensorInfo> topk_shapes(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "TopK requires an input");
    }
    if (node.outputs.size() != 2) {
        fail(node, "TopK must produce values and indices");
    }
    if (node.inputs.size() < 2 || node.inputs[1].empty()) {
        fail(node, "TopK requires a K input");
    }
    const std::vector<int64_t> k_vals =
        int64_constant(graph, node, node.inputs[1]);
    if (k_vals.size() != 1) {
        fail(node, "TopK K input must be a scalar Int64 constant");
    }
    const int64_t k = k_vals.front();
    if (k <= 0) {
        fail(node, "TopK K must be positive");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t axis = normalize_axis(node, attr_int(node, "axis", -1),
        static_cast<int64_t>(in.size()), "TopK");
    if (k > in[static_cast<size_t>(axis)]) {
        fail(node, "TopK K exceeds the axis dim");
    }
    std::vector<int64_t> out = in;
    out[static_cast<size_t>(axis)] = k;
    return {TensorInfo{DataType::Float32, out},
        TensorInfo{DataType::Float32, out}};
}

TensorInfo constantofshape_shape(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr || node.inputs[0].empty()) {
        fail(node, "ConstantOfShape requires a shape input");
    }
    if (inputs[0]->dims.size() != 1) {
        fail(node, "ConstantOfShape shape input must be rank 1");
    }
    const std::vector<int64_t> shape =
        int64_constant(graph, node, node.inputs[0]);
    for (const int64_t v : shape) {
        if (v < 0) {
            fail(node, "ConstantOfShape shape values must be non-negative");
        }
    }
    return TensorInfo{DataType::Int64, shape};
}

TensorInfo shape_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Shape requires an input");
    }
    return TensorInfo{DataType::Int64,
        {static_cast<int64_t>(inputs[0]->dims.size())}};
}

TensorInfo reducemax_shape(const Graph& graph, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "ReduceMax requires an input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    const bool keepdims = attr_int(node, "keepdims", 1) != 0;

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
            fail(node, "ReduceMax axis is out of range");
        }
        if (!seen.insert(axis).second) {
            fail(node, "ReduceMax axes must be unique");
        }
    }

    std::vector<int64_t> out = in;
    if (axes.empty()) {
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

    if (op == "Add" || op == "Mul" || op == "Div" || op == "Sub") {
        return {TensorInfo{DataType::Float32,
            broadcast_shape(need(0).dims, need(1).dims, node)}};
    }
    if (op == "Mod") {
        const TensorInfo& lhs = need(0);
        const TensorInfo& rhs = need(1);
        if (lhs.dtype != rhs.dtype) {
            fail(node, "Mod inputs must share dtype");
        }
        return {TensorInfo{lhs.dtype,
            broadcast_shape(lhs.dims, rhs.dims, node)}};
    }
    if (op == "Relu" || op == "Sigmoid" || op == "Softmax") {
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
    if (op == "Flatten") {
        return {flatten_shape(node, inputs)};
    }
    if (op == "Concat") {
        return {concat_shape(node, inputs)};
    }
    if (op == "Split") {
        return split_shapes(graph, node, inputs);
    }
    if (op == "Unsqueeze") {
        return {unsqueeze_shape(graph, node, inputs)};
    }
    if (op == "Expand") {
        return {expand_shape(graph, node, inputs)};
    }
    if (op == "Transpose") {
        return {transpose_shape(node, inputs)};
    }
    if (op == "GatherElements") {
        return {gatherelements_shape(node, inputs)};
    }
    if (op == "Gather") {
        return {gather_shape(node, inputs)};
    }
    if (op == "Cast") {
        return {cast_shape(node, inputs)};
    }
    if (op == "Resize") {
        return {resize_shape(graph, node, inputs)};
    }
    if (op == "Slice") {
        return {slice_shape(graph, node, inputs)};
    }
    if (op == "TopK") {
        return topk_shapes(graph, node, inputs);
    }
    if (op == "ConstantOfShape") {
        return {constantofshape_shape(graph, node, inputs)};
    }
    if (op == "Equal") {
        return {TensorInfo{DataType::Bool,
            broadcast_shape(need(0).dims, need(1).dims, node)}};
    }
    if (op == "Where") {
        if (inputs.size() < 3 || inputs[0] == nullptr ||
            inputs[1] == nullptr || inputs[2] == nullptr) {
            fail(node, "Where requires condition, X, and Y inputs");
        }
        return {TensorInfo{DataType::Float32,
            broadcast_shape(broadcast_shape(inputs[0]->dims, inputs[1]->dims,
                                node),
                inputs[2]->dims, node)}};
    }
    if (op == "Shape") {
        return {shape_shape(node, inputs)};
    }
    if (op == "ReduceMax") {
        return {reducemax_shape(graph, node, inputs)};
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
