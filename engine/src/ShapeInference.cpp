// Static shape inference: derive a concrete TensorInfo for every tensor a
// graph references, from a concrete shape for each declared input.
//
// The walk trusts nothing but the graph's nodes and initializers. The optional
// value_info table (Graph::tensors) is ignored for intermediates, so a graph
// with no recorded intermediate shapes still infers fully. Compute tensors
// are Float32 or Int64 (Int64 flows through Add/Sub/Mul/Div/Where and the
// shape-parameter expressions they form); Bool appears for comparison
// outputs. A shape input with no initializer is resolved by the static
// evaluator below over Shape, Gather, Add, Sub, Mul, Div, Equal, Where, and
// ConstantOfShape.

#include "engine/Shapes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
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

// ConstantOfShape's output element type follows its `value` attribute: INT64,
// FLOAT, or BOOL; an absent `value` means float32 zero (opset 18). Any other
// attribute kind is a node-local failure.
DataType constantofshape_dtype(const Node& node)
{
    const Attribute* attr = find_attr(node, "value");
    if (attr == nullptr || std::holds_alternative<float>(attr->value)) {
        return DataType::Float32;
    }
    if (std::holds_alternative<int64_t>(attr->value)) {
        return DataType::Int64;
    }
    if (std::holds_alternative<bool>(attr->value)) {
        return DataType::Bool;
    }
    fail(node, "ConstantOfShape 'value' must be an INT64, FLOAT, or BOOL scalar");
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

// Shape's [start, end) dim range (opset 15): negatives count from the back
// and both clamp to [0, rank].
std::pair<int64_t, int64_t> shape_range(const Node& node, int64_t rank)
{
    const auto clamp = [rank](int64_t v) {
        v = v < 0 ? v + rank : v;
        return std::min(std::max(v, int64_t{0}), rank);
    };
    const int64_t start = clamp(attr_int(node, "start", 0));
    const int64_t end = std::max(start, clamp(attr_int(node, "end", rank)));
    return {start, end};
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
    if (x.size() < 3 || x.size() > 5 || w.size() != x.size()) {
        fail(node, "Conv input and weight must share rank 3, 4, or 5");
    }
    const size_t spatial = x.size() - 2;
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

    const std::vector<int64_t> in_spatial(x.begin() + 2, x.end());
    const std::vector<int64_t> kernel(w.begin() + 2, w.end());
    std::vector<int64_t> strides = attr_ints(node, "strides");
    std::vector<int64_t> dilations = attr_ints(node, "dilations");
    if (strides.empty()) {
        strides.assign(spatial, 1);
    }
    if (dilations.empty()) {
        dilations.assign(spatial, 1);
    }
    if (strides.size() != spatial || dilations.size() != spatial) {
        fail(node, "Conv stride/dilation must have one entry per spatial dim");
    }
    std::vector<int64_t> pad_begin;
    std::vector<int64_t> pad_end;
    resolve_pads(node, attr_str(node, "auto_pad", "NOTSET"), in_spatial,
        kernel, dilations, strides, attr_ints(node, "pads"), pad_begin, pad_end);

    std::vector<int64_t> out = {x[0], w[0]};
    for (size_t i = 0; i < spatial; ++i) {
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

// Load-time context for shape-parameter resolution: the graph, the shapes
// inferred so far (topological order holds, so every earlier node output is
// present), the producer of every named tensor, and the index of the node
// currently being shaped.
struct StaticContext {
    const Graph& graph;
    const std::map<std::string, TensorInfo>& inferred;
    const std::map<std::string, size_t>& producers;
    size_t current = 0;
};

// A statically evaluated tensor: int64 shape data, a boolean mask, or a
// float32 scalar/vector (the f64 lane, holding float32-representable values)
// that only Range operands and the Cast round trip use. Anything else fails
// back to the caller's rejection.
struct EvalTensor {
    DataType dtype = DataType::Unknown;
    std::vector<int64_t> dims;
    std::vector<int64_t> i64;
    std::vector<char> boolean;
    std::vector<double> f64;
};

int64_t eval_element_count(const std::vector<int64_t>& dims)
{
    int64_t count = 1;
    for (const int64_t dim : dims) {
        count *= dim;
    }
    return count;
}

std::vector<int64_t> eval_strides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

// NumPy multidirectional broadcast that reports incompatibility instead of
// throwing: the caller maps it to the consumer's existing rejection.
std::optional<std::vector<int64_t>> eval_broadcast(
    const std::vector<int64_t>& a, const std::vector<int64_t>& b)
{
    const size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t ad = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
        const int64_t bd = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
        if (ad != bd && ad != 1 && bd != 1) {
            return std::nullopt;
        }
        out[i] = std::max(ad, bd);
    }
    return out;
}

// Offset of a broadcast operand at a linear index of the broadcast output.
// The caller establishes broadcast compatibility first.
int64_t eval_offset(const std::vector<int64_t>& dims,
    const std::vector<int64_t>& out_dims, int64_t index)
{
    if (out_dims.empty()) {
        return 0;
    }
    const std::vector<int64_t> out_strides = eval_strides(out_dims);
    const std::vector<int64_t> in_strides = eval_strides(dims);
    int64_t offset = 0;
    int64_t remaining = index;
    for (size_t j = 0; j < out_dims.size(); ++j) {
        const int64_t coord = remaining / out_strides[j];
        remaining %= out_strides[j];
        if (j + dims.size() < out_dims.size()) {
            continue; // a missing leading axis broadcasts as size 1
        }
        const size_t k = j - (out_dims.size() - dims.size());
        if (dims[k] == 1) {
            continue; // broadcast
        }
        offset += coord * in_strides[k];
    }
    return offset;
}

// Read an Int64 constant operand (a shape or axes table), evaluating the
// static shape-parameter expression that produces it when no initializer
// carries the name. The evaluator covers exactly Shape, Gather, Add, Sub,
// Mul, Div, Equal, Where, and ConstantOfShape; anything else — an
// out-of-whitelist op, a data-dependent root, or any evaluation failure —
// keeps this function's existing rejection, unchanged.
std::optional<EvalTensor> eval_node(
    const StaticContext& ctx, const Node& node, int depth);

std::optional<EvalTensor> eval_tensor(
    const StaticContext& ctx, const std::string& name, int depth)
{
    if (depth > 64) {
        return std::nullopt;
    }
    const auto init_it = ctx.graph.initializers.find(name);
    if (init_it != ctx.graph.initializers.end() &&
        init_it->second.dtype == DataType::Bool) {
        const auto* values =
            std::get_if<std::vector<uint8_t>>(&init_it->second.values);
        if (values == nullptr) {
            return std::nullopt;
        }
        EvalTensor tensor;
        tensor.dtype = DataType::Bool;
        tensor.dims = init_it->second.dims;
        tensor.boolean.assign(values->begin(), values->end());
        return tensor;
    }
    if (init_it != ctx.graph.initializers.end() &&
        init_it->second.dtype == DataType::Float32) {
        const auto* values =
            std::get_if<std::vector<float>>(&init_it->second.values);
        if (values == nullptr ||
            eval_element_count(init_it->second.dims) !=
                static_cast<int64_t>(values->size())) {
            return std::nullopt;
        }
        EvalTensor tensor;
        tensor.dtype = DataType::Float32;
        tensor.dims = init_it->second.dims;
        tensor.f64.assign(values->begin(), values->end());
        return tensor;
    }
    if (init_it != ctx.graph.initializers.end()) {
        if (init_it->second.dtype != DataType::Int64) {
            return std::nullopt;
        }
        const std::vector<int64_t>* values =
            std::get_if<std::vector<int64_t>>(&init_it->second.values);
        if (values == nullptr) {
            return std::nullopt;
        }
        if (eval_element_count(init_it->second.dims) !=
            static_cast<int64_t>(values->size())) {
            return std::nullopt;
        }
        EvalTensor tensor;
        tensor.dtype = DataType::Int64;
        tensor.dims = init_it->second.dims;
        tensor.i64 = *values;
        return tensor;
    }
    const auto prod_it = ctx.producers.find(name);
    if (prod_it == ctx.producers.end()) {
        return std::nullopt; // graph inputs and unknowns are data-dependent
    }
    if (prod_it->second >= ctx.current) {
        return std::nullopt;
    }
    return eval_node(ctx, ctx.graph.nodes[prod_it->second], depth + 1);
}

// Elementwise int64 evaluation with NumPy broadcasting. Division truncates
// toward zero like the kernel; a zero divisor fails, matching its throw.
std::optional<EvalTensor> eval_int64_binary(const StaticContext& ctx,
    const Node& node, int depth, const std::string& op)
{
    if (node.inputs.size() != 2) {
        return std::nullopt;
    }
    const auto lhs = eval_tensor(ctx, node.inputs[0], depth);
    const auto rhs = eval_tensor(ctx, node.inputs[1], depth);
    if (!lhs.has_value() || !rhs.has_value() ||
        lhs->dtype != DataType::Int64 || rhs->dtype != DataType::Int64) {
        return std::nullopt;
    }
    const auto dims = eval_broadcast(lhs->dims, rhs->dims);
    if (!dims.has_value()) {
        return std::nullopt;
    }
    EvalTensor out;
    out.dtype = DataType::Int64;
    out.dims = *dims;
    out.i64.assign(static_cast<size_t>(eval_element_count(out.dims)), 0);
    for (int64_t i = 0; i < eval_element_count(out.dims); ++i) {
        const size_t at = static_cast<size_t>(i);
        const int64_t a = lhs->i64[static_cast<size_t>(
            eval_offset(lhs->dims, out.dims, i))];
        const int64_t b = rhs->i64[static_cast<size_t>(
            eval_offset(rhs->dims, out.dims, i))];
        if (op == "Div") {
            if (b == 0) {
                return std::nullopt;
            }
            out.i64[at] = a / b;
        } else if (op == "Sub") {
            out.i64[at] = a - b;
        } else if (op == "Mul") {
            out.i64[at] = a * b;
        } else {
            out.i64[at] = a + b;
        }
    }
    return out;
}

std::optional<EvalTensor> eval_node(
    const StaticContext& ctx, const Node& node, int depth)
{
    const std::string& op = node.op_type;
    if (op == "Identity" || op == "Squeeze" || op == "Unsqueeze" ||
        op == "Reshape" || op == "Flatten") {
        // Layout-only: the values pass through in order; the dims are the
        // node's already-inferred output shape.
        if (node.inputs.empty() || node.outputs.size() != 1) {
            return std::nullopt;
        }
        auto value = eval_tensor(ctx, node.inputs[0], depth);
        const auto out_it = ctx.inferred.find(node.outputs[0]);
        if (!value.has_value() || out_it == ctx.inferred.end()) {
            return std::nullopt;
        }
        value->dims = out_it->second.dims;
        return value;
    }
    if (op == "Shape") {
        if (node.inputs.size() != 1) {
            return std::nullopt;
        }
        const auto it = ctx.inferred.find(node.inputs[0]);
        if (it == ctx.inferred.end()) {
            return std::nullopt;
        }
        const std::pair<int64_t, int64_t> range = shape_range(node,
            static_cast<int64_t>(it->second.dims.size()));
        EvalTensor tensor;
        tensor.dtype = DataType::Int64;
        tensor.dims = {range.second - range.first};
        tensor.i64.assign(it->second.dims.begin() + range.first,
            it->second.dims.begin() + range.second);
        return tensor;
    }
    if (op == "Concat") {
        // Rank-1 int64 pieces of a shape vector, joined in order.
        EvalTensor out;
        out.dtype = DataType::Int64;
        for (const std::string& name : node.inputs) {
            const auto piece = eval_tensor(ctx, name, depth);
            if (!piece.has_value() || piece->dtype != DataType::Int64 ||
                piece->dims.size() != 1) {
                return std::nullopt;
            }
            out.i64.insert(out.i64.end(), piece->i64.begin(), piece->i64.end());
        }
        out.dims = {static_cast<int64_t>(out.i64.size())};
        return out;
    }
    if (op == "Cast") {
        const auto value = node.inputs.size() == 1
            ? eval_tensor(ctx, node.inputs[0], depth)
            : std::nullopt;
        const int64_t to = attr_int(node, "to", 0);
        if (!value.has_value() ||
            (to != 7 /*INT64*/ && to != 1 /*FLOAT*/ && to != 9 /*BOOL*/)) {
            return std::nullopt;
        }
        EvalTensor out;
        out.dims = value->dims;
        if (to == 9) {
            // Any nonzero element converts to true.
            if (value->dtype == DataType::Bool) {
                return value;
            }
            out.dtype = DataType::Bool;
            if (value->dtype == DataType::Int64) {
                for (const int64_t v : value->i64) {
                    out.boolean.push_back(static_cast<char>(v != 0));
                }
            } else {
                for (const double v : value->f64) {
                    out.boolean.push_back(static_cast<char>(v != 0.0));
                }
            }
            return out;
        }
        if (to == 1) {
            // Only int64 (or float) sources reach the float lane.
            if (value->dtype == DataType::Float32) {
                return value;
            }
            if (value->dtype != DataType::Int64) {
                return std::nullopt;
            }
            out.dtype = DataType::Float32;
            for (const int64_t v : value->i64) {
                out.f64.push_back(static_cast<double>(static_cast<float>(v)));
            }
            return out;
        }
        if (value->dtype == DataType::Int64) {
            return value;
        }
        out.dtype = DataType::Int64;
        if (value->dtype == DataType::Float32) {
            for (const double v : value->f64) {
                if (!std::isfinite(v) || std::fabs(v) >= 9.0e18) {
                    return std::nullopt;
                }
                out.i64.push_back(static_cast<int64_t>(v)); // toward zero
            }
            return out;
        }
        out.i64.assign(value->boolean.begin(), value->boolean.end());
        return out;
    }
    if (op == "Slice") {
        // Rank-1 int64 data with constant starts/ends (and optional axes=0,
        // steps): the slice of a shape vector.
        if (node.inputs.size() < 3) {
            return std::nullopt;
        }
        const auto data = eval_tensor(ctx, node.inputs[0], depth);
        const auto starts = eval_tensor(ctx, node.inputs[1], depth);
        const auto ends = eval_tensor(ctx, node.inputs[2], depth);
        if (!data.has_value() || !starts.has_value() || !ends.has_value() ||
            data->dtype != DataType::Int64 || data->dims.size() != 1 ||
            starts->i64.size() != 1 || ends->i64.size() != 1) {
            return std::nullopt;
        }
        if (node.inputs.size() > 3 && !node.inputs[3].empty()) {
            const auto axes = eval_tensor(ctx, node.inputs[3], depth);
            if (!axes.has_value() || axes->i64.size() != 1 ||
                (axes->i64[0] != 0 && axes->i64[0] != -1)) {
                return std::nullopt;
            }
        }
        if (node.inputs.size() > 4 && !node.inputs[4].empty()) {
            const auto steps = eval_tensor(ctx, node.inputs[4], depth);
            if (!steps.has_value() || steps->i64.size() != 1 ||
                steps->i64[0] != 1) {
                return std::nullopt;
            }
        }
        const int64_t len = data->dims[0];
        const auto clamp = [len](int64_t v) {
            v = v < 0 ? v + len : v;
            return std::min(std::max(v, int64_t{0}), len);
        };
        const int64_t begin = clamp(starts->i64[0]);
        const int64_t end = std::max(begin, clamp(ends->i64[0]));
        EvalTensor out;
        out.dtype = DataType::Int64;
        out.i64.assign(data->i64.begin() + begin, data->i64.begin() + end);
        out.dims = {end - begin};
        return out;
    }
    if (op == "Gather") {
        if (node.inputs.size() != 2) {
            return std::nullopt;
        }
        const auto data = eval_tensor(ctx, node.inputs[0], depth);
        const auto indices = eval_tensor(ctx, node.inputs[1], depth);
        if (!data.has_value() || !indices.has_value() ||
            data->dtype != DataType::Int64 ||
            indices->dtype != DataType::Int64) {
            return std::nullopt;
        }
        const int64_t rank = static_cast<int64_t>(data->dims.size());
        if (rank == 0) {
            return std::nullopt;
        }
        int64_t axis = 0;
        if (const Attribute* attr = find_attr(node, "axis")) {
            const int64_t* value = std::get_if<int64_t>(&attr->value);
            if (value == nullptr) {
                return std::nullopt;
            }
            axis = *value;
        }
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            return std::nullopt;
        }
        const int64_t dim = data->dims[static_cast<size_t>(axis)];
        std::vector<int64_t> picked;
        picked.reserve(indices->i64.size());
        for (int64_t k : indices->i64) {
            if (k < 0) {
                k += dim; // negative wrap, like the kernel
            }
            if (k < 0 || k >= dim) {
                return std::nullopt;
            }
            picked.push_back(k);
        }
        std::vector<int64_t> out_dims;
        for (int64_t i = 0; i < axis; ++i) {
            out_dims.push_back(data->dims[static_cast<size_t>(i)]);
        }
        out_dims.insert(out_dims.end(), indices->dims.begin(),
            indices->dims.end());
        for (size_t i = static_cast<size_t>(axis) + 1; i < data->dims.size();
             ++i) {
            out_dims.push_back(data->dims[i]);
        }
        int64_t outer = 1;
        for (int64_t i = 0; i < axis; ++i) {
            outer *= data->dims[static_cast<size_t>(i)];
        }
        int64_t inner = 1;
        for (int64_t i = axis + 1; i < rank; ++i) {
            inner *= data->dims[static_cast<size_t>(i)];
        }
        const int64_t index_count = eval_element_count(indices->dims);
        EvalTensor out;
        out.dtype = DataType::Int64;
        out.dims = out_dims;
        out.i64.assign(static_cast<size_t>(eval_element_count(out_dims)), 0);
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t j = 0; j < index_count; ++j) {
                const int64_t k = picked[static_cast<size_t>(j)];
                for (int64_t n = 0; n < inner; ++n) {
                    out.i64[static_cast<size_t>(
                        (o * index_count + j) * inner + n)] =
                        data->i64[static_cast<size_t>(
                            (o * dim + k) * inner + n)];
                }
            }
        }
        return out;
    }
    if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div") {
        return eval_int64_binary(ctx, node, depth, op);
    }
    if (op == "Equal") {
        if (node.inputs.size() != 2) {
            return std::nullopt;
        }
        const auto lhs = eval_tensor(ctx, node.inputs[0], depth);
        const auto rhs = eval_tensor(ctx, node.inputs[1], depth);
        if (!lhs.has_value() || !rhs.has_value() ||
            lhs->dtype != DataType::Int64 ||
            rhs->dtype != DataType::Int64) {
            return std::nullopt;
        }
        const auto dims = eval_broadcast(lhs->dims, rhs->dims);
        if (!dims.has_value()) {
            return std::nullopt;
        }
        EvalTensor out;
        out.dtype = DataType::Bool;
        out.dims = *dims;
        out.boolean.assign(static_cast<size_t>(eval_element_count(out.dims)),
            0);
        for (int64_t i = 0; i < eval_element_count(out.dims); ++i) {
            const int64_t a = lhs->i64[static_cast<size_t>(
                eval_offset(lhs->dims, out.dims, i))];
            const int64_t b = rhs->i64[static_cast<size_t>(
                eval_offset(rhs->dims, out.dims, i))];
            out.boolean[static_cast<size_t>(i)] = static_cast<char>(a == b);
        }
        return out;
    }
    if (op == "Greater" || op == "Less") {
        if (node.inputs.size() != 2) {
            return std::nullopt;
        }
        const auto lhs = eval_tensor(ctx, node.inputs[0], depth);
        const auto rhs = eval_tensor(ctx, node.inputs[1], depth);
        if (!lhs.has_value() || !rhs.has_value() ||
            lhs->dtype != DataType::Int64 ||
            rhs->dtype != DataType::Int64) {
            return std::nullopt;
        }
        const auto dims = eval_broadcast(lhs->dims, rhs->dims);
        if (!dims.has_value()) {
            return std::nullopt;
        }
        EvalTensor out;
        out.dtype = DataType::Bool;
        out.dims = *dims;
        out.boolean.assign(static_cast<size_t>(eval_element_count(out.dims)),
            0);
        for (int64_t i = 0; i < eval_element_count(out.dims); ++i) {
            const int64_t a = lhs->i64[static_cast<size_t>(
                eval_offset(lhs->dims, out.dims, i))];
            const int64_t b = rhs->i64[static_cast<size_t>(
                eval_offset(rhs->dims, out.dims, i))];
            out.boolean[static_cast<size_t>(i)] =
                static_cast<char>(op == "Greater" ? a > b : a < b);
        }
        return out;
    }
    if (op == "And") {
        if (node.inputs.size() != 2) {
            return std::nullopt;
        }
        const auto lhs = eval_tensor(ctx, node.inputs[0], depth);
        const auto rhs = eval_tensor(ctx, node.inputs[1], depth);
        if (!lhs.has_value() || !rhs.has_value() ||
            lhs->dtype != DataType::Bool || rhs->dtype != DataType::Bool) {
            return std::nullopt;
        }
        const auto dims = eval_broadcast(lhs->dims, rhs->dims);
        if (!dims.has_value()) {
            return std::nullopt;
        }
        EvalTensor out;
        out.dtype = DataType::Bool;
        out.dims = *dims;
        out.boolean.assign(static_cast<size_t>(eval_element_count(out.dims)),
            0);
        for (int64_t i = 0; i < eval_element_count(out.dims); ++i) {
            const char a = lhs->boolean[static_cast<size_t>(
                eval_offset(lhs->dims, out.dims, i))];
            const char b = rhs->boolean[static_cast<size_t>(
                eval_offset(rhs->dims, out.dims, i))];
            out.boolean[static_cast<size_t>(i)] = static_cast<char>(a && b);
        }
        return out;
    }
    if (op == "Not") {
        if (node.inputs.size() != 1) {
            return std::nullopt;
        }
        const auto in = eval_tensor(ctx, node.inputs[0], depth);
        if (!in.has_value() || in->dtype != DataType::Bool) {
            return std::nullopt;
        }
        EvalTensor out;
        out.dtype = DataType::Bool;
        out.dims = in->dims;
        out.boolean.reserve(in->boolean.size());
        for (const char v : in->boolean) {
            out.boolean.push_back(static_cast<char>(!v));
        }
        return out;
    }
    if (op == "Where") {
        if (node.inputs.size() != 3) {
            return std::nullopt;
        }
        const auto cond = eval_tensor(ctx, node.inputs[0], depth);
        const auto x = eval_tensor(ctx, node.inputs[1], depth);
        const auto y = eval_tensor(ctx, node.inputs[2], depth);
        if (!cond.has_value() || !x.has_value() || !y.has_value() ||
            cond->dtype != DataType::Bool || x->dtype != DataType::Int64 ||
            y->dtype != DataType::Int64) {
            return std::nullopt;
        }
        const auto first = eval_broadcast(cond->dims, x->dims);
        if (!first.has_value()) {
            return std::nullopt;
        }
        const auto dims = eval_broadcast(*first, y->dims);
        if (!dims.has_value()) {
            return std::nullopt;
        }
        EvalTensor out;
        out.dtype = DataType::Int64;
        out.dims = *dims;
        out.i64.assign(static_cast<size_t>(eval_element_count(out.dims)), 0);
        for (int64_t i = 0; i < eval_element_count(out.dims); ++i) {
            const size_t at = static_cast<size_t>(i);
            const bool take = cond->boolean[static_cast<size_t>(
                                eval_offset(cond->dims, out.dims, i))] != 0;
            out.i64[at] = take
                ? x->i64[static_cast<size_t>(
                      eval_offset(x->dims, out.dims, i))]
                : y->i64[static_cast<size_t>(
                      eval_offset(y->dims, out.dims, i))];
        }
        return out;
    }
    if (op == "Range") {
        if (node.inputs.size() != 3) {
            return std::nullopt;
        }
        const auto start = eval_tensor(ctx, node.inputs[0], depth);
        const auto limit = eval_tensor(ctx, node.inputs[1], depth);
        const auto delta = eval_tensor(ctx, node.inputs[2], depth);
        if (!start.has_value() || !limit.has_value() || !delta.has_value() ||
            start->dtype != limit->dtype || start->dtype != delta->dtype ||
            eval_element_count(start->dims) != 1 ||
            eval_element_count(limit->dims) != 1 ||
            eval_element_count(delta->dims) != 1) {
            return std::nullopt;
        }
        constexpr int64_t kMaxEvalRange = int64_t{1} << 24;
        EvalTensor out;
        out.dtype = start->dtype;
        if (start->dtype == DataType::Int64) {
            const int64_t a = start->i64[0];
            const int64_t b = limit->i64[0];
            const int64_t d = delta->i64[0];
            if (d == 0) {
                return std::nullopt;
            }
            int64_t count = 0;
            if (d > 0 && b > a) {
                count = (b - a + d - 1) / d;
            } else if (d < 0 && b < a) {
                count = (a - b + (-d) - 1) / (-d);
            }
            if (count > kMaxEvalRange) {
                return std::nullopt;
            }
            for (int64_t i = 0; i < count; ++i) {
                out.i64.push_back(a + i * d);
            }
            out.dims = {count};
            return out;
        }
        if (start->dtype != DataType::Float32) {
            return std::nullopt;
        }
        const double a = start->f64[0];
        const double b = limit->f64[0];
        const double d = delta->f64[0];
        if (d == 0.0 || !std::isfinite(a) || !std::isfinite(b) ||
            !std::isfinite(d)) {
            return std::nullopt;
        }
        const double span = std::ceil((b - a) / d);
        if (span > static_cast<double>(kMaxEvalRange)) {
            return std::nullopt;
        }
        const int64_t count = span > 0.0 ? static_cast<int64_t>(span) : 0;
        for (int64_t i = 0; i < count; ++i) {
            out.f64.push_back(static_cast<double>(
                static_cast<float>(a) +
                static_cast<float>(i) * static_cast<float>(d)));
        }
        out.dims = {count};
        return out;
    }
    if (op == "ReduceSum") {
        if (node.inputs.empty() || node.inputs.size() > 2 ||
            node.outputs.size() != 1) {
            return std::nullopt;
        }
        const auto data = eval_tensor(ctx, node.inputs[0], depth);
        const auto out_it = ctx.inferred.find(node.outputs[0]);
        if (!data.has_value() || data->dtype != DataType::Int64 ||
            out_it == ctx.inferred.end()) {
            return std::nullopt;
        }
        const int64_t rank = static_cast<int64_t>(data->dims.size());
        std::vector<int64_t> axes;
        if (node.inputs.size() == 2 && !node.inputs[1].empty()) {
            const auto axes_value = eval_tensor(ctx, node.inputs[1], depth);
            if (!axes_value.has_value() ||
                axes_value->dtype != DataType::Int64) {
                return std::nullopt;
            }
            axes = axes_value->i64;
        }
        std::vector<char> reduced(static_cast<size_t>(rank), 0);
        if (axes.empty()) {
            if (attr_int(node, "noop_with_empty_axes", 0) != 0) {
                EvalTensor same = *data;
                same.dims = out_it->second.dims;
                return same;
            }
            reduced.assign(static_cast<size_t>(rank), 1);
        }
        for (int64_t axis : axes) {
            if (axis < 0) {
                axis += rank;
            }
            if (axis < 0 || axis >= rank) {
                return std::nullopt;
            }
            reduced[static_cast<size_t>(axis)] = 1;
        }
        const std::vector<int64_t> in_strides = eval_strides(data->dims);
        // Output strides over the kept axes only, in input-axis order.
        std::vector<int64_t> kept_dims;
        for (int64_t j = 0; j < rank; ++j) {
            if (reduced[static_cast<size_t>(j)] == 0) {
                kept_dims.push_back(data->dims[static_cast<size_t>(j)]);
            }
        }
        const std::vector<int64_t> kept_strides = eval_strides(kept_dims);
        EvalTensor out;
        out.dtype = DataType::Int64;
        out.dims = out_it->second.dims;
        out.i64.assign(static_cast<size_t>(eval_element_count(kept_dims)), 0);
        for (int64_t i = 0; i < eval_element_count(data->dims); ++i) {
            int64_t remaining = i;
            int64_t offset = 0;
            size_t kept = 0;
            for (int64_t j = 0; j < rank; ++j) {
                const size_t uj = static_cast<size_t>(j);
                const int64_t coord = remaining / in_strides[uj];
                remaining %= in_strides[uj];
                if (reduced[uj] == 0) {
                    offset += coord * kept_strides[kept];
                    ++kept;
                }
            }
            out.i64[static_cast<size_t>(offset)] +=
                data->i64[static_cast<size_t>(i)];
        }
        return out;
    }
    if (op == "ConstantOfShape") {
        if (node.inputs.size() != 1) {
            return std::nullopt;
        }
        const auto shape = eval_tensor(ctx, node.inputs[0], depth);
        if (!shape.has_value() || shape->dtype != DataType::Int64 ||
            shape->dims.size() != 1) {
            return std::nullopt;
        }
        for (const int64_t v : shape->i64) {
            if (v < 0) {
                return std::nullopt;
            }
        }
        const Attribute* attr = find_attr(node, "value");
        EvalTensor out;
        out.dims = shape->i64;
        const size_t count = static_cast<size_t>(eval_element_count(out.dims));
        if (attr == nullptr) {
            out.dtype = DataType::Float32;
            out.f64.assign(count, 0.0);
        } else if (const int64_t* i = std::get_if<int64_t>(&attr->value)) {
            out.dtype = DataType::Int64;
            out.i64.assign(count, *i);
        } else if (const float* f = std::get_if<float>(&attr->value)) {
            out.dtype = DataType::Float32;
            out.f64.assign(count, static_cast<double>(*f));
        } else if (const bool* b = std::get_if<bool>(&attr->value)) {
            out.dtype = DataType::Bool;
            out.boolean.assign(count, static_cast<char>(*b));
        } else {
            return std::nullopt;
        }
        return out;
    }
    return std::nullopt;
}

std::vector<int64_t> int64_constant(const StaticContext& ctx, const Node& node,
    const std::string& name)
{
    const auto it = ctx.graph.initializers.find(name);
    if (it != ctx.graph.initializers.end()) {
        if (it->second.dtype != DataType::Int64) {
            fail(node, "input " + quote(name) + " must be an Int64 constant");
        }
        const std::vector<int64_t>* values =
            std::get_if<std::vector<int64_t>>(&it->second.values);
        if (values == nullptr) {
            fail(node, "input " + quote(name) + " is not an Int64 constant");
        }
        return *values;
    }
    const auto evaluated = eval_tensor(ctx, name, 0);
    if (evaluated.has_value() && evaluated->dtype == DataType::Int64) {
        return evaluated->i64;
    }
    fail(node, "input " + quote(name) + " must be an Int64 constant");
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

TensorInfo reducemean_shape(const StaticContext& ctx, const Node& node,
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
        axes = int64_constant(ctx, node, node.inputs[1]);
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

TensorInfo reducesum_shape(const StaticContext& ctx, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "ReduceSum requires an input");
    }
    if (inputs[0]->dtype != DataType::Float32 &&
        inputs[0]->dtype != DataType::Int64) {
        fail(node, "ReduceSum data must be Float32 or Int64");
    }
    const DataType dtype = inputs[0]->dtype;
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    const bool keepdims = attr_int(node, "keepdims", 1) != 0;
    const bool noop_empty = attr_int(node, "noop_with_empty_axes", 0) != 0;

    std::vector<int64_t> axes;
    if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
        axes = int64_constant(ctx, node, node.inputs[1]);
    }
    std::set<int64_t> seen;
    for (int64_t& axis : axes) {
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            fail(node, "ReduceSum axis is out of range");
        }
        if (!seen.insert(axis).second) {
            fail(node, "ReduceSum axes must be unique");
        }
    }

    std::vector<int64_t> out = in;
    if (axes.empty()) {
        if (noop_empty) {
            return TensorInfo{dtype, out};
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
    return TensorInfo{dtype, out};
}

// Read a scalar Float32 Range operand from an initializer or the evaluator's
// float lane.
double range_float_scalar(const StaticContext& ctx, const Node& node,
    const std::string& name)
{
    const auto value = eval_tensor(ctx, name, 0);
    if (!value.has_value() || value->dtype != DataType::Float32 ||
        value->f64.size() != 1) {
        fail(node, "input " + quote(name) + " must be a Float32 constant");
    }
    return value->f64[0];
}

TensorInfo range_shape(const StaticContext& ctx, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (node.inputs.size() != 3 || inputs.size() != 3 ||
        inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) {
        fail(node, "Range requires start, limit and delta");
    }
    const DataType dtype = inputs[0]->dtype;
    if (inputs[1]->dtype != dtype || inputs[2]->dtype != dtype ||
        (dtype != DataType::Float32 && dtype != DataType::Int64)) {
        fail(node, "Range operands must share dtype Float32 or Int64");
    }
    for (const TensorInfo* info : inputs) {
        if (element_count(info->dims) != 1) {
            fail(node, "Range operands must be scalars");
        }
    }
    int64_t count = 0;
    if (dtype == DataType::Int64) {
        const std::vector<int64_t> start =
            int64_constant(ctx, node, node.inputs[0]);
        const std::vector<int64_t> limit =
            int64_constant(ctx, node, node.inputs[1]);
        const std::vector<int64_t> delta =
            int64_constant(ctx, node, node.inputs[2]);
        if (start.size() != 1 || limit.size() != 1 || delta.size() != 1) {
            fail(node, "Range operands must be scalars");
        }
        const int64_t d = delta[0];
        if (d == 0) {
            fail(node, "Range delta must be non-zero");
        }
        if (d > 0 && limit[0] > start[0]) {
            count = (limit[0] - start[0] + d - 1) / d;
        } else if (d < 0 && limit[0] < start[0]) {
            count = (start[0] - limit[0] + (-d) - 1) / (-d);
        }
    } else {
        const double start = range_float_scalar(ctx, node, node.inputs[0]);
        const double limit = range_float_scalar(ctx, node, node.inputs[1]);
        const double delta = range_float_scalar(ctx, node, node.inputs[2]);
        if (delta == 0.0) {
            fail(node, "Range delta must be non-zero");
        }
        if (!std::isfinite(start) || !std::isfinite(limit) ||
            !std::isfinite(delta)) {
            fail(node, "Range operands must be finite");
        }
        const double span = std::ceil((limit - start) / delta);
        if (span > 1.0e12) {
            fail(node, "Range is too long");
        }
        count = span > 0.0 ? static_cast<int64_t>(span) : 0;
    }
    return TensorInfo{dtype, {count}};
}

TensorInfo pad_shape(const StaticContext& ctx, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.size() < 2 || inputs[0] == nullptr || node.inputs.size() < 2 ||
        node.inputs[1].empty()) {
        fail(node, "Pad requires data and pads");
    }
    if (inputs[0]->dtype != DataType::Float32 &&
        inputs[0]->dtype != DataType::Int64) {
        fail(node, "Pad data must be Float32 or Int64");
    }
    const std::string mode = attr_str(node, "mode", "constant");
    if (mode != "constant" && mode != "reflect" && mode != "edge") {
        fail(node, "Pad mode " + quote(mode) + " is not supported");
    }
    if (node.inputs.size() > 2 && !node.inputs[2].empty() &&
        inputs.size() > 2 && inputs[2] != nullptr &&
        (inputs[2]->dtype != inputs[0]->dtype ||
            element_count(inputs[2]->dims) != 1)) {
        fail(node, "Pad constant_value must be a scalar of the data dtype");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    const std::vector<int64_t> pads =
        int64_constant(ctx, node, node.inputs[1]);

    std::vector<int64_t> axes;
    if (node.inputs.size() > 3 && !node.inputs[3].empty()) {
        axes = int64_constant(ctx, node, node.inputs[3]);
    } else {
        for (int64_t i = 0; i < rank; ++i) {
            axes.push_back(i);
        }
    }
    if (pads.size() != axes.size() * 2) {
        fail(node, "Pad pads must hold two entries per padded axis");
    }
    std::vector<int64_t> out = in;
    std::set<int64_t> seen;
    for (size_t k = 0; k < axes.size(); ++k) {
        const int64_t axis = normalize_axis(node, axes[k], rank, "Pad");
        if (!seen.insert(axis).second) {
            fail(node, "Pad axes must be unique");
        }
        out[static_cast<size_t>(axis)] += pads[k] + pads[axes.size() + k];
    }
    for (const int64_t dim : out) {
        if (dim <= 0) {
            fail(node, "Pad produces a non-positive output dim");
        }
    }
    return TensorInfo{inputs[0]->dtype, out};
}

TensorInfo reshape_shape(const StaticContext& ctx, const Node& node,
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
        int64_constant(ctx, node, node.inputs[1]);
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
    // Reshape is layout-only: the output keeps the data's dtype (transformer
    // exports reshape int64 shape tensors as well as float32 activations).
    return TensorInfo{inputs[0]->dtype, out};
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
    return TensorInfo{inputs[0]->dtype, {first, second}};
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
    const DataType dtype = inputs[0]->dtype;
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i] == nullptr) {
            fail(node, "Concat requires all inputs");
        }
        if (inputs[i]->dtype != dtype) {
            fail(node, "Concat inputs must share dtype");
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
    return TensorInfo{dtype, out};
}

std::vector<TensorInfo> split_shapes(const StaticContext& ctx, const Node& node,
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
        parts = int64_constant(ctx, node, node.inputs[1]);
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
        outputs.push_back(TensorInfo{inputs[0]->dtype, dims});
    }
    return outputs;
}

TensorInfo unsqueeze_shape(const StaticContext& ctx, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Unsqueeze requires an input");
    }
    if (node.inputs.size() < 2 || node.inputs[1].empty()) {
        fail(node, "Unsqueeze requires an axes input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    std::vector<int64_t> axes = int64_constant(ctx, node, node.inputs[1]);
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
    return TensorInfo{inputs[0]->dtype, out};
}

TensorInfo expand_shape(const StaticContext& ctx, const Node& node,
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
        int64_constant(ctx, node, node.inputs[1]);
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
        if (d == s) {
            out[i] = s;
        } else if (d == 1 || di < 0) {
            out[i] = s;
        } else if (s == 1) {
            out[i] = d;
        } else {
            fail(node, "Expand input is not broadcastable to the target shape");
        }
    }
    return TensorInfo{inputs[0]->dtype, out};
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
        return TensorInfo{inputs[0]->dtype, out};
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
    return TensorInfo{inputs[0]->dtype, out};
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
    return TensorInfo{inputs[0]->dtype, inputs[1]->dims};
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
    return TensorInfo{inputs[0]->dtype, out};
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
    if (to == 9) {
        return TensorInfo{DataType::Bool, inputs[0]->dims};
    }
    fail(node, "Cast 'to' names an unsupported element type");
}

TensorInfo resize_shape(const StaticContext& ctx, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Resize requires an input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    // An empty scales tensor means "absent", as older exporters write it.
    const bool has_scales = node.inputs.size() > 2 && !node.inputs[2].empty() &&
        !(inputs.size() > 2 && inputs[2] != nullptr &&
            element_count(inputs[2]->dims) == 0);
    const bool has_sizes = node.inputs.size() > 3 && !node.inputs[3].empty();
    if (has_sizes && has_scales) {
        fail(node, "Resize takes exactly one of scales and sizes");
    }
    if (has_sizes) {
        const std::vector<int64_t> sizes =
            int64_constant(ctx, node, node.inputs[3]);
        if (static_cast<int64_t>(sizes.size()) != rank) {
            fail(node, "Resize sizes rank must match the input rank");
        }
        for (const int64_t dim : sizes) {
            if (dim <= 0) {
                fail(node, "Resize produces a non-positive output dim");
            }
        }
        return TensorInfo{inputs[0]->dtype, sizes};
    }
    if (has_scales) {
        const std::vector<float> scales =
            float32_constant(ctx.graph, node, node.inputs[2]);
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
        return TensorInfo{inputs[0]->dtype, out};
    }
    fail(node, "Resize requires a scales input");
}

TensorInfo slice_shape(const StaticContext& ctx, const Node& node,
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
        int64_constant(ctx, node, node.inputs[1]);
    const std::vector<int64_t> ends =
        int64_constant(ctx, node, node.inputs[2]);
    if (starts.size() != ends.size()) {
        fail(node, "Slice starts and ends must have the same length");
    }
    const size_t count = starts.size();
    std::vector<int64_t> axes(count, 0);
    for (size_t i = 0; i < count; ++i) {
        axes[i] = static_cast<int64_t>(i);
    }
    if (node.inputs.size() > 3 && !node.inputs[3].empty()) {
        axes = int64_constant(ctx, node, node.inputs[3]);
        if (axes.size() != count) {
            fail(node, "Slice axes must match starts in length");
        }
    }
    std::vector<int64_t> steps(count, 1);
    if (node.inputs.size() > 4 && !node.inputs[4].empty()) {
        steps = int64_constant(ctx, node, node.inputs[4]);
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
    return TensorInfo{inputs[0]->dtype, out};
}

std::vector<TensorInfo> topk_shapes(const StaticContext& ctx, const Node& node,
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
        int64_constant(ctx, node, node.inputs[1]);
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
    // Values stay float32: the TopK kernel takes float32 input only, so the
    // values output can never be anything else (Y8 audit stay-as-is).
    return {TensorInfo{DataType::Float32, out},
        TensorInfo{DataType::Int64, out}};
}

TensorInfo constantofshape_shape(const StaticContext& ctx, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr || node.inputs[0].empty()) {
        fail(node, "ConstantOfShape requires a shape input");
    }
    if (inputs[0]->dims.size() != 1) {
        fail(node, "ConstantOfShape shape input must be rank 1");
    }
    const std::vector<int64_t> shape =
        int64_constant(ctx, node, node.inputs[0]);
    for (const int64_t v : shape) {
        if (v < 0) {
            fail(node, "ConstantOfShape shape values must be non-negative");
        }
    }
    return TensorInfo{constantofshape_dtype(node), shape};
}

TensorInfo shape_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Shape requires an input");
    }
    const std::pair<int64_t, int64_t> range =
        shape_range(node, static_cast<int64_t>(inputs[0]->dims.size()));
    if (range.second == range.first) {
        fail(node, "Shape selects no dims");
    }
    return TensorInfo{DataType::Int64, {range.second - range.first}};
}

TensorInfo reducemax_shape(const StaticContext& ctx, const Node& node,
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
        axes = int64_constant(ctx, node, node.inputs[1]);
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

TensorInfo averagepool_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "AveragePool requires an input");
    }
    const std::vector<int64_t>& x = inputs[0]->dims;
    if (x.size() != 4) {
        fail(node, "AveragePool input must be rank 4 (NCHW)");
    }
    const std::vector<int64_t> kernel = attr_ints(node, "kernel_shape");
    if (kernel.size() != 2) {
        fail(node, "AveragePool kernel_shape must have two spatial entries");
    }
    const std::vector<int64_t> in_spatial = {x[2], x[3]};
    std::vector<int64_t> strides = attr_ints(node, "strides");
    if (strides.empty()) {
        strides = {1, 1};
    }
    if (strides.size() != 2) {
        fail(node, "AveragePool strides must have two spatial entries");
    }
    const std::vector<int64_t> dilations = {1, 1};
    const bool ceil_mode = attr_int(node, "ceil_mode", 0) != 0;
    std::vector<int64_t> pad_begin;
    std::vector<int64_t> pad_end;
    resolve_pads(node, attr_str(node, "auto_pad", "NOTSET"), in_spatial,
        kernel, dilations, strides, attr_ints(node, "pads"), pad_begin, pad_end);

    std::vector<int64_t> out = {x[0], x[1]};
    for (size_t i = 0; i < 2; ++i) {
        out.push_back(window_output(in_spatial[i], kernel[i], 1, pad_begin[i],
            pad_end[i], strides[i], ceil_mode, node, "AveragePool"));
    }
    return TensorInfo{DataType::Float32, out};
}

// ConvTranspose: stride*(in-1) + output_padding + dilation*(k-1) + 1 - pads.
TensorInfo convtranspose_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.size() < 2 || inputs[0] == nullptr || inputs[1] == nullptr) {
        fail(node, "ConvTranspose requires an input and a weight");
    }
    const std::vector<int64_t>& x = inputs[0]->dims;
    const std::vector<int64_t>& w = inputs[1]->dims;
    if (x.size() != 4 || w.size() != 4) {
        fail(node, "ConvTranspose input and weight must be rank 4");
    }
    const int64_t group = attr_int(node, "group", 1);
    if (group <= 0 || x[1] % group != 0) {
        fail(node, "ConvTranspose input channels are not divisible by group");
    }
    if (w[0] != x[1]) {
        fail(node, "ConvTranspose weight channel dim does not match input");
    }
    std::vector<int64_t> strides = attr_ints(node, "strides");
    std::vector<int64_t> dilations = attr_ints(node, "dilations");
    std::vector<int64_t> output_padding = attr_ints(node, "output_padding");
    std::vector<int64_t> pads = attr_ints(node, "pads");
    if (strides.empty()) {
        strides = {1, 1};
    }
    if (dilations.empty()) {
        dilations = {1, 1};
    }
    if (output_padding.empty()) {
        output_padding = {0, 0};
    }
    if (pads.empty() || attr_str(node, "auto_pad", "NOTSET") == "VALID") {
        pads = {0, 0, 0, 0};
    }
    if (strides.size() != 2 || dilations.size() != 2 ||
        output_padding.size() != 2 || pads.size() != 4) {
        fail(node, "ConvTranspose strides, dilations, output_padding, and "
                   "pads must be 2-D");
    }
    std::vector<int64_t> out = {x[0], w[1] * group};
    for (size_t i = 0; i < 2; ++i) {
        if (strides[i] <= 0 || dilations[i] <= 0) {
            fail(node, "ConvTranspose stride and dilation must be positive");
        }
        const int64_t dim = strides[i] * (x[2 + i] - 1) + output_padding[i] +
            dilations[i] * (w[2 + i] - 1) + 1 - pads[i] - pads[2 + i];
        if (dim <= 0) {
            fail(node, "ConvTranspose produces a non-positive output dim");
        }
        out.push_back(dim);
    }
    if (inputs.size() > 2 && inputs[2] != nullptr) {
        const std::vector<int64_t>& bias = inputs[2]->dims;
        if (bias.size() != 1 || bias[0] != out[1]) {
            fail(node, "ConvTranspose bias must be rank-1 of length M");
        }
    }
    return TensorInfo{DataType::Float32, out};
}

TensorInfo tile_shape(const StaticContext& ctx, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.size() < 2 || inputs[0] == nullptr || node.inputs[1].empty()) {
        fail(node, "Tile requires an input and a repeats input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const std::vector<int64_t> repeats =
        int64_constant(ctx, node, node.inputs[1]);
    if (repeats.size() != in.size()) {
        fail(node, "Tile repeats must have one entry per input axis");
    }
    std::vector<int64_t> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        if (repeats[i] < 1) {
            fail(node, "Tile repeats must be positive");
        }
        out[i] = in[i] * repeats[i];
    }
    return TensorInfo{inputs[0]->dtype, out};
}

TensorInfo squeeze_shape(const StaticContext& ctx, const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.empty() || inputs[0] == nullptr) {
        fail(node, "Squeeze requires an input");
    }
    const std::vector<int64_t>& in = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(in.size());
    std::set<int64_t> drop;
    if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
        for (const int64_t axis : int64_constant(ctx, node, node.inputs[1])) {
            const int64_t a = normalize_axis(node, axis, rank, "Squeeze");
            if (in[static_cast<size_t>(a)] != 1) {
                fail(node, "Squeeze axis does not have size 1");
            }
            drop.insert(a);
        }
    } else {
        for (int64_t a = 0; a < rank; ++a) {
            if (in[static_cast<size_t>(a)] == 1) {
                drop.insert(a);
            }
        }
    }
    std::vector<int64_t> out;
    for (int64_t a = 0; a < rank; ++a) {
        if (drop.count(a) == 0) {
            out.push_back(in[static_cast<size_t>(a)]);
        }
    }
    return TensorInfo{inputs[0]->dtype, out};
}

// LayerNormalization: Y has X's shape; Mean and InvStdDev keep the leading
// dims and set the normalized ones to 1.
std::vector<TensorInfo> layernorm_shapes(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.size() < 2 || inputs[0] == nullptr || inputs[1] == nullptr) {
        fail(node, "LayerNormalization requires X and Scale");
    }
    const std::vector<int64_t>& x = inputs[0]->dims;
    const int64_t rank = static_cast<int64_t>(x.size());
    if (rank == 0) {
        fail(node, "LayerNormalization input must not be a scalar");
    }
    const int64_t axis = normalize_axis(node, attr_int(node, "axis", -1), rank,
        "LayerNormalization");
    std::vector<int64_t> stats = x;
    for (int64_t a = axis; a < rank; ++a) {
        stats[static_cast<size_t>(a)] = 1;
    }
    std::vector<TensorInfo> out = {TensorInfo{DataType::Float32, x}};
    for (size_t i = 1; i < node.outputs.size(); ++i) {
        out.push_back(TensorInfo{DataType::Float32, stats});
    }
    return out;
}

TensorInfo gridsample_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    if (inputs.size() < 2 || inputs[0] == nullptr || inputs[1] == nullptr) {
        fail(node, "GridSample requires X and grid inputs");
    }
    const std::string mode = attr_str(node, "mode", "bilinear");
    if (mode != "bilinear" && mode != "nearest") {
        fail(node, "GridSample mode must be 'bilinear' or 'nearest'");
    }
    const std::string padding = attr_str(node, "padding_mode", "zeros");
    if (padding != "zeros" && padding != "border" && padding != "reflection") {
        fail(node, "GridSample padding_mode must be 'zeros', 'border' or "
            "'reflection'");
    }
    const int64_t corners = attr_int(node, "align_corners", 0);
    if (corners != 0 && corners != 1) {
        fail(node, "GridSample align_corners must be 0 or 1");
    }
    const TensorInfo& x = *inputs[0];
    const TensorInfo& grid = *inputs[1];
    if (x.dtype != DataType::Float32 || grid.dtype != DataType::Float32) {
        fail(node, "GridSample inputs must be Float32");
    }
    if (x.dims.size() != 4) {
        fail(node, "GridSample X must be rank 4 (NCHW)");
    }
    if (grid.dims.size() != 4 || grid.dims[3] != 2) {
        fail(node, "GridSample grid must have shape [N,Ho,Wo,2]");
    }
    if (grid.dims[0] != x.dims[0]) {
        fail(node, "GridSample X and grid batch sizes differ");
    }
    return TensorInfo{DataType::Float32,
        {x.dims[0], x.dims[1], grid.dims[1], grid.dims[2]}};
}

// Einsum: output dims from the label sizes bound by the operands.
TensorInfo einsum_shape(const Node& node,
    const std::vector<const TensorInfo*>& inputs)
{
    const std::string raw = attr_str(node, "equation", "");
    std::string text;
    for (const char c : raw) {
        if (c != ' ') {
            text.push_back(c);
        }
    }
    if (text.empty()) {
        fail(node, "Einsum requires an 'equation' attribute");
    }
    if (text.find('.') != std::string::npos) {
        fail(node, "Einsum equation must not use an ellipsis");
    }
    const size_t arrow = text.find("->");
    const bool has_output = arrow != std::string::npos;
    const std::string lhs = has_output ? text.substr(0, arrow) : text;
    std::vector<std::string> subs;
    size_t start = 0;
    while (true) {
        const size_t comma = lhs.find(',', start);
        subs.push_back(lhs.substr(start,
            comma == std::string::npos ? std::string::npos : comma - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (subs.size() > 2) {
        fail(node, "Einsum supports at most two operands");
    }
    if (subs.size() != inputs.size()) {
        fail(node, "Einsum operand count does not match the equation");
    }
    std::map<char, int64_t> sizes;
    std::map<char, int> counts;
    for (size_t i = 0; i < subs.size(); ++i) {
        if (inputs[i] == nullptr) {
            fail(node, "missing required input");
        }
        if (inputs[i]->dtype != DataType::Float32) {
            fail(node, "Einsum inputs must be Float32");
        }
        if (inputs[i]->dims.size() != subs[i].size()) {
            fail(node, "Einsum operand rank does not match its subscript");
        }
        for (size_t j = 0; j < subs[i].size(); ++j) {
            const char c = subs[i][j];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
                fail(node, "Einsum equation has an invalid character");
            }
            ++counts[c];
            const auto inserted = sizes.emplace(c, inputs[i]->dims[j]);
            if (!inserted.second && inserted.first->second != inputs[i]->dims[j]) {
                fail(node, std::string("Einsum label '") + c +
                    "' is bound to two different sizes");
            }
        }
    }
    std::string out_labels;
    if (has_output) {
        out_labels = text.substr(arrow + 2);
        for (size_t i = 0; i < out_labels.size(); ++i) {
            if (sizes.find(out_labels[i]) == sizes.end()) {
                fail(node, "Einsum output label does not appear in the inputs");
            }
            if (out_labels.find(out_labels[i]) != i) {
                fail(node, "Einsum output label is repeated");
            }
        }
    } else {
        for (const auto& entry : counts) {
            if (entry.second == 1) {
                out_labels.push_back(entry.first);
            }
        }
    }
    std::vector<int64_t> out;
    for (const char c : out_labels) {
        out.push_back(sizes.at(c));
    }
    return TensorInfo{DataType::Float32, out};
}

std::vector<TensorInfo> compute_outputs(const StaticContext& ctx, const Node& node,
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
        const TensorInfo& lhs = need(0);
        const TensorInfo& rhs = need(1);
        if (lhs.dtype != rhs.dtype) {
            fail(node, op + " inputs must share dtype");
        }
        return {TensorInfo{lhs.dtype,
            broadcast_shape(lhs.dims, rhs.dims, node)}};
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
    if (op == "Relu" || op == "Sigmoid" || op == "Softmax" || op == "Floor" ||
        op == "Erf" || op == "Sqrt" || op == "Tanh" || op == "Exp" ||
        op == "Log" || op == "Cos" || op == "Sin" || op == "Clip") {
        return {TensorInfo{DataType::Float32, need(0).dims}};
    }
    if (op == "Abs" || op == "Neg") {
        return {need(0)};
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
        return {reducemean_shape(ctx, node, inputs)};
    }
    if (op == "ReduceSum") {
        return {reducesum_shape(ctx, node, inputs)};
    }
    if (op == "Range") {
        return {range_shape(ctx, node, inputs)};
    }
    if (op == "Pad") {
        return {pad_shape(ctx, node, inputs)};
    }
    if (op == "Reshape") {
        return {reshape_shape(ctx, node, inputs)};
    }
    if (op == "Flatten") {
        return {flatten_shape(node, inputs)};
    }
    if (op == "Concat") {
        return {concat_shape(node, inputs)};
    }
    if (op == "Split") {
        return split_shapes(ctx, node, inputs);
    }
    if (op == "Unsqueeze") {
        return {unsqueeze_shape(ctx, node, inputs)};
    }
    if (op == "Expand") {
        return {expand_shape(ctx, node, inputs)};
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
        return {resize_shape(ctx, node, inputs)};
    }
    if (op == "Slice") {
        return {slice_shape(ctx, node, inputs)};
    }
    if (op == "TopK") {
        return topk_shapes(ctx, node, inputs);
    }
    if (op == "ConstantOfShape") {
        return {constantofshape_shape(ctx, node, inputs)};
    }
    if (op == "Equal") {
        return {TensorInfo{DataType::Bool,
            broadcast_shape(need(0).dims, need(1).dims, node)}};
    }
    if (op == "Greater" || op == "Less") {
        if (need(0).dtype != need(1).dtype) {
            fail(node, op + " inputs must share dtype");
        }
        return {TensorInfo{DataType::Bool,
            broadcast_shape(need(0).dims, need(1).dims, node)}};
    }
    if (op == "And") {
        return {TensorInfo{DataType::Bool,
            broadcast_shape(need(0).dims, need(1).dims, node)}};
    }
    if (op == "Not") {
        return {TensorInfo{DataType::Bool, need(0).dims}};
    }
    if (op == "Where") {
        if (inputs.size() < 3 || inputs[0] == nullptr ||
            inputs[1] == nullptr || inputs[2] == nullptr) {
            fail(node, "Where requires condition, X, and Y inputs");
        }
        if (inputs[1]->dtype != inputs[2]->dtype) {
            fail(node, "Where X and Y inputs must share dtype");
        }
        return {TensorInfo{inputs[1]->dtype,
            broadcast_shape(broadcast_shape(inputs[0]->dims, inputs[1]->dims,
                                node),
                inputs[2]->dims, node)}};
    }
    if (op == "Shape") {
        return {shape_shape(node, inputs)};
    }
    if (op == "ReduceMax") {
        return {reducemax_shape(ctx, node, inputs)};
    }
    if (op == "Identity") {
        return {need(0)};
    }
    if (op == "Pow") {
        const TensorInfo& base = need(0);
        const TensorInfo& exponent = need(1);
        if (base.dtype != DataType::Float32 ||
            exponent.dtype != DataType::Float32) {
            fail(node, "Pow inputs must be Float32");
        }
        return {TensorInfo{DataType::Float32,
            broadcast_shape(base.dims, exponent.dims, node)}};
    }
    if (op == "Tile") {
        return {tile_shape(ctx, node, inputs)};
    }
    if (op == "AveragePool") {
        return {averagepool_shape(node, inputs)};
    }
    if (op == "ConvTranspose") {
        return {convtranspose_shape(node, inputs)};
    }
    if (op == "IsNaN") {
        return {TensorInfo{DataType::Bool, need(0).dims}};
    }
    if (op == "Squeeze") {
        return {squeeze_shape(ctx, node, inputs)};
    }
    if (op == "LayerNormalization") {
        return layernorm_shapes(node, inputs);
    }
    if (op == "GridSample") {
        return {gridsample_shape(node, inputs)};
    }
    if (op == "Einsum") {
        return {einsum_shape(node, inputs)};
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
        // Each graph input keeps its declared dtype (FLOAT, INT64, or BOOL);
        // a hand-built graph that left it Unknown is treated as float32.
        const DataType declared = entry.second.dtype == DataType::Unknown
            ? DataType::Float32
            : entry.second.dtype;
        result.tensors[name] = TensorInfo{declared, it->second};
    }

    for (const auto& init : graph.initializers) {
        result.tensors[init.first] =
            TensorInfo{init.second.dtype, init.second.dims};
    }

    std::map<std::string, size_t> producers;
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        for (const std::string& output : graph.nodes[i].outputs) {
            if (!output.empty()) {
                producers.emplace(output, i);
            }
        }
    }

    for (size_t current = 0; current < graph.nodes.size(); ++current) {
        const Node& node = graph.nodes[current];
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

        const StaticContext ctx{graph, result.tensors, producers, current};
        const std::vector<TensorInfo> outputs =
            compute_outputs(ctx, node, inputs);
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
