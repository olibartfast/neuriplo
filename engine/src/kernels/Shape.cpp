// Shape: emit the input's dimensions as a rank-1 int64 tensor, optionally only
// dims[start:end] (opset 15; negative values count from the back, both clamp).
//
// The output has one element per input axis holding that axis extent. The
// input element type is unconstrained; the output is always int64.

#include "Kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace engine {
namespace kernels {
namespace {

int64_t ElementCount(const std::vector<int64_t>& dims, const std::string& context) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        count *= dim;
    }
    return count;
}

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Shape: attribute '" + name + "' must be an integer (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

} // namespace

void Shape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Shape (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": Shape input must be float32, int64, or bool");
    }
    if (out.dtype != DataType::Int64) {
        throw InferenceException(context + ": Shape output must be int64");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    int64_t start = IntAttribute(node, "start", 0);
    int64_t end = IntAttribute(node, "end", rank);
    start = std::min(std::max(start < 0 ? start + rank : start, int64_t{0}), rank);
    end = std::min(std::max(end < 0 ? end + rank : end, int64_t{0}), rank);
    if (end < start) {
        end = start;
    }
    if (out.dims != std::vector<int64_t>{end - start}) {
        throw InferenceException(context + ": Shape output shape must be the selected dim count");
    }
    if (end > start && out.data == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }

    int64_t* y = static_cast<int64_t*>(out.data);
    for (int64_t i = start; i < end; ++i) {
        y[i - start] = in.dims[static_cast<std::size_t>(i)];
    }
}

} // namespace kernels
} // namespace engine
