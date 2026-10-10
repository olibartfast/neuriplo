// Clip: elementwise clamp of float32 data to optional scalar min and max.

#include "Kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
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

// Reads an optional scalar bound; returns `fallback` when the input is omitted.
float ReadBound(const std::vector<TensorView>& inputs, size_t index, float fallback, const std::string& context) {
    if (index >= inputs.size() || inputs[index].dtype == DataType::Unknown) {
        return fallback;
    }
    const TensorView& bound = inputs[index];
    if (bound.dtype != DataType::Float32) {
        throw InferenceException(context + ": min/max must be float32");
    }
    if (ElementCount(bound.dims, context) != 1) {
        throw InferenceException(context + ": min/max must be scalar");
    }
    if (bound.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    return *static_cast<const float*>(bound.data);
}

} // namespace

void Clip(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Clip (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one to three inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    const int64_t count = ElementCount(out.dims, context);
    if (ElementCount(in.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const float lo = ReadBound(inputs, 1, -std::numeric_limits<float>::infinity(), context);
    const float hi = ReadBound(inputs, 2, std::numeric_limits<float>::infinity(), context);

    const float* x = static_cast<const float*>(in.data);
    float* y = static_cast<float*>(out.data);
    for (int64_t i = 0; i < count; ++i) {
        y[i] = std::min(std::max(x[i], lo), hi);
    }
}

} // namespace kernels
} // namespace engine
