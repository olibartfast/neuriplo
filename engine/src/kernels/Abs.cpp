// Abs: elementwise absolute value over float32 or int64 data.

#include "Kernels.hpp"

#include <cmath>
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

template <typename T> void Apply(const TensorView& in, const TensorView& out, int64_t count) {
    const T* x = static_cast<const T*>(in.data);
    T* y = static_cast<T*>(out.data);
    for (int64_t i = 0; i < count; ++i) {
        y[i] = std::abs(x[i]);
    }
}

} // namespace

void Abs(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Abs (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if ((in.dtype != DataType::Float32 && in.dtype != DataType::Int64) || out.dtype != in.dtype) {
        throw InferenceException(context + ": tensors must be float32 or int64 with matching dtype");
    }
    const int64_t count = ElementCount(out.dims, context);
    if (ElementCount(in.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    if (in.dtype == DataType::Float32) {
        Apply<float>(in, out, count);
    } else {
        Apply<int64_t>(in, out, count);
    }
}

} // namespace kernels
} // namespace engine
