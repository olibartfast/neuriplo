// Range: 1-D sequence start, start + delta, ... up to (excluding) limit; float32 or int64.

#include "Kernels.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace engine {
namespace kernels {
namespace {

// Reads a scalar operand; the view must hold exactly one element of type T.
template <typename T>
T ReadScalar(const TensorView& view, DataType dtype, const char* what, const std::string& context) {
    if (view.dtype != dtype) {
        throw InferenceException(context + ": " + what + " dtype must match the other operands");
    }
    int64_t count = 1;
    for (int64_t dim : view.dims) {
        count *= dim;
    }
    if (count != 1) {
        throw InferenceException(context + ": " + what + " must be a scalar");
    }
    if (view.data == nullptr) {
        throw InferenceException(context + ": null " + what + " buffer");
    }
    return *static_cast<const T*>(view.data);
}

} // namespace

void Range(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Range (node '" + node.name + "')";
    if (inputs.size() != 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected start, limit, delta and one output");
    }
    const TensorView& out = outputs[0];
    const DataType dtype = inputs[0].dtype;
    if ((dtype != DataType::Float32 && dtype != DataType::Int64) || out.dtype != dtype) {
        throw InferenceException(context + ": tensors must share dtype float32 or int64");
    }
    if (out.dims.size() != 1 || out.dims[0] < 0) {
        throw InferenceException(context + ": output must be rank 1");
    }
    const int64_t outCount = out.dims[0];

    if (dtype == DataType::Float32) {
        const float start = ReadScalar<float>(inputs[0], dtype, "start", context);
        const float limit = ReadScalar<float>(inputs[1], dtype, "limit", context);
        const float delta = ReadScalar<float>(inputs[2], dtype, "delta", context);
        if (delta == 0.0f) {
            throw InferenceException(context + ": delta must be non-zero");
        }
        const double span =
            std::ceil((static_cast<double>(limit) - static_cast<double>(start)) / static_cast<double>(delta));
        const int64_t expected = span > 0.0 ? static_cast<int64_t>(span) : 0;
        if (outCount != expected) {
            throw InferenceException(context + ": output length does not match the range");
        }
        float* y = static_cast<float*>(out.data);
        if (outCount > 0 && y == nullptr) {
            throw InferenceException(context + ": null output buffer");
        }
        for (int64_t i = 0; i < outCount; ++i) {
            y[i] = start + static_cast<float>(i) * delta;
        }
        return;
    }

    const int64_t start = ReadScalar<int64_t>(inputs[0], dtype, "start", context);
    const int64_t limit = ReadScalar<int64_t>(inputs[1], dtype, "limit", context);
    const int64_t delta = ReadScalar<int64_t>(inputs[2], dtype, "delta", context);
    if (delta == 0) {
        throw InferenceException(context + ": delta must be non-zero");
    }
    int64_t expected = 0;
    if (delta > 0 && limit > start) {
        expected = (limit - start + delta - 1) / delta;
    } else if (delta < 0 && limit < start) {
        expected = (start - limit + (-delta) - 1) / (-delta);
    }
    if (outCount != expected) {
        throw InferenceException(context + ": output length does not match the range");
    }
    int64_t* y = static_cast<int64_t*>(out.data);
    if (outCount > 0 && y == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }
    for (int64_t i = 0; i < outCount; ++i) {
        y[i] = start + i * delta;
    }
}

} // namespace kernels
} // namespace engine
