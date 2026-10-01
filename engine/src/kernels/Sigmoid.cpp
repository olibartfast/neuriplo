// Sigmoid: elementwise 1 / (1 + exp(-x)) over the output element count.

#include "Kernels.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace engine {
namespace kernels {
namespace {

// Product of the dimensions; every dimension must be positive. An empty shape
// is a scalar and yields one element.
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

} // namespace

void Sigmoid(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Sigmoid (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }

    const int64_t outCount = ElementCount(out.dims, context);
    const int64_t inCount = ElementCount(in.dims, context);
    if (inCount != outCount) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const float* x = static_cast<const float*>(in.data);
    float* y = static_cast<float*>(out.data);
    for (int64_t i = 0; i < outCount; ++i) {
        y[i] = 1.0F / (1.0F + std::exp(-x[i]));
    }
}

} // namespace kernels
} // namespace engine
