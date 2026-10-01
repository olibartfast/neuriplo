// Reshape: copy the input flat buffer to the output in order.

#include "Kernels.hpp"

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

} // namespace

void Reshape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Reshape (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one or two inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": data tensors must be float32");
    }

    const int64_t inCount = ElementCount(in.dims, context);
    const int64_t outCount = ElementCount(out.dims, context);
    if (inCount != outCount) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    if (inputs.size() == 2) {
        const TensorView& shape = inputs[1];
        if (shape.dtype != DataType::Int64) {
            throw InferenceException(context + ": shape operand must be int64");
        }
        const int64_t shapeCount = ElementCount(shape.dims, context);
        if (shapeCount > 0 && shape.data == nullptr) {
            throw InferenceException(context + ": null shape buffer");
        }
        const int64_t* values = static_cast<const int64_t*>(shape.data);
        int64_t product = 1;
        for (int64_t i = 0; i < shapeCount; ++i) {
            product *= values[i];
        }
        if (product != inCount) {
            throw InferenceException(context + ": shape operand product does not match the element count");
        }
    }

    const float* x = static_cast<const float*>(in.data);
    float* y = static_cast<float*>(out.data);
    for (int64_t i = 0; i < inCount; ++i) {
        y[i] = x[i];
    }
}

} // namespace kernels
} // namespace engine
