// Not: elementwise logical negation of a bool tensor; the output keeps the input shape.

#include "Kernels.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace engine {
namespace kernels {

void Not(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Not (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Bool || out.dtype != DataType::Bool) {
        throw InferenceException(context + ": input and output must be bool");
    }
    if (in.dims != out.dims) {
        throw InferenceException(context + ": output shape must match the input shape");
    }
    int64_t count = 1;
    for (int64_t dim : in.dims) {
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        count *= dim;
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const bool* x = static_cast<const bool*>(in.data);
    bool* y = static_cast<bool*>(out.data);
    for (int64_t i = 0; i < count; ++i) {
        y[i] = !x[i];
    }
}

} // namespace kernels
} // namespace engine
