// ConstantOfShape: fill an int64 tensor of the requested shape.
//
// The shape arrives as a rank-1 int64 input whose values must be non-negative;
// the output shape equals those values. The fill value is the scalar int64
// `value` attribute the loader pinned.

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
        if (dim < 0) {
            throw InferenceException(context + ": negative dimension");
        }
        count *= dim;
    }
    return count;
}

} // namespace

void ConstantOfShape(const Node& node, const std::vector<TensorView>& inputs,
    const std::vector<TensorView>& outputs) {
    const std::string context = "ConstantOfShape (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a shape input and one output");
    }
    const TensorView& shape_view = inputs[0];
    const TensorView& out = outputs[0];
    if (shape_view.dtype != DataType::Int64) {
        throw InferenceException(context + ": ConstantOfShape requires a shape input");
    }
    if (shape_view.dims.size() != 1) {
        throw InferenceException(context + ": ConstantOfShape shape input must be rank 1");
    }
    const int64_t extent = ElementCount(shape_view.dims, context);
    if (extent > 0 && shape_view.data == nullptr) {
        throw InferenceException(context + ": null shape buffer");
    }
    const int64_t* raw = static_cast<const int64_t*>(shape_view.data);
    const std::vector<int64_t> shape(raw, raw + extent);
    for (const int64_t v : shape) {
        if (v < 0) {
            throw InferenceException(context + ": ConstantOfShape shape values must be non-negative");
        }
    }

    int64_t fill = 0;
    bool have_value = false;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "value") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": ConstantOfShape 'value' must be an int64 scalar");
            }
            fill = *value;
            have_value = true;
        }
    }
    if (!have_value) {
        throw InferenceException(context + ": ConstantOfShape requires a 'value' attribute");
    }

    if (out.dtype != DataType::Int64) {
        throw InferenceException(context + ": ConstantOfShape output must be int64");
    }
    if (out.dims != shape) {
        throw InferenceException(context + ": output shape does not match the shape input");
    }
    int64_t out_count = 1;
    for (const int64_t d : out.dims) {
        out_count *= d;
    }
    if (out_count > 0 && out.data == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }
    int64_t* y = static_cast<int64_t*>(out.data);
    for (int64_t i = 0; i < out_count; ++i) {
        y[i] = fill;
    }
}

} // namespace kernels
} // namespace engine
