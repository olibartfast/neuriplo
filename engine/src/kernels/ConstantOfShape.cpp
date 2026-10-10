// ConstantOfShape: fill a tensor of the requested shape with one value.
//
// The shape arrives as a rank-1 int64 input whose values must be non-negative;
// the output shape equals those values. The fill value is the scalar `value`
// attribute the loader pinned: an int64, float32, or bool scalar, which also
// fixes the output dtype. An absent `value` fills float32 zero (opset 18).

#include "Kernels.hpp"

#include <cstdint>
#include <string>
#include <variant>
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

void ConstantOfShape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
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

    const Attribute* value = nullptr;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "value") {
            value = &attr;
        }
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

    if (value == nullptr || std::holds_alternative<float>(value->value)) {
        const float fill = value == nullptr ? 0.0f : std::get<float>(value->value);
        if (out.dtype != DataType::Float32) {
            throw InferenceException(context + ": ConstantOfShape output must be float32 for a float value");
        }
        float* y = static_cast<float*>(out.data);
        for (int64_t i = 0; i < out_count; ++i) {
            y[i] = fill;
        }
        return;
    }
    if (const int64_t* fill = std::get_if<int64_t>(&value->value)) {
        if (out.dtype != DataType::Int64) {
            throw InferenceException(context + ": ConstantOfShape output must be int64 for an int64 value");
        }
        int64_t* y = static_cast<int64_t*>(out.data);
        for (int64_t i = 0; i < out_count; ++i) {
            y[i] = *fill;
        }
        return;
    }
    if (const bool* fill = std::get_if<bool>(&value->value)) {
        if (out.dtype != DataType::Bool) {
            throw InferenceException(context + ": ConstantOfShape output must be bool for a bool value");
        }
        bool* y = static_cast<bool*>(out.data);
        for (int64_t i = 0; i < out_count; ++i) {
            y[i] = *fill;
        }
        return;
    }
    throw InferenceException(context + ": ConstantOfShape 'value' must be an int64, float, or bool scalar");
}

} // namespace kernels
} // namespace engine
