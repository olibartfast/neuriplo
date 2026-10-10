// Reshape: copy the input flat buffer to the output in order.
//
// When a shape operand is present, the effective shape is resolved exactly as
// `reshape_shape` in ShapeInference.cpp does: a 0 copies the input dimension
// at the same position unless `allowzero` is set, at most one -1 is inferred
// from the element count, and the final product must equal the element count.

#include "Kernels.hpp"

#include <cstdint>
#include <cstring>
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
    if (in.dtype != out.dtype ||
        (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool)) {
        throw InferenceException(context + ": data tensors must share one float32, int64, or bool dtype");
    }

    bool allowzero = false;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "allowzero") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'allowzero' must be an integer");
            }
            allowzero = (*value != 0);
        }
    }

    const int64_t inCount = ElementCount(in.dims, context);
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
        std::vector<int64_t> target;
        for (int64_t i = 0; i < shapeCount; ++i) {
            target.push_back(values[i]);
        }

        // Mirror of reshape_shape in ShapeInference.cpp.
        std::vector<int64_t> effective(target.size(), 0);
        int64_t inferred = -1;
        for (size_t i = 0; i < target.size(); ++i) {
            const int64_t v = target[i];
            if (v == 0 && !allowzero) {
                if (i >= in.dims.size()) {
                    throw InferenceException(context + ": Reshape 0 has no matching input dimension");
                }
                effective[i] = in.dims[i];
            } else if (v == -1) {
                if (inferred >= 0) {
                    throw InferenceException(context + ": Reshape allows at most one -1");
                }
                inferred = static_cast<int64_t>(i);
            } else if (v < 0) {
                throw InferenceException(context + ": Reshape target dimensions must be positive");
            } else {
                effective[i] = v;
            }
        }

        int64_t known = 1;
        for (size_t i = 0; i < effective.size(); ++i) {
            if (static_cast<int64_t>(i) == inferred) {
                continue;
            }
            known *= effective[i];
        }
        if (inferred >= 0) {
            if (known == 0 || inCount % known != 0) {
                throw InferenceException(context + ": Reshape target product does not match input elements");
            }
            effective[static_cast<size_t>(inferred)] = inCount / known;
        }

        int64_t product = 1;
        for (int64_t dim : effective) {
            product *= dim;
        }
        if (product != inCount) {
            throw InferenceException(context + ": Reshape target product does not match input elements");
        }
        if (out.dims != effective) {
            throw InferenceException(context + ": output shape does not match the resolved shape operand");
        }
    }

    const int64_t outCount = ElementCount(out.dims, context);
    if (inCount != outCount) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const std::size_t element = in.dtype == DataType::Float32 ? sizeof(float)
                                : in.dtype == DataType::Int64 ? sizeof(int64_t)
                                                              : sizeof(bool);
    std::memcpy(out.data, in.data, static_cast<std::size_t>(inCount) * element);
}

} // namespace kernels
} // namespace engine
