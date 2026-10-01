// Softmax: numerically stable softmax over one axis (subtract the max first).
//
// The `axis` attribute defaults to -1; negative values count back from the
// last axis.

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

// Reads an int64 attribute, or returns the fallback when the attribute is absent.
int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be an integer");
            }
            return *value;
        }
    }
    return fallback;
}

} // namespace

void Softmax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Softmax (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (in.dims != out.dims) {
        throw InferenceException(context + ": output shape must match the input shape");
    }

    (void)ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    int64_t axis = IntAttribute(node, "axis", -1, context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": axis out of range");
    }
    if (axis < 0) {
        axis += rank;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    const int64_t dim = rank == 0 ? 1 : in.dims[static_cast<std::size_t>(axis)];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }

    const float* x = static_cast<const float*>(in.data);
    float* y = static_cast<float*>(out.data);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t n = 0; n < inner; ++n) {
            float peak = x[(o * dim) * inner + n];
            for (int64_t k = 1; k < dim; ++k) {
                const float value = x[(o * dim + k) * inner + n];
                if (value > peak) {
                    peak = value;
                }
            }
            float total = 0.0F;
            for (int64_t k = 0; k < dim; ++k) {
                const float shifted = std::exp(x[(o * dim + k) * inner + n] - peak);
                y[(o * dim + k) * inner + n] = shifted;
                total += shifted;
            }
            for (int64_t k = 0; k < dim; ++k) {
                y[(o * dim + k) * inner + n] /= total;
            }
        }
    }
}

} // namespace kernels
} // namespace engine
