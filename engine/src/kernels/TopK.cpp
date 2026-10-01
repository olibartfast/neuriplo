// TopK: select the K largest elements along `axis` in descending order.
//
// The K arrives as a scalar int64 input; it must be positive and fit the axis
// dim (a larger K throws, matching the shape rule). Only largest=1 and
// sorted=1 are accepted here. Values are float32, indices are int64.

#include "Kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
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

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

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

void TopK(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "TopK (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 2) {
        throw InferenceException(context + ": expected an input, a K input, and values/indices outputs");
    }
    const TensorView& in = inputs[0];
    const TensorView& k_view = inputs[1];
    const TensorView& values = outputs[0];
    const TensorView& indices = outputs[1];
    if (in.dtype != DataType::Float32) {
        throw InferenceException(context + ": TopK input must be float32");
    }
    if (values.dtype != DataType::Float32) {
        throw InferenceException(context + ": TopK values must be float32");
    }
    if (indices.dtype != DataType::Int64) {
        throw InferenceException(context + ": TopK indices must be int64");
    }
    if (k_view.dtype != DataType::Int64) {
        throw InferenceException(context + ": TopK requires a K input");
    }
    if (IntAttribute(node, "largest", 1, context) != 1) {
        throw InferenceException(context + ": TopK 'largest' must be 1 here");
    }
    if (IntAttribute(node, "sorted", 1, context) != 1) {
        throw InferenceException(context + ": TopK 'sorted' must be 1 here");
    }

    if (ElementCount(k_view.dims, context) != 1) {
        throw InferenceException(context + ": TopK K input must be a scalar Int64 constant");
    }
    if (k_view.data == nullptr) {
        throw InferenceException(context + ": null K buffer");
    }
    const int64_t k = *static_cast<const int64_t*>(k_view.data);
    if (k <= 0) {
        throw InferenceException(context + ": TopK K must be positive");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    int64_t axis = IntAttribute(node, "axis", -1, context);
    if (rank == 0 || axis < -rank || axis >= rank) {
        throw InferenceException(context + ": TopK axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];
    if (k > dim) {
        throw InferenceException(context + ": TopK K exceeds the axis dim");
    }
    std::vector<int64_t> expected = in.dims;
    expected[static_cast<std::size_t>(axis)] = k;
    if (values.dims != expected || indices.dims != expected) {
        throw InferenceException(context + ": output shapes do not match the selected K");
    }

    const int64_t values_count = ElementCount(values.dims, context);
    (void)ElementCount(in.dims, context);
    if (values_count > 0 && (in.data == nullptr || values.data == nullptr || indices.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }
    const float* x = static_cast<const float*>(in.data);
    float* y = static_cast<float*>(values.data);
    int64_t* iy = static_cast<int64_t*>(indices.data);
    std::vector<int64_t> order(static_cast<std::size_t>(dim), 0);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t n = 0; n < inner; ++n) {
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                const float va = x[(o * dim + a) * inner + n];
                const float vb = x[(o * dim + b) * inner + n];
                return va != vb ? va > vb : a < b;
            });
            for (int64_t t = 0; t < k; ++t) {
                const int64_t picked = order[static_cast<std::size_t>(t)];
                y[(o * k + t) * inner + n] = x[(o * dim + picked) * inner + n];
                iy[(o * k + t) * inner + n] = picked;
            }
        }
    }
}

} // namespace kernels
} // namespace engine
