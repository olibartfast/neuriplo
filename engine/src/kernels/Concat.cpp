// Concat: join the inputs along `axis` (negative values count from the back).
//
// Every input shares rank and element type (float32 or int64); every dimension
// off the concat axis must agree. The output's axis extent is the sum of the
// inputs' extents along that axis.

#include "Kernels.hpp"

#include <algorithm>
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
        if (dim < 0) {
            throw InferenceException(context + ": negative dimension");
        }
        count *= dim;
    }
    return count;
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

template <typename T>
void ConcatTyped(const std::vector<TensorView>& inputs, T* out, int64_t axis, int64_t outer, int64_t inner,
    int64_t out_extent) {
    int64_t written = 0;
    for (const TensorView& view : inputs) {
        const T* x = static_cast<const T*>(view.data);
        const int64_t extent = view.dims[static_cast<std::size_t>(axis)];
        if (extent == 0) {
            continue;
        }
        for (int64_t o = 0; o < outer; ++o) {
            const T* src = x + o * extent * inner;
            T* dst = out + (o * out_extent + written) * inner;
            std::memcpy(dst, src, static_cast<std::size_t>(extent * inner) * sizeof(T));
        }
        written += extent;
    }
}

} // namespace

void Concat(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Concat (node '" + node.name + "')";
    if (inputs.empty() || outputs.size() != 1) {
        throw InferenceException(context + ": expected at least one input and one output");
    }
    const TensorView& first = inputs[0];
    if (first.dtype != DataType::Float32 && first.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    const TensorView& out = outputs[0];
    if (out.dtype != first.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const int64_t rank = static_cast<int64_t>(first.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": Concat axis is out of range");
    }
    int64_t axis = IntAttribute(node, "axis", 1, context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": Concat axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }

    std::vector<int64_t> expected = first.dims;
    for (const TensorView& view : inputs) {
        if (view.dtype != first.dtype) {
            throw InferenceException(context + ": all inputs must share the element type");
        }
        if (static_cast<int64_t>(view.dims.size()) != rank) {
            throw InferenceException(context + ": Concat inputs must share rank");
        }
        for (int64_t j = 0; j < rank; ++j) {
            if (j == axis) {
                continue;
            }
            if (view.dims[static_cast<std::size_t>(j)] != expected[static_cast<std::size_t>(j)]) {
                throw InferenceException(context + ": Concat inputs differ off the concat axis");
            }
        }
        expected[static_cast<std::size_t>(axis)] += view.dims[static_cast<std::size_t>(axis)];
    }
    // The first input's axis extent was counted twice above; correct it.
    expected[static_cast<std::size_t>(axis)] -= first.dims[static_cast<std::size_t>(axis)];
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the concatenated inputs");
    }

    for (const TensorView& view : inputs) {
        (void)ElementCount(view.dims, context);
        if (ElementCount(view.dims, context) > 0 && view.data == nullptr) {
            throw InferenceException(context + ": null input buffer");
        }
    }
    const int64_t out_count = ElementCount(out.dims, context);
    if (out_count > 0 && out.data == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= expected[static_cast<std::size_t>(i)];
    }
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= expected[static_cast<std::size_t>(i)];
    }
    const int64_t out_extent = expected[static_cast<std::size_t>(axis)];
    if (first.dtype == DataType::Float32) {
        ConcatTyped<float>(inputs, static_cast<float*>(out.data), axis, outer, inner, out_extent);
    } else {
        ConcatTyped<int64_t>(inputs, static_cast<int64_t*>(out.data), axis, outer, inner, out_extent);
    }
}

} // namespace kernels
} // namespace engine
