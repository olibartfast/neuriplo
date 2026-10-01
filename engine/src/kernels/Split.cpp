// Split: divide the input along `axis` into one output per part.
//
// The part sizes arrive as an optional int64 input; when it is absent the axis
// dim divides evenly across the outputs. Unequal parts are allowed when the
// sizes input is present. Data is float32 or int64.

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
void SplitTyped(const TensorView& in, const std::vector<TensorView>& outputs, const std::vector<int64_t>& parts,
    int64_t axis, int64_t outer, int64_t inner) {
    const T* x = static_cast<const T*>(in.data);
    int64_t begin = 0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        T* y = static_cast<T*>(outputs[i].data);
        const int64_t extent = parts[i];
        if (extent == 0) {
            continue;
        }
        for (int64_t o = 0; o < outer; ++o) {
            const T* src = x + (o * in.dims[static_cast<std::size_t>(axis)] + begin) * inner;
            T* dst = y + o * extent * inner;
            std::memcpy(dst, src, static_cast<std::size_t>(extent * inner) * sizeof(T));
        }
        begin += extent;
    }
}

} // namespace

void Split(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Split (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 2 || outputs.empty()) {
        throw InferenceException(context + ": expected one or two inputs and at least one output");
    }
    const TensorView& in = inputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    for (const TensorView& view : outputs) {
        if (view.dtype != in.dtype) {
            throw InferenceException(context + ": output dtype must match the input dtype");
        }
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": Split axis is out of range");
    }
    int64_t axis = IntAttribute(node, "axis", 0, context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": Split axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];

    std::vector<int64_t> parts;
    if (inputs.size() == 2 && inputs[1].dtype != DataType::Unknown) {
        const TensorView& sizes = inputs[1];
        if (sizes.dtype != DataType::Int64) {
            throw InferenceException(context + ": split sizes must be int64");
        }
        const int64_t count = ElementCount(sizes.dims, context);
        if (count > 0 && sizes.data == nullptr) {
            throw InferenceException(context + ": null split-sizes buffer");
        }
        const int64_t* values = static_cast<const int64_t*>(sizes.data);
        parts.assign(values, values + count);
        if (parts.size() != outputs.size()) {
            throw InferenceException(context + ": Split split sizes must match the output count");
        }
        int64_t sum = 0;
        for (const int64_t v : parts) {
            if (v < 0) {
                throw InferenceException(context + ": Split split sizes must be non-negative");
            }
            sum += v;
        }
        if (sum != dim) {
            throw InferenceException(context + ": Split split sizes must sum to the axis dim");
        }
    } else {
        if (dim % static_cast<int64_t>(outputs.size()) != 0) {
            throw InferenceException(context + ": Split axis dim is not divisible by the output count");
        }
        parts.assign(outputs.size(), dim / static_cast<int64_t>(outputs.size()));
    }

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::vector<int64_t> want = in.dims;
        want[static_cast<std::size_t>(axis)] = parts[i];
        if (outputs[i].dims != want) {
            throw InferenceException(context + ": output shape does not match the split sizes");
        }
        if (ElementCount(outputs[i].dims, context) > 0 && outputs[i].data == nullptr) {
            throw InferenceException(context + ": null output buffer");
        }
    }
    if (ElementCount(in.dims, context) > 0 && in.data == nullptr) {
        throw InferenceException(context + ": null input buffer");
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }
    if (in.dtype == DataType::Float32) {
        SplitTyped<float>(in, outputs, parts, axis, outer, inner);
    } else {
        SplitTyped<int64_t>(in, outputs, parts, axis, outer, inner);
    }
}

} // namespace kernels
} // namespace engine
