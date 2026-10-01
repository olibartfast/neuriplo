// Gather: select whole slices of the data along `axis` (negative values
// count from the back).
//
// The indices are int64; negative values wrap by the axis dim and anything
// still out of range throws. Data is float32 or int64.

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

template <typename T>
void GatherTyped(const T* x, const int64_t* idx, T* y, int64_t axis, int64_t outer, int64_t inner,
    int64_t index_count, int64_t dim, const std::string& context) {
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t j = 0; j < index_count; ++j) {
            int64_t k = idx[j];
            if (k < 0) {
                k += dim;
            }
            if (k < 0 || k >= dim) {
                throw InferenceException(context + ": Gather index is out of range");
            }
            for (int64_t n = 0; n < inner; ++n) {
                y[(o * index_count + j) * inner + n] = x[(o * dim + k) * inner + n];
            }
        }
    }
}

} // namespace

void Gather(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Gather (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected data and indices inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& indices = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    if (indices.dtype != DataType::Int64) {
        throw InferenceException(context + ": Gather indices must be int64");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": Gather axis is out of range");
    }
    int64_t axis = 0;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "axis") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'axis' must be an integer");
            }
            axis = *value;
        }
    }
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": Gather axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }

    std::vector<int64_t> expected;
    for (int64_t i = 0; i < axis; ++i) {
        expected.push_back(in.dims[static_cast<std::size_t>(i)]);
    }
    expected.insert(expected.end(), indices.dims.begin(), indices.dims.end());
    for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < in.dims.size(); ++i) {
        expected.push_back(in.dims[i]);
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the gathered indices");
    }

    const int64_t index_count = ElementCount(indices.dims, context);
    const int64_t out_count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (index_count > 0 && indices.data == nullptr) {
        throw InferenceException(context + ": null indices buffer");
    }
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }
    const int64_t* idx = static_cast<const int64_t*>(indices.data);
    if (in.dtype == DataType::Float32) {
        GatherTyped<float>(static_cast<const float*>(in.data), idx, static_cast<float*>(out.data), axis, outer, inner,
            index_count, dim, context);
    } else {
        GatherTyped<int64_t>(static_cast<const int64_t*>(in.data), idx, static_cast<int64_t*>(out.data), axis, outer,
            inner, index_count, dim, context);
    }
}

} // namespace kernels
} // namespace engine
