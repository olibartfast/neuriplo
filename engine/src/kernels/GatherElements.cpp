// GatherElements: pick one element per index along `axis` (negative values
// count from the back).
//
// Data and indices share rank; the output shape equals the indices shape. The
// indices are int64; negative values wrap by the axis dim and anything still
// out of range throws. Data is float32 or int64.

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

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

template <typename T>
void GatherElementsTyped(const T* x, const int64_t* idx, T* y, const std::vector<int64_t>& strides,
    const std::vector<int64_t>& in_strides, const std::vector<int64_t>& in_dims, int64_t axis, int64_t dim,
    int64_t count, int64_t rank, const std::string& context) {
    for (int64_t i = 0; i < count; ++i) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int64_t j = 0; j < rank; ++j) {
            const std::size_t uj = static_cast<std::size_t>(j);
            const int64_t coord = remaining / strides[uj];
            remaining %= strides[uj];
            if (j != axis) {
                if (coord >= in_dims[uj]) {
                    throw InferenceException(context + ": GatherElements index is out of range");
                }
                offset += coord * in_strides[uj];
            }
        }
        int64_t k = idx[i];
        if (k < 0) {
            k += dim;
        }
        if (k < 0 || k >= dim) {
            throw InferenceException(context + ": GatherElements index is out of range");
        }
        offset += k * strides[static_cast<std::size_t>(axis)];
        y[i] = x[offset];
    }
}

} // namespace

void GatherElements(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "GatherElements (node '" + node.name + "')";
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
        throw InferenceException(context + ": GatherElements indices must be int64");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (static_cast<int64_t>(indices.dims.size()) != rank) {
        throw InferenceException(context + ": GatherElements data and indices must share rank");
    }
    int64_t axis = 1;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "axis") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'axis' must be an integer");
            }
            axis = *value;
        }
    }
    if (rank == 0 || axis < -rank || axis >= rank) {
        throw InferenceException(context + ": GatherElements axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }
    if (out.dims != indices.dims) {
        throw InferenceException(context + ": output shape must match the indices shape");
    }

    const int64_t count = ElementCount(indices.dims, context);
    (void)ElementCount(in.dims, context);
    (void)ElementCount(out.dims, context);
    if (count > 0 && indices.data == nullptr) {
        throw InferenceException(context + ": null indices buffer");
    }
    if (count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    const std::vector<int64_t> strides = Strides(indices.dims);
    const std::vector<int64_t> in_strides = Strides(in.dims);
    // The data and indices share rank; positions decompose on the indices
    // grid and re-anchor onto the data grid along the gather axis.
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];
    const int64_t* idx = static_cast<const int64_t*>(indices.data);
    if (in.dtype == DataType::Float32) {
        GatherElementsTyped<float>(static_cast<const float*>(in.data), idx, static_cast<float*>(out.data), strides,
            in_strides, in.dims, axis, dim, count, rank, context);
    } else {
        GatherElementsTyped<int64_t>(static_cast<const int64_t*>(in.data), idx, static_cast<int64_t*>(out.data),
            strides, in_strides, in.dims, axis, dim, count, rank, context);
    }
}

} // namespace kernels
} // namespace engine
