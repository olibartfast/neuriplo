// Transpose: permute the input axes per the `perm` attribute.
//
// An absent `perm` reverses the axes. A present `perm` must be a permutation
// of [0, rank). Data is float32 or int64.

#include "Kernels.hpp"

#include <algorithm>
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
void TransposeTyped(const T* x, T* y, const std::vector<int64_t>& perm, const std::vector<int64_t>& in_strides,
    const std::vector<int64_t>& out_strides, int64_t out_count, int64_t rank) {
    for (int64_t i = 0; i < out_count; ++i) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int64_t j = 0; j < rank; ++j) {
            const std::size_t uj = static_cast<std::size_t>(j);
            const int64_t coord = remaining / out_strides[uj];
            remaining %= out_strides[uj];
            offset += coord * in_strides[static_cast<std::size_t>(perm[uj])];
        }
        y[i] = x[offset];
    }
}

} // namespace

void Transpose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Transpose (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());

    std::vector<int64_t> perm;
    bool have_perm = false;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "perm") {
            const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'perm' must be an int list");
            }
            perm = *value;
            have_perm = true;
        }
    }
    if (!have_perm) {
        perm.resize(static_cast<std::size_t>(rank));
        for (int64_t i = 0; i < rank; ++i) {
            perm[static_cast<std::size_t>(i)] = rank - 1 - i;
        }
    }
    if (static_cast<int64_t>(perm.size()) != rank) {
        throw InferenceException(context + ": Transpose perm rank must match the input rank");
    }
    std::vector<int64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (int64_t i = 0; i < rank; ++i) {
        if (sorted[static_cast<std::size_t>(i)] != i) {
            throw InferenceException(context + ": Transpose perm must be a permutation");
        }
    }
    std::vector<int64_t> expected(static_cast<std::size_t>(rank), 0);
    for (int64_t i = 0; i < rank; ++i) {
        expected[static_cast<std::size_t>(i)] = in.dims[static_cast<std::size_t>(perm[i])];
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the transposed input");
    }

    const int64_t count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    const std::vector<int64_t> in_strides = Strides(in.dims);
    const std::vector<int64_t> out_strides = Strides(out.dims);
    if (in.dtype == DataType::Float32) {
        TransposeTyped<float>(static_cast<const float*>(in.data), static_cast<float*>(out.data), perm, in_strides,
            out_strides, count, rank);
    } else {
        TransposeTyped<int64_t>(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), perm,
            in_strides, out_strides, count, rank);
    }
}

} // namespace kernels
} // namespace engine
