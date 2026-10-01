// Expand: broadcast the input to the int64 shape input following NumPy
// right-alignment rules.
//
// Every input dimension must equal the corresponding target dimension or be
// one (missing leading axes broadcast as size one). Data is float32 or int64.

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
void ExpandTyped(const T* x, T* y, const std::vector<int64_t>& aligned, const std::vector<int64_t>& out_strides,
    int64_t out_count, int64_t out_rank) {
    for (int64_t i = 0; i < out_count; ++i) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int64_t j = 0; j < out_rank; ++j) {
            const std::size_t uj = static_cast<std::size_t>(j);
            const int64_t coord = remaining / out_strides[uj];
            remaining %= out_strides[uj];
            offset += coord * aligned[uj];
        }
        y[i] = x[offset];
    }
}

} // namespace

void Expand(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Expand (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input, a shape input, and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& shape_view = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    if (shape_view.dtype != DataType::Int64) {
        throw InferenceException(context + ": Expand requires a shape input");
    }
    const int64_t target_count = ElementCount(shape_view.dims, context);
    if (target_count > 0 && shape_view.data == nullptr) {
        throw InferenceException(context + ": null shape buffer");
    }
    const int64_t* raw = static_cast<const int64_t*>(shape_view.data);
    const std::vector<int64_t> target(raw, raw + target_count);

    const std::size_t rank = std::max(in.dims.size(), target.size());
    std::vector<int64_t> expected(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const int64_t di = static_cast<int64_t>(in.dims.size()) - static_cast<int64_t>(rank) + static_cast<int64_t>(i);
        const int64_t si = static_cast<int64_t>(target.size()) - static_cast<int64_t>(rank) + static_cast<int64_t>(i);
        const int64_t d = di < 0 ? 1 : in.dims[static_cast<std::size_t>(di)];
        if (si < 0) {
            expected[i] = d;
            continue;
        }
        const int64_t s = target[static_cast<std::size_t>(si)];
        if (s < 1) {
            throw InferenceException(context + ": Expand target shape must be positive");
        }
        if (d == s || di < 0 || d == 1) {
            expected[i] = s;
        } else {
            throw InferenceException(context + ": Expand input is not broadcastable to the target shape");
        }
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the expanded shape");
    }

    const int64_t out_count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    const std::vector<int64_t> in_strides = Strides(in.dims);
    const std::vector<int64_t> out_strides = Strides(out.dims);
    const int64_t out_rank = static_cast<int64_t>(rank);
    const int64_t in_rank = static_cast<int64_t>(in.dims.size());
    std::vector<int64_t> aligned(static_cast<std::size_t>(out_rank), 0);
    for (int64_t j = 0; j < out_rank; ++j) {
        const int64_t k = j - (out_rank - in_rank);
        if (k < 0 || in.dims[static_cast<std::size_t>(k)] == 1) {
            aligned[static_cast<std::size_t>(j)] = 0;
        } else {
            aligned[static_cast<std::size_t>(j)] = in_strides[static_cast<std::size_t>(k)];
        }
    }
    if (in.dtype == DataType::Float32) {
        ExpandTyped<float>(static_cast<const float*>(in.data), static_cast<float*>(out.data), aligned, out_strides,
            out_count, out_rank);
    } else {
        ExpandTyped<int64_t>(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), aligned,
            out_strides, out_count, out_rank);
    }
}

} // namespace kernels
} // namespace engine
