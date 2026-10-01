// Slice: extract strided ranges per the opset-18 input form
// (data, starts, ends, axes, steps).
//
// The starts/ends/steps are int64 inputs; axes defaults to [0, count) and
// steps to all ones. Steps must be positive. Negative begins/ends wrap by the
// axis dim and both clamp to [0, dim], exactly as the shape rule does. Data
// is float32 or int64.

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

std::vector<int64_t> ReadInt64s(const TensorView& view, const std::string& what, const std::string& context) {
    if (view.dtype != DataType::Int64) {
        throw InferenceException(context + ": Slice " + what + " must be int64");
    }
    const int64_t count = ElementCount(view.dims, context);
    if (count > 0 && view.data == nullptr) {
        throw InferenceException(context + ": null " + what + " buffer");
    }
    const int64_t* raw = static_cast<const int64_t*>(view.data);
    return std::vector<int64_t>(raw, raw + count);
}

template <typename T>
void SliceTyped(const T* x, T* y, const std::vector<int64_t>& begins, const std::vector<int64_t>& steps,
    const std::vector<int64_t>& in_strides, const std::vector<int64_t>& out_strides, int64_t out_count,
    int64_t rank) {
    for (int64_t i = 0; i < out_count; ++i) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int64_t j = 0; j < rank; ++j) {
            const std::size_t uj = static_cast<std::size_t>(j);
            const int64_t coord = remaining / out_strides[uj];
            remaining %= out_strides[uj];
            offset += (begins[uj] + coord * steps[uj]) * in_strides[uj];
        }
        y[i] = x[offset];
    }
}

} // namespace

void Slice(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Slice (node '" + node.name + "')";
    if (inputs.size() < 3 || inputs.size() > 5 || outputs.size() != 1) {
        throw InferenceException(context + ": expected data, starts, ends, and optional axes/steps inputs");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const std::vector<int64_t> starts = ReadInt64s(inputs[1], "starts", context);
    const std::vector<int64_t> ends = ReadInt64s(inputs[2], "ends", context);
    if (starts.size() != ends.size()) {
        throw InferenceException(context + ": Slice starts and ends must have the same length");
    }
    const std::size_t count = starts.size();

    std::vector<int64_t> axes(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
        axes[i] = static_cast<int64_t>(i);
    }
    if (inputs.size() > 3 && inputs[3].dtype != DataType::Unknown) {
        axes = ReadInt64s(inputs[3], "axes", context);
        if (axes.size() != count) {
            throw InferenceException(context + ": Slice axes must match starts in length");
        }
    }
    std::vector<int64_t> steps(count, 1);
    if (inputs.size() > 4 && inputs[4].dtype != DataType::Unknown) {
        steps = ReadInt64s(inputs[4], "steps", context);
        if (steps.size() != count) {
            throw InferenceException(context + ": Slice steps must match starts in length");
        }
    }

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    std::vector<int64_t> begins(static_cast<std::size_t>(rank), 0);
    std::vector<int64_t> strides(static_cast<std::size_t>(rank), 1);
    std::vector<int64_t> expected = in.dims;
    for (std::size_t i = 0; i < count; ++i) {
        if (steps[i] <= 0) {
            throw InferenceException(context + ": Slice steps must be positive");
        }
        int64_t axis = axes[i];
        if (axis < -rank || axis >= rank) {
            throw InferenceException(context + ": Slice axis is out of range");
        }
        if (axis < 0) {
            axis += rank;
        }
        const std::size_t ua = static_cast<std::size_t>(axis);
        const int64_t dim = in.dims[ua];
        int64_t begin = starts[i];
        int64_t end = ends[i];
        if (begin < 0) {
            begin += dim;
        }
        if (end < 0) {
            end += dim;
        }
        begin = std::max<int64_t>(0, std::min(begin, dim));
        end = std::max<int64_t>(0, std::min(end, dim));
        begins[ua] = begin;
        strides[ua] = steps[i];
        expected[ua] = end > begin ? (end - begin + steps[i] - 1) / steps[i] : 0;
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the sliced ranges");
    }

    // Empty selections carry no elements; only the shape contract applies.
    int64_t out_count = 1;
    for (const int64_t d : out.dims) {
        if (d < 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        out_count *= d;
    }
    (void)ElementCount(in.dims, context);
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    const std::vector<int64_t> in_strides = Strides(in.dims);
    const std::vector<int64_t> out_strides = Strides(out.dims);
    if (in.dtype == DataType::Float32) {
        SliceTyped<float>(static_cast<const float*>(in.data), static_cast<float*>(out.data), begins, strides,
            in_strides, out_strides, out_count, rank);
    } else {
        SliceTyped<int64_t>(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), begins, strides,
            in_strides, out_strides, out_count, rank);
    }
}

} // namespace kernels
} // namespace engine
