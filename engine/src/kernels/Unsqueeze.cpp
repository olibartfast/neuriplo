// Unsqueeze: insert size-1 axes at the positions named by the int64 axes
// input (negative values count from the back of the output rank).
//
// The axes must be unique and within range; the data buffer is copied flat in
// order. Data is float32 or int64.

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

} // namespace

void Unsqueeze(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Unsqueeze (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input, an axes input, and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& axes_view = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    if (axes_view.dtype != DataType::Int64) {
        throw InferenceException(context + ": Unsqueeze requires an axes input");
    }
    const int64_t axes_count = ElementCount(axes_view.dims, context);
    if (axes_count > 0 && axes_view.data == nullptr) {
        throw InferenceException(context + ": null axes buffer");
    }
    const int64_t* raw = static_cast<const int64_t*>(axes_view.data);
    std::vector<int64_t> axes(raw, raw + axes_count);

    const int64_t rank_out = static_cast<int64_t>(in.dims.size()) + axes_count;
    std::vector<bool> seen(static_cast<std::size_t>(rank_out), false);
    for (int64_t& axis : axes) {
        if (axis < -rank_out || axis >= rank_out) {
            throw InferenceException(context + ": Unsqueeze axis is out of range");
        }
        if (axis < 0) {
            axis += rank_out;
        }
        if (seen[static_cast<std::size_t>(axis)]) {
            throw InferenceException(context + ": Unsqueeze axes must be unique");
        }
        seen[static_cast<std::size_t>(axis)] = true;
    }
    std::vector<int64_t> expected(static_cast<std::size_t>(rank_out), 0);
    std::size_t data_pos = 0;
    for (int64_t i = 0; i < rank_out; ++i) {
        if (seen[static_cast<std::size_t>(i)]) {
            expected[static_cast<std::size_t>(i)] = 1;
        } else {
            expected[static_cast<std::size_t>(i)] = in.dims[data_pos++];
        }
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the unsqueezed input");
    }

    const int64_t in_count = ElementCount(in.dims, context);
    const int64_t out_count = ElementCount(out.dims, context);
    if (in_count != out_count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    const std::size_t bytes =
        static_cast<std::size_t>(in_count) * (in.dtype == DataType::Float32 ? sizeof(float) : sizeof(int64_t));
    if (in_count > 0) {
        std::memcpy(out.data, in.data, bytes);
    }
}

} // namespace kernels
} // namespace engine
