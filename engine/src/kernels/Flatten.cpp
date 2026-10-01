// Flatten: collapse the input into [prod(dims[:axis]), prod(dims[axis:])].
//
// The `axis` attribute defaults to 1; negative values count back from
// rank + 1, so -1 keeps a trailing singleton. The buffer copies flat in
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

void Flatten(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Flatten (node '" + node.name + "')";
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
    if (axis < 0) {
        axis += rank + 1;
    }
    if (axis < 0 || axis > rank) {
        throw InferenceException(context + ": Flatten axis is out of range");
    }
    int64_t first = 1;
    for (int64_t i = 0; i < axis; ++i) {
        first *= in.dims[static_cast<std::size_t>(i)];
    }
    int64_t second = 1;
    for (std::size_t i = static_cast<std::size_t>(axis); i < in.dims.size(); ++i) {
        second *= in.dims[i];
    }
    if (out.dims != std::vector<int64_t>{first, second}) {
        throw InferenceException(context + ": output shape does not match the flattened input");
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
