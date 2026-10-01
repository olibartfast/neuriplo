// Resize: nearest-neighbor upsampling per the pinned policy (mode nearest,
// coordinate_transformation_mode asymmetric, nearest_mode floor).
//
// The scales arrive as a float32 input with one entry per axis; the output dim
// is floor(in * scale). A sizes input is rejected here: the fixture carries
// scales only. Data is float32 or int64.

#include "Kernels.hpp"

#include <cmath>
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

const Attribute* FindAttr(const Node& node, const std::string& name) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            return &attr;
        }
    }
    return nullptr;
}

void RequireMode(const Node& node, const std::string& name, const std::string& want, const std::string& context) {
    const Attribute* attr = FindAttr(node, name);
    if (attr == nullptr) {
        return;
    }
    const std::string* value = std::get_if<std::string>(&attr->value);
    if (value == nullptr || *value != want) {
        throw InferenceException(context + ": Resize '" + name + "' must be '" + want + "' here");
    }
}

template <typename T>
void ResizeTyped(const T* x, T* y, const std::vector<int64_t>& in_dims, const std::vector<int64_t>& in_strides,
    const std::vector<int64_t>& out_strides, const std::vector<float>& scales, int64_t out_count, int64_t rank) {
    for (int64_t i = 0; i < out_count; ++i) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int64_t j = 0; j < rank; ++j) {
            const std::size_t uj = static_cast<std::size_t>(j);
            const int64_t coord = remaining / out_strides[uj];
            remaining %= out_strides[uj];
            const double scale = static_cast<double>(scales[uj]);
            int64_t src = static_cast<int64_t>(std::floor(static_cast<double>(coord) / scale));
            if (src < 0) {
                src = 0;
            }
            if (src >= in_dims[uj]) {
                src = in_dims[uj] - 1;
            }
            offset += src * in_strides[uj];
        }
        y[i] = x[offset];
    }
}

} // namespace

void Resize(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Resize (node '" + node.name + "')";
    if (inputs.size() < 3 || inputs.size() > 4 || outputs.size() != 1) {
        throw InferenceException(context + ": expected an input, roi, scales, and an optional sizes input");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    if (inputs.size() > 3 && inputs[3].dtype != DataType::Unknown) {
        throw InferenceException(context + ": Resize takes scales, not sizes, here");
    }
    const TensorView& scales_view = inputs[2];
    if (scales_view.dtype != DataType::Float32) {
        throw InferenceException(context + ": Resize requires a scales input");
    }
    RequireMode(node, "mode", "nearest", context);
    RequireMode(node, "coordinate_transformation_mode", "asymmetric", context);
    RequireMode(node, "nearest_mode", "floor", context);

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    const int64_t scales_count = ElementCount(scales_view.dims, context);
    if (scales_count > 0 && scales_view.data == nullptr) {
        throw InferenceException(context + ": null scales buffer");
    }
    if (scales_count != rank) {
        throw InferenceException(context + ": Resize scales rank must match the input rank");
    }
    const float* scales = static_cast<const float*>(scales_view.data);
    std::vector<int64_t> expected(static_cast<std::size_t>(rank), 0);
    for (int64_t i = 0; i < rank; ++i) {
        const std::size_t ui = static_cast<std::size_t>(i);
        const int64_t dim = static_cast<int64_t>(
            std::floor(static_cast<double>(in.dims[ui]) * static_cast<double>(scales[ui])));
        if (dim <= 0) {
            throw InferenceException(context + ": Resize produces a non-positive output dim");
        }
        expected[ui] = dim;
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the scaled input");
    }

    const int64_t out_count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    const std::vector<int64_t> in_strides = Strides(in.dims);
    const std::vector<int64_t> out_strides = Strides(out.dims);
    const std::vector<float> scale_vec(scales, scales + rank);
    if (in.dtype == DataType::Float32) {
        ResizeTyped<float>(static_cast<const float*>(in.data), static_cast<float*>(out.data), in.dims, in_strides,
            out_strides, scale_vec, out_count, rank);
    } else {
        ResizeTyped<int64_t>(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), in.dims,
            in_strides, out_strides, scale_vec, out_count, rank);
    }
}

} // namespace kernels
} // namespace engine
