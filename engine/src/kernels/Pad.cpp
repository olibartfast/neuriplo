// Pad: constant, reflect, or edge padding of float32 or int64 data.
//
// The int64 pads input lists every begin then every end for the padded axes
// (all axes, or those named by the optional axes input). Negative pads crop.

#include "Kernels.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace engine {
namespace kernels {
namespace {

enum class PadMode { Constant, Reflect, Edge };

int64_t ElementCount(const std::vector<int64_t>& dims, const std::string& context) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        count *= dim;
    }
    return count;
}

std::vector<int64_t> ReadInt64s(const TensorView& view, const char* what, const std::string& context) {
    if (view.dtype != DataType::Int64) {
        throw InferenceException(context + ": " + what + " must be int64");
    }
    int64_t count = 1;
    for (int64_t dim : view.dims) {
        count *= dim;
    }
    if (count > 0 && view.data == nullptr) {
        throw InferenceException(context + ": null " + what + " buffer");
    }
    const int64_t* values = static_cast<const int64_t*>(view.data);
    return std::vector<int64_t>(values, values + count);
}

// Maps a source position along one axis of extent `dim` into [0, dim), or -1 for constant fill.
int64_t MapPosition(int64_t pos, int64_t dim, PadMode mode) {
    if (pos >= 0 && pos < dim) {
        return pos;
    }
    switch (mode) {
    case PadMode::Constant:
        return -1;
    case PadMode::Edge:
        return pos < 0 ? 0 : dim - 1;
    case PadMode::Reflect: {
        if (dim == 1) {
            return 0;
        }
        const int64_t period = 2 * (dim - 1);
        int64_t m = pos % period;
        if (m < 0) {
            m += period;
        }
        return m < dim ? m : period - m;
    }
    }
    return -1;
}

template <typename T>
void PadTyped(const T* x, T* y, const std::vector<int64_t>& inDims, const std::vector<int64_t>& outDims,
              const std::vector<int64_t>& begins, PadMode mode, T fill, int64_t outCount) {
    std::vector<int64_t> outStrides(outDims.size(), 1);
    for (std::size_t i = outDims.size(); i-- > 1;) {
        outStrides[i - 1] = outStrides[i] * outDims[i];
    }
    std::vector<int64_t> inStrides(inDims.size(), 1);
    for (std::size_t i = inDims.size(); i-- > 1;) {
        inStrides[i - 1] = inStrides[i] * inDims[i];
    }
    for (int64_t i = 0; i < outCount; ++i) {
        int64_t remaining = i;
        int64_t offset = 0;
        bool filled = false;
        for (std::size_t j = 0; j < outDims.size(); ++j) {
            const int64_t coord = remaining / outStrides[j];
            remaining %= outStrides[j];
            const int64_t src = MapPosition(coord - begins[j], inDims[j], mode);
            if (src < 0) {
                filled = true;
                break;
            }
            offset += src * inStrides[j];
        }
        y[i] = filled ? fill : x[offset];
    }
}

} // namespace

void Pad(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Pad (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 4 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two to four inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if ((in.dtype != DataType::Float32 && in.dtype != DataType::Int64) || out.dtype != in.dtype) {
        throw InferenceException(context + ": data tensors must share dtype float32 or int64");
    }

    PadMode mode = PadMode::Constant;
    for (const Attribute& attr : node.attributes) {
        if (attr.name != "mode") {
            continue;
        }
        const std::string* value = std::get_if<std::string>(&attr.value);
        if (value == nullptr) {
            throw InferenceException(context + ": attribute 'mode' must be a string");
        }
        if (*value == "constant") {
            mode = PadMode::Constant;
        } else if (*value == "reflect") {
            mode = PadMode::Reflect;
        } else if (*value == "edge") {
            mode = PadMode::Edge;
        } else {
            throw InferenceException(context + ": unsupported mode '" + *value + "'");
        }
    }

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    ElementCount(in.dims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const std::vector<int64_t> pads = ReadInt64s(inputs[1], "pads", context);
    std::vector<int64_t> axes;
    if (inputs.size() == 4 && inputs[3].dtype != DataType::Unknown) {
        axes = ReadInt64s(inputs[3], "axes", context);
    } else {
        for (int64_t i = 0; i < rank; ++i) {
            axes.push_back(i);
        }
    }
    if (pads.size() != axes.size() * 2) {
        throw InferenceException(context + ": pads must hold two entries per padded axis");
    }

    std::vector<int64_t> begins(static_cast<std::size_t>(rank), 0);
    std::vector<int64_t> ends(static_cast<std::size_t>(rank), 0);
    std::vector<bool> seen(static_cast<std::size_t>(rank), false);
    for (std::size_t k = 0; k < axes.size(); ++k) {
        int64_t axis = axes[k];
        if (axis < -rank || axis >= rank) {
            throw InferenceException(context + ": axis out of range");
        }
        if (axis < 0) {
            axis += rank;
        }
        if (seen[static_cast<std::size_t>(axis)]) {
            throw InferenceException(context + ": duplicate axis");
        }
        seen[static_cast<std::size_t>(axis)] = true;
        begins[static_cast<std::size_t>(axis)] = pads[k];
        ends[static_cast<std::size_t>(axis)] = pads[axes.size() + k];
    }

    std::vector<int64_t> expectedDims = in.dims;
    for (std::size_t i = 0; i < expectedDims.size(); ++i) {
        expectedDims[i] += begins[i] + ends[i];
    }
    if (out.dims != expectedDims) {
        throw InferenceException(context + ": output shape does not match the pads");
    }
    const int64_t outCount = ElementCount(out.dims, context);

    const bool hasValue = inputs.size() >= 3 && inputs[2].dtype != DataType::Unknown;
    if (hasValue) {
        if (inputs[2].dtype != in.dtype) {
            throw InferenceException(context + ": constant_value must match the data dtype");
        }
        if (ElementCount(inputs[2].dims, context) != 1 || inputs[2].data == nullptr) {
            throw InferenceException(context + ": constant_value must be a scalar");
        }
    }

    if (in.dtype == DataType::Float32) {
        const float fill = hasValue ? *static_cast<const float*>(inputs[2].data) : 0.0f;
        PadTyped<float>(static_cast<const float*>(in.data), static_cast<float*>(out.data), in.dims, out.dims, begins,
                        mode, fill, outCount);
    } else {
        const int64_t fill = hasValue ? *static_cast<const int64_t*>(inputs[2].data) : 0;
        PadTyped<int64_t>(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), in.dims, out.dims,
                          begins, mode, fill, outCount);
    }
}

} // namespace kernels
} // namespace engine
