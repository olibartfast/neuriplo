// ReduceMax: maximum over the requested axes.
//
// The axes arrive as an optional int64 input; an absent axes input reduces
// every axis. The `keepdims` attribute defaults to 1.

#include "Kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace engine {
namespace kernels {
namespace {

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

// Reads an int64 attribute, or returns the fallback when the attribute is absent.
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

} // namespace

void ReduceMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ReduceMax (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one or two inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": data tensors must be float32");
    }

    const std::vector<int64_t>& inDims = in.dims;
    const int64_t rank = static_cast<int64_t>(inDims.size());
    const int64_t inCount = ElementCount(inDims, context);
    if (in.data == nullptr) {
        throw InferenceException(context + ": null input buffer");
    }

    const int64_t keepdims = IntAttribute(node, "keepdims", 1, context);

    // Resolve the axes: an absent axes input means every axis.
    std::vector<int64_t> axes;
    if (inputs.size() == 2) {
        const TensorView& axesView = inputs[1];
        if (axesView.dtype != DataType::Int64) {
            throw InferenceException(context + ": axes tensor must be int64");
        }
        const int64_t axesCount = ElementCount(axesView.dims, context);
        if (axesCount > 0 && axesView.data == nullptr) {
            throw InferenceException(context + ": null axes buffer");
        }
        if (axesCount == 0) {
            for (int64_t i = 0; i < rank; ++i) {
                axes.push_back(i);
            }
        } else {
            const int64_t* values = static_cast<const int64_t*>(axesView.data);
            axes.assign(values, values + axesCount);
        }
    } else {
        for (int64_t i = 0; i < rank; ++i) {
            axes.push_back(i);
        }
    }

    // Normalize and validate the axes: unique and within [-rank, rank - 1].
    std::vector<bool> reduced(static_cast<std::size_t>(rank), false);
    for (int64_t axis : axes) {
        if (axis < -rank || axis >= rank) {
            throw InferenceException(context + ": axis out of range");
        }
        const int64_t normalized = axis < 0 ? axis + rank : axis;
        if (reduced[static_cast<std::size_t>(normalized)]) {
            throw InferenceException(context + ": duplicate axis");
        }
        reduced[static_cast<std::size_t>(normalized)] = true;
    }

    std::vector<int64_t> expectedDims;
    for (int64_t i = 0; i < rank; ++i) {
        if (reduced[static_cast<std::size_t>(i)]) {
            if (keepdims != 0) {
                expectedDims.push_back(1);
            }
        } else {
            expectedDims.push_back(inDims[static_cast<std::size_t>(i)]);
        }
    }
    if (out.dims != expectedDims) {
        throw InferenceException(context + ": output shape does not match the reduced axes");
    }
    const int64_t outCount = ElementCount(out.dims, context);

    // Contiguous row-major strides for the input and the output.
    std::vector<int64_t> inStride(inDims.size(), 1);
    for (std::size_t i = inDims.size(); i-- > 1;) {
        inStride[i - 1] = inStride[i] * inDims[i];
    }
    std::vector<int64_t> outStride(expectedDims.size(), 1);
    for (std::size_t i = expectedDims.size(); i-- > 1;) {
        outStride[i - 1] = outStride[i] * expectedDims[i];
    }

    // Map each surviving input axis to its output axis.
    std::vector<int64_t> outAxis(inDims.size(), -1);
    int64_t nextOutAxis = 0;
    for (int64_t i = 0; i < rank; ++i) {
        if (!reduced[static_cast<std::size_t>(i)]) {
            outAxis[static_cast<std::size_t>(i)] = keepdims != 0 ? i : nextOutAxis;
            ++nextOutAxis;
        }
    }

    const float* x = static_cast<const float*>(in.data);
    float* y = static_cast<float*>(out.data);
    if (y == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }
    std::fill(y, y + outCount, std::numeric_limits<float>::lowest());
    for (int64_t i = 0; i < inCount; ++i) {
        int64_t remaining = i;
        int64_t outOffset = 0;
        for (int64_t j = 0; j < rank; ++j) {
            const std::size_t uj = static_cast<std::size_t>(j);
            const int64_t coord = remaining / inStride[uj];
            remaining %= inStride[uj];
            if (!reduced[uj]) {
                outOffset += coord * outStride[static_cast<std::size_t>(outAxis[uj])];
            }
        }
        y[outOffset] = std::max(y[outOffset], x[i]);
    }
}

} // namespace kernels
} // namespace engine
