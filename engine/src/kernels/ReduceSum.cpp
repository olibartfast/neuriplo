// ReduceSum: sum over the requested axes; float32 (float accumulation) or int64 data.

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
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        count *= dim;
    }
    return count;
}

// Reads an int64 attribute, or returns the fallback when the attribute is absent.
int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("ReduceSum: attribute '" + name + "' must be an integer (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

template <typename T>
void SumTyped(const T* x, T* y, const std::vector<int64_t>& inDims, const std::vector<bool>& reduced, bool keepdims,
              int64_t inCount, int64_t outCount, const std::vector<int64_t>& outDims) {
    const int64_t rank = static_cast<int64_t>(inDims.size());
    std::vector<int64_t> inStride(inDims.size(), 1);
    for (std::size_t i = inDims.size(); i-- > 1;) {
        inStride[i - 1] = inStride[i] * inDims[i];
    }
    std::vector<int64_t> outStride(outDims.size(), 1);
    for (std::size_t i = outDims.size(); i-- > 1;) {
        outStride[i - 1] = outStride[i] * outDims[i];
    }
    std::vector<int64_t> outAxis(inDims.size(), -1);
    int64_t nextOutAxis = 0;
    for (int64_t i = 0; i < rank; ++i) {
        if (!reduced[static_cast<std::size_t>(i)]) {
            outAxis[static_cast<std::size_t>(i)] = keepdims ? i : nextOutAxis;
            ++nextOutAxis;
        }
    }
    for (int64_t i = 0; i < outCount; ++i) {
        y[i] = T{};
    }
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
        y[outOffset] += x[i];
    }
}

} // namespace

void ReduceSum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ReduceSum (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one or two inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if ((in.dtype != DataType::Float32 && in.dtype != DataType::Int64) || out.dtype != in.dtype) {
        throw InferenceException(context + ": data tensors must share dtype float32 or int64");
    }

    const std::vector<int64_t>& inDims = in.dims;
    const int64_t rank = static_cast<int64_t>(inDims.size());
    const int64_t inCount = ElementCount(inDims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const bool keepdims = IntAttribute(node, "keepdims", 1) != 0;
    const bool noopEmpty = IntAttribute(node, "noop_with_empty_axes", 0) != 0;

    // An absent or empty axes input means every axis unless noop_with_empty_axes asks for identity.
    std::vector<int64_t> axes;
    bool identity = false;
    if (inputs.size() == 2 && inputs[1].dtype != DataType::Unknown) {
        const TensorView& axesView = inputs[1];
        if (axesView.dtype != DataType::Int64) {
            throw InferenceException(context + ": axes tensor must be int64");
        }
        int64_t axesCount = 1;
        for (int64_t dim : axesView.dims) {
            axesCount *= dim;
        }
        if (axesCount > 0) {
            if (axesView.data == nullptr) {
                throw InferenceException(context + ": null axes buffer");
            }
            const int64_t* values = static_cast<const int64_t*>(axesView.data);
            axes.assign(values, values + axesCount);
        }
    }
    if (axes.empty()) {
        if (noopEmpty) {
            identity = true;
        } else {
            for (int64_t i = 0; i < rank; ++i) {
                axes.push_back(i);
            }
        }
    }

    const std::size_t elemSize = in.dtype == DataType::Float32 ? sizeof(float) : sizeof(int64_t);
    if (identity) {
        if (out.dims != inDims) {
            throw InferenceException(context + ": output shape does not match the input");
        }
        const char* src = static_cast<const char*>(in.data);
        char* dst = static_cast<char*>(out.data);
        for (std::size_t i = 0; i < static_cast<std::size_t>(inCount) * elemSize; ++i) {
            dst[i] = src[i];
        }
        return;
    }

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
            if (keepdims) {
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

    if (in.dtype == DataType::Float32) {
        SumTyped<float>(static_cast<const float*>(in.data), static_cast<float*>(out.data), inDims, reduced, keepdims,
                        inCount, outCount, expectedDims);
    } else {
        SumTyped<int64_t>(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), inDims, reduced,
                          keepdims, inCount, outCount, expectedDims);
    }
}

} // namespace kernels
} // namespace engine
