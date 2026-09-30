// MaxPool: the ONNX NCHW max pooling window.
//
// Input is [N,C,H,W]. Kernel shape is required; strides and dilations default to
// 1, padding is explicit or automatic, and ceil_mode selects the rounding of the
// output size. Positions that fall outside the padded input sample -inf, so the
// max only sees the in-bounds window. Only the primary output is written.
// Naive, float32, single-threaded.

#include "Kernels.hpp"

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
int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("MaxPool: attribute '" + name + "' must be an integer (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

// Reads an int list attribute, or an empty list when the attribute is absent.
std::vector<int64_t> IntListAttribute(const Node& node, const std::string& name) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("MaxPool: attribute '" + name + "' must be an int list (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return {};
}

// Reads a string attribute, or returns the fallback when the attribute is absent.
std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("MaxPool: attribute '" + name + "' must be a string (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

// Resolves per-spatial-axis begin/end padding from auto_pad or the explicit pads
// list. Mirrors the static shape inference, so the two never disagree.
void ResolvePads(const std::string& context, const std::string& auto_pad, const std::vector<int64_t>& in_spatial,
                 const std::vector<int64_t>& kernel, const std::vector<int64_t>& dilations,
                 const std::vector<int64_t>& strides, const std::vector<int64_t>& pads_attr,
                 std::vector<int64_t>& pad_begin, std::vector<int64_t>& pad_end) {
    const std::size_t rank = in_spatial.size();
    pad_begin.assign(rank, 0);
    pad_end.assign(rank, 0);

    if (auto_pad == "VALID") {
        return;
    }
    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        for (std::size_t i = 0; i < rank; ++i) {
            const int64_t out = (in_spatial[i] + strides[i] - 1) / strides[i];
            const int64_t effective = dilations[i] * (kernel[i] - 1) + 1;
            int64_t total = (out - 1) * strides[i] + effective - in_spatial[i];
            if (total < 0) {
                total = 0;
            }
            if (auto_pad == "SAME_UPPER") {
                pad_begin[i] = total / 2;
                pad_end[i] = total - pad_begin[i];
            } else {
                pad_end[i] = total / 2;
                pad_begin[i] = total - pad_end[i];
            }
        }
        return;
    }

    // NOTSET (or an empty value): use the explicit pads, defaulting to zero.
    if (pads_attr.empty()) {
        return;
    }
    if (pads_attr.size() != 2 * rank) {
        throw InferenceException(context + ": pads must have two entries per spatial dimension");
    }
    for (std::size_t i = 0; i < rank; ++i) {
        pad_begin[i] = pads_attr[i];
        pad_end[i] = pads_attr[rank + i];
    }
}

// One spatial axis of the output window, matching the shape-inference formula.
int64_t WindowOutput(const std::string& context, int64_t in, int64_t kernel, int64_t dilation, int64_t pad_begin,
                     int64_t pad_end, int64_t stride, bool ceil_mode) {
    const int64_t effective = dilation * (kernel - 1) + 1;
    const int64_t numer = in + pad_begin + pad_end - effective;
    if (numer < 0) {
        throw InferenceException(context + ": window is larger than the padded input");
    }
    const int64_t out = (ceil_mode ? (numer + stride - 1) / stride : numer / stride) + 1;
    if (out <= 0) {
        throw InferenceException(context + ": computes a non-positive output dimension");
    }
    return out;
}

} // namespace

void MaxPool(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "MaxPool (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.empty()) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& out = outputs[0];
    if (x.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (x.dims.size() != 4) {
        throw InferenceException(context + ": input must be rank 4 (NCHW)");
    }
    (void)ElementCount(x.dims, context);
    (void)ElementCount(out.dims, context);

    const int64_t n_batch = x.dims[0];
    const int64_t channels = x.dims[1];
    const int64_t inHeight = x.dims[2];
    const int64_t inWidth = x.dims[3];

    const std::vector<int64_t> kernel = IntListAttribute(node, "kernel_shape");
    if (kernel.empty()) {
        throw InferenceException(context + ": kernel_shape is required");
    }
    if (kernel.size() != 2) {
        throw InferenceException(context + ": kernel_shape must have two spatial entries");
    }
    std::vector<int64_t> strides = IntListAttribute(node, "strides");
    std::vector<int64_t> dilations = IntListAttribute(node, "dilations");
    if (strides.empty()) {
        strides = {1, 1};
    }
    if (dilations.empty()) {
        dilations = {1, 1};
    }
    if (strides.size() != 2 || dilations.size() != 2) {
        throw InferenceException(context + ": stride and dilation must each have two spatial entries");
    }
    for (std::size_t i = 0; i < 2; ++i) {
        if (kernel[i] <= 0) {
            throw InferenceException(context + ": kernel dimensions must be positive");
        }
        if (strides[i] <= 0 || dilations[i] <= 0) {
            throw InferenceException(context + ": stride and dilation must be positive");
        }
    }
    const bool ceilMode = IntAttribute(node, "ceil_mode", 0) != 0;

    const std::vector<int64_t> in_spatial = {inHeight, inWidth};
    std::vector<int64_t> pad_begin;
    std::vector<int64_t> pad_end;
    ResolvePads(context, StringAttribute(node, "auto_pad", "NOTSET"), in_spatial, kernel, dilations, strides,
                IntListAttribute(node, "pads"), pad_begin, pad_end);

    const int64_t outHeight =
        WindowOutput(context, inHeight, kernel[0], dilations[0], pad_begin[0], pad_end[0], strides[0], ceilMode);
    const int64_t outWidth =
        WindowOutput(context, inWidth, kernel[1], dilations[1], pad_begin[1], pad_end[1], strides[1], ceilMode);
    const std::vector<int64_t> expected = {n_batch, channels, outHeight, outWidth};
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the pooling window");
    }

    if (x.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const float* xData = static_cast<const float*>(x.data);
    float* yData = static_cast<float*>(out.data);
    const float negInf = -std::numeric_limits<float>::infinity();

    for (int64_t n = 0; n < n_batch; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const float* xChannel = xData + ((n * channels + c) * inHeight) * inWidth;
            for (int64_t oh = 0; oh < outHeight; ++oh) {
                const int64_t inRowBase = oh * strides[0] - pad_begin[0];
                for (int64_t ow = 0; ow < outWidth; ++ow) {
                    const int64_t inColBase = ow * strides[1] - pad_begin[1];
                    float best = negInf;
                    for (int64_t kh = 0; kh < kernel[0]; ++kh) {
                        const int64_t ih = inRowBase + kh * dilations[0];
                        if (ih < 0 || ih >= inHeight) {
                            continue;
                        }
                        const float* xRow = xChannel + ih * inWidth;
                        for (int64_t kw = 0; kw < kernel[1]; ++kw) {
                            const int64_t iw = inColBase + kw * dilations[1];
                            if (iw < 0 || iw >= inWidth) {
                                continue;
                            }
                            if (xRow[iw] > best) {
                                best = xRow[iw];
                            }
                        }
                    }
                    yData[((n * channels + c) * outHeight + oh) * outWidth + ow] = best;
                }
            }
        }
    }
}

} // namespace kernels
} // namespace engine
