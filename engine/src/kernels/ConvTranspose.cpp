// ConvTranspose: the ONNX NCHW transposed convolution (gradient of Conv).
//
// Input is [N,C,H,W] and the weight is [C, M/group, kH, kW]. Output spatial size
// is stride*(in-1) + output_padding + dilation*(k-1) + 1 - pad_begin - pad_end.
// Padding is explicit (auto_pad NOTSET or VALID); output_shape and SAME_* are
// rejected at load. An optional rank-1 bias of length M is added. Naive,
// float32, single-threaded scatter.

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
                throw InferenceException("ConvTranspose: attribute '" + name + "' must be an integer (node '" +
                                         node.name + "')");
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
                throw InferenceException("ConvTranspose: attribute '" + name + "' must be an int list (node '" +
                                         node.name + "')");
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
                throw InferenceException("ConvTranspose: attribute '" + name + "' must be a string (node '" +
                                         node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

} // namespace

void ConvTranspose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ConvTranspose (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected input, weight, optional bias, and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    const TensorView* bias = inputs.size() == 3 && inputs[2].data != nullptr ? &inputs[2] : nullptr;
    const TensorView& out = outputs[0];
    if (x.dtype != DataType::Float32 || w.dtype != DataType::Float32 || out.dtype != DataType::Float32 ||
        (bias != nullptr && bias->dtype != DataType::Float32)) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (x.dims.size() != 4 || w.dims.size() != 4) {
        throw InferenceException(context + ": input and weight must be rank 4 (NCHW)");
    }
    (void)ElementCount(x.dims, context);
    (void)ElementCount(w.dims, context);
    (void)ElementCount(out.dims, context);

    const int64_t group = IntAttribute(node, "group", 1);
    const int64_t nBatch = x.dims[0];
    const int64_t inChannels = x.dims[1];
    const int64_t inHeight = x.dims[2];
    const int64_t inWidth = x.dims[3];
    if (group <= 0 || inChannels % group != 0 || w.dims[0] != inChannels) {
        throw InferenceException(context + ": weight channels do not match the input and group");
    }
    const int64_t outPerGroup = w.dims[1];
    const int64_t outChannels = outPerGroup * group;
    const int64_t inPerGroup = inChannels / group;
    const int64_t kH = w.dims[2];
    const int64_t kW = w.dims[3];

    std::vector<int64_t> strides = IntListAttribute(node, "strides");
    std::vector<int64_t> dilations = IntListAttribute(node, "dilations");
    std::vector<int64_t> outputPadding = IntListAttribute(node, "output_padding");
    std::vector<int64_t> pads = IntListAttribute(node, "pads");
    if (strides.empty()) {
        strides = {1, 1};
    }
    if (dilations.empty()) {
        dilations = {1, 1};
    }
    if (outputPadding.empty()) {
        outputPadding = {0, 0};
    }
    if (pads.empty() || StringAttribute(node, "auto_pad", "NOTSET") == "VALID") {
        pads = {0, 0, 0, 0};
    }
    if (strides.size() != 2 || dilations.size() != 2 || outputPadding.size() != 2 || pads.size() != 4) {
        throw InferenceException(context + ": strides, dilations, output_padding, and pads must be 2-D");
    }

    const int64_t outHeight =
        strides[0] * (inHeight - 1) + outputPadding[0] + dilations[0] * (kH - 1) + 1 - pads[0] - pads[2];
    const int64_t outWidth =
        strides[1] * (inWidth - 1) + outputPadding[1] + dilations[1] * (kW - 1) + 1 - pads[1] - pads[3];
    const std::vector<int64_t> expected = {nBatch, outChannels, outHeight, outWidth};
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the transposed window");
    }
    if (bias != nullptr && (bias->dims.size() != 1 || bias->dims[0] != outChannels)) {
        throw InferenceException(context + ": bias must be rank-1 of length M");
    }
    if (x.data == nullptr || w.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const float* xData = static_cast<const float*>(x.data);
    const float* wData = static_cast<const float*>(w.data);
    const float* bData = bias != nullptr ? static_cast<const float*>(bias->data) : nullptr;
    float* yData = static_cast<float*>(out.data);
    const int64_t outPlane = outHeight * outWidth;

    for (int64_t n = 0; n < nBatch; ++n) {
        for (int64_t m = 0; m < outChannels; ++m) {
            float* yPlane = yData + (n * outChannels + m) * outPlane;
            const float init = bData != nullptr ? bData[m] : 0.0F;
            for (int64_t i = 0; i < outPlane; ++i) {
                yPlane[i] = init;
            }
        }
        for (int64_t g = 0; g < group; ++g) {
            for (int64_t ic = 0; ic < inPerGroup; ++ic) {
                const int64_t c = g * inPerGroup + ic;
                const float* xPlane = xData + (n * inChannels + c) * inHeight * inWidth;
                for (int64_t oc = 0; oc < outPerGroup; ++oc) {
                    const int64_t m = g * outPerGroup + oc;
                    float* yPlane = yData + (n * outChannels + m) * outPlane;
                    const float* wKernel = wData + (c * outPerGroup + oc) * kH * kW;
                    for (int64_t ih = 0; ih < inHeight; ++ih) {
                        for (int64_t iw = 0; iw < inWidth; ++iw) {
                            const float value = xPlane[ih * inWidth + iw];
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                const int64_t oh = ih * strides[0] - pads[0] + kh * dilations[0];
                                if (oh < 0 || oh >= outHeight) {
                                    continue;
                                }
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    const int64_t ow = iw * strides[1] - pads[1] + kw * dilations[1];
                                    if (ow < 0 || ow >= outWidth) {
                                        continue;
                                    }
                                    yPlane[oh * outWidth + ow] += value * wKernel[kh * kW + kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace kernels
} // namespace engine
