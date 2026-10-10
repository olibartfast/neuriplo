// Resize: nearest or linear resampling with the ONNX coordinate transforms.
//
// The output dims come from a sizes input, or floor(in * scale) from a scales
// input (never both). Each output coordinate maps to the input through
// coordinate_transformation_mode (half_pixel, pytorch_half_pixel,
// align_corners, asymmetric; default half_pixel). `nearest` then rounds per
// nearest_mode (round_prefer_floor, round_prefer_ceil, floor, ceil; default
// round_prefer_floor); `linear` interpolates between the two neighbours on
// every axis whose sample falls between them, clamping at the border.
// Antialiasing and cubic mode are rejected at load. Data is float32, or
// int64 for nearest.

#include "Kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
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

std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Resize: attribute '" + name + "' must be a string (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

// The input coordinate an output coordinate samples, per the ONNX transform.
double SourceCoordinate(const std::string& mode, int64_t out_coord, int64_t in_len, int64_t out_len, double scale,
                        const std::string& context) {
    const double x = static_cast<double>(out_coord);
    if (mode == "half_pixel") {
        return (x + 0.5) / scale - 0.5;
    }
    if (mode == "pytorch_half_pixel") {
        return out_len > 1 ? (x + 0.5) / scale - 0.5 : 0.0;
    }
    if (mode == "align_corners") {
        return out_len == 1 ? 0.0 : x * static_cast<double>(in_len - 1) / static_cast<double>(out_len - 1);
    }
    if (mode == "asymmetric") {
        return x / scale;
    }
    throw InferenceException(context + ": unsupported coordinate_transformation_mode '" + mode + "'");
}

int64_t NearestIndex(const std::string& mode, double x, int64_t in_len, const std::string& context) {
    int64_t index = 0;
    if (mode == "round_prefer_floor") {
        const double f = std::floor(x);
        index = static_cast<int64_t>(x - f == 0.5 ? f : std::round(x));
    } else if (mode == "round_prefer_ceil") {
        index = static_cast<int64_t>(std::round(x));
    } else if (mode == "floor") {
        index = static_cast<int64_t>(std::floor(x));
    } else if (mode == "ceil") {
        index = static_cast<int64_t>(std::ceil(x));
    } else {
        throw InferenceException(context + ": unsupported nearest_mode '" + mode + "'");
    }
    return std::min(std::max(index, int64_t{0}), in_len - 1);
}

// One output coordinate on one axis: the two input neighbours and the weight
// of the upper one (zero for nearest, or when the sample lands on a pixel).
struct AxisSample {
    int64_t lower = 0;
    int64_t upper = 0;
    double weight = 0.0;
};

std::vector<AxisSample> AxisTable(bool linear, const std::string& transform, const std::string& nearest, int64_t in_len,
                                  int64_t out_len, double scale, const std::string& context) {
    std::vector<AxisSample> table(static_cast<std::size_t>(out_len));
    for (int64_t o = 0; o < out_len; ++o) {
        const double x = SourceCoordinate(transform, o, in_len, out_len, scale, context);
        AxisSample& sample = table[static_cast<std::size_t>(o)];
        if (!linear) {
            sample.lower = sample.upper = NearestIndex(nearest, x, in_len, context);
            continue;
        }
        const double clamped = std::min(std::max(x, 0.0), static_cast<double>(in_len - 1));
        sample.lower = static_cast<int64_t>(std::floor(clamped));
        sample.upper = std::min(sample.lower + 1, in_len - 1);
        sample.weight = sample.upper == sample.lower ? 0.0 : clamped - static_cast<double>(sample.lower);
    }
    return table;
}

template <typename T>
void ResizeTyped(const T* x, T* y, const std::vector<int64_t>& out_dims, const std::vector<int64_t>& in_strides,
                 const std::vector<int64_t>& out_strides, const std::vector<std::vector<AxisSample>>& tables,
                 int64_t out_count) {
    const std::size_t rank = out_dims.size();
    std::vector<std::pair<int64_t, double>> taps;
    std::vector<std::pair<int64_t, double>> next;
    for (int64_t i = 0; i < out_count; ++i) {
        int64_t remaining = i;
        taps.assign(1, {0, 1.0});
        for (std::size_t j = 0; j < rank; ++j) {
            const int64_t coord = remaining / out_strides[j];
            remaining %= out_strides[j];
            const AxisSample& s = tables[j][static_cast<std::size_t>(coord)];
            if (s.weight == 0.0) {
                for (auto& tap : taps) {
                    tap.first += s.lower * in_strides[j];
                }
                continue;
            }
            next.clear();
            for (const auto& tap : taps) {
                next.push_back({tap.first + s.lower * in_strides[j], tap.second * (1.0 - s.weight)});
                next.push_back({tap.first + s.upper * in_strides[j], tap.second * s.weight});
            }
            taps.swap(next);
        }
        if (taps.size() == 1) {
            y[i] = x[taps[0].first];
            continue;
        }
        double value = 0.0;
        for (const auto& tap : taps) {
            value += tap.second * static_cast<double>(x[tap.first]);
        }
        y[i] = static_cast<T>(value);
    }
}

} // namespace

void Resize(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Resize (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 4 || outputs.size() != 1) {
        throw InferenceException(context + ": expected an input, roi, scales, and an optional sizes input");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    const std::string mode = StringAttribute(node, "mode", "nearest");
    if (mode != "nearest" && mode != "linear") {
        throw InferenceException(context + ": Resize 'mode' must be 'nearest' or 'linear' here");
    }
    const bool linear = mode == "linear";
    if (in.dtype != DataType::Float32 && !(in.dtype == DataType::Int64 && !linear)) {
        throw InferenceException(context + ": tensors must be float32 (or int64 for nearest)");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const std::string transform = StringAttribute(node, "coordinate_transformation_mode", "half_pixel");
    const std::string nearest = StringAttribute(node, "nearest_mode", "round_prefer_floor");

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    std::vector<double> scale_vec(static_cast<std::size_t>(rank), 1.0);
    const bool has_scales =
        inputs.size() > 2 && inputs[2].dtype != DataType::Unknown && ElementCount(inputs[2].dims, context) > 0;
    const bool has_sizes = inputs.size() > 3 && inputs[3].dtype != DataType::Unknown;
    if (has_sizes && has_scales) {
        throw InferenceException(context + ": Resize takes exactly one of scales and sizes");
    }
    if (has_sizes) {
        // A sizes input fixes the output dims; the per-axis scale used by the
        // coordinate transform is then out / in, as ONNX defines it.
        const TensorView& sizes_view = inputs[3];
        if (sizes_view.dtype != DataType::Int64 || ElementCount(sizes_view.dims, context) != rank ||
            sizes_view.data == nullptr) {
            throw InferenceException(context + ": Resize sizes must be int64 with one entry per input axis");
        }
        const int64_t* sizes = static_cast<const int64_t*>(sizes_view.data);
        for (int64_t i = 0; i < rank; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            if (out.dims.size() != static_cast<std::size_t>(rank) || out.dims[ui] != sizes[ui]) {
                throw InferenceException(context + ": output shape does not match the sizes input");
            }
            scale_vec[ui] = static_cast<double>(static_cast<float>(sizes[ui]) / static_cast<float>(in.dims[ui]));
        }
    } else {
        if (!has_scales || inputs[2].dtype != DataType::Float32) {
            throw InferenceException(context + ": Resize requires a scales or sizes input");
        }
        const TensorView& scales_view = inputs[2];
        if (ElementCount(scales_view.dims, context) != rank) {
            throw InferenceException(context + ": Resize scales rank must match the input rank");
        }
        if (scales_view.data == nullptr) {
            throw InferenceException(context + ": null scales buffer");
        }
        const float* scales = static_cast<const float*>(scales_view.data);
        std::vector<int64_t> expected(static_cast<std::size_t>(rank), 0);
        for (int64_t i = 0; i < rank; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const int64_t dim =
                static_cast<int64_t>(std::floor(static_cast<double>(in.dims[ui]) * static_cast<double>(scales[ui])));
            if (dim <= 0) {
                throw InferenceException(context + ": Resize produces a non-positive output dim");
            }
            expected[ui] = dim;
            scale_vec[ui] = static_cast<double>(scales[ui]);
        }
        if (out.dims != expected) {
            throw InferenceException(context + ": output shape does not match the scaled input");
        }
    }

    const int64_t out_count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    std::vector<std::vector<AxisSample>> tables;
    for (int64_t i = 0; i < rank; ++i) {
        const std::size_t ui = static_cast<std::size_t>(i);
        tables.push_back(AxisTable(linear, transform, nearest, in.dims[ui], out.dims[ui], scale_vec[ui], context));
    }
    const std::vector<int64_t> in_strides = Strides(in.dims);
    const std::vector<int64_t> out_strides = Strides(out.dims);
    if (in.dtype == DataType::Float32) {
        ResizeTyped<float>(static_cast<const float*>(in.data), static_cast<float*>(out.data), out.dims, in_strides,
                           out_strides, tables, out_count);
    } else {
        ResizeTyped<int64_t>(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), out.dims,
                             in_strides, out_strides, tables, out_count);
    }
}

} // namespace kernels
} // namespace engine
