// GridSample: sample a 4-D float32 input at normalized grid positions.
//
// X is [N,C,H,W] and grid is [N,Ho,Wo,2] with (x, y) in [-1, 1] (x indexes W,
// y indexes H); the output is [N,C,Ho,Wo]. Attributes: mode (bilinear or
// nearest, default bilinear), padding_mode (zeros, border, reflection, default
// zeros), align_corners (0 or 1, default 0). Nearest rounds half to even.

#include "Kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace engine {
namespace kernels {
namespace {

std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("GridSample: attribute '" + name + "' must be a string (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("GridSample: attribute '" + name + "' must be an int (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

// Reflect x about [lo, hi] (the border coordinates), then clamp.
double Reflect(double x, double lo, double hi) {
    if (hi == lo) {
        return lo;
    }
    const double range = hi - lo;
    double d = std::fabs(x - lo);
    const double flips = std::floor(d / range);
    const double extra = d - flips * range;
    return std::fmod(flips, 2.0) == 0.0 ? lo + extra : hi - extra;
}

// Normalized coordinate -> source pixel coordinate on an axis of `len` pixels,
// with the padding mode applied to the continuous coordinate.
double SourceCoordinate(double g, int64_t len, bool alignCorners, const std::string& padding) {
    const double size = static_cast<double>(len);
    double x = alignCorners ? (g + 1.0) / 2.0 * (size - 1.0) : ((g + 1.0) * size - 1.0) / 2.0;
    if (padding == "reflection") {
        x = alignCorners ? Reflect(x, 0.0, size - 1.0) : Reflect(x, -0.5, size - 0.5);
        x = std::min(std::max(x, 0.0), size - 1.0);
    } else if (padding == "border") {
        x = std::min(std::max(x, 0.0), size - 1.0);
    }
    return x;
}

} // namespace

void GridSample(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "GridSample (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    const std::string mode = StringAttribute(node, "mode", "bilinear");
    const std::string padding = StringAttribute(node, "padding_mode", "zeros");
    const int64_t alignAttr = IntAttribute(node, "align_corners", 0);
    if (mode != "bilinear" && mode != "nearest") {
        throw InferenceException(context + ": unsupported mode '" + mode + "'");
    }
    if (padding != "zeros" && padding != "border" && padding != "reflection") {
        throw InferenceException(context + ": unsupported padding_mode '" + padding + "'");
    }
    if (alignAttr != 0 && alignAttr != 1) {
        throw InferenceException(context + ": align_corners must be 0 or 1");
    }
    const bool alignCorners = alignAttr == 1;

    const TensorView& x = inputs[0];
    const TensorView& grid = inputs[1];
    const TensorView& out = outputs[0];
    if (x.dtype != DataType::Float32 || grid.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (x.dims.size() != 4 || grid.dims.size() != 4 || grid.dims[3] != 2) {
        throw InferenceException(context + ": expected X [N,C,H,W] and grid [N,Ho,Wo,2]");
    }
    const int64_t n = x.dims[0];
    const int64_t c = x.dims[1];
    const int64_t h = x.dims[2];
    const int64_t w = x.dims[3];
    const int64_t ho = grid.dims[1];
    const int64_t wo = grid.dims[2];
    if (grid.dims[0] != n) {
        throw InferenceException(context + ": batch sizes of X and grid differ");
    }
    if (out.dims != std::vector<int64_t>({n, c, ho, wo})) {
        throw InferenceException(context + ": output shape does not match the operands");
    }
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || ho <= 0 || wo <= 0) {
        throw InferenceException(context + ": non-positive dimension");
    }
    if (x.data == nullptr || grid.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const float* xp = static_cast<const float*>(x.data);
    const float* gp = static_cast<const float*>(grid.data);
    float* yp = static_cast<float*>(out.data);
    const bool nearest = mode == "nearest";

    for (int64_t b = 0; b < n; ++b) {
        for (int64_t i = 0; i < ho; ++i) {
            for (int64_t j = 0; j < wo; ++j) {
                const float* g = gp + ((b * ho + i) * wo + j) * 2;
                const double px = SourceCoordinate(static_cast<double>(g[0]), w, alignCorners, padding);
                const double py = SourceCoordinate(static_cast<double>(g[1]), h, alignCorners, padding);
                for (int64_t ch = 0; ch < c; ++ch) {
                    const float* plane = xp + (b * c + ch) * h * w;
                    auto at = [&](int64_t yy, int64_t xx) -> double {
                        if (yy < 0 || yy >= h || xx < 0 || xx >= w) {
                            return 0.0;
                        }
                        return static_cast<double>(plane[yy * w + xx]);
                    };
                    double value = 0.0;
                    if (nearest) {
                        value = at(static_cast<int64_t>(std::nearbyint(py)), static_cast<int64_t>(std::nearbyint(px)));
                    } else {
                        const double fx = std::floor(px);
                        const double fy = std::floor(py);
                        const int64_t x0 = static_cast<int64_t>(fx);
                        const int64_t y0 = static_cast<int64_t>(fy);
                        const double wx = px - fx;
                        const double wy = py - fy;
                        value = at(y0, x0) * (1.0 - wx) * (1.0 - wy) + at(y0, x0 + 1) * wx * (1.0 - wy) +
                                at(y0 + 1, x0) * (1.0 - wx) * wy + at(y0 + 1, x0 + 1) * wx * wy;
                    }
                    yp[((b * c + ch) * ho + i) * wo + j] = static_cast<float>(value);
                }
            }
        }
    }
}

} // namespace kernels
} // namespace engine
