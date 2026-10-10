// Pow: elementwise base^exponent with NumPy multidirectional broadcasting.
//
// Base and exponent are float32; the output matches the base. An exponent of
// exactly two squares the base, which is what every surveyed export uses.

#include "Kernels.hpp"

#include <algorithm>
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
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        count *= dim;
    }
    return count;
}

std::vector<int64_t> BroadcastDims(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                                   const std::string& context) {
    const std::size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
        const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
        if (da != db && da != 1 && db != 1) {
            throw InferenceException(context + ": shapes are not broadcast-compatible");
        }
        out[i] = std::max(da, db);
    }
    return out;
}

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

std::vector<int64_t> AlignedStrides(const std::vector<int64_t>& inDims, const std::vector<int64_t>& outDims) {
    const std::size_t outRank = outDims.size();
    const std::size_t inRank = inDims.size();
    const std::vector<int64_t> inStrides = Strides(inDims);
    std::vector<int64_t> aligned(outRank, 0);
    for (std::size_t j = 0; j < outRank; ++j) {
        if (j + inRank < outRank) {
            continue;
        }
        const std::size_t k = j - (outRank - inRank);
        aligned[j] = inDims[k] == outDims[j] ? inStrides[k] : 0;
    }
    return aligned;
}

float PowOne(float base, float exponent) {
    if (exponent == 2.0F) {
        return base * base;
    }
    return std::pow(base, exponent);
}

} // namespace

void Pow(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Pow (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    if (a.dtype != DataType::Float32 || b.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    const int64_t outCount = ElementCount(out.dims, context);
    (void)ElementCount(a.dims, context);
    (void)ElementCount(b.dims, context);
    if (BroadcastDims(a.dims, b.dims, context) != out.dims) {
        throw InferenceException(context + ": output shape does not match the broadcast of the inputs");
    }
    if (a.data == nullptr || b.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const std::vector<int64_t> outStrides = Strides(out.dims);
    const std::vector<int64_t> aStrides = AlignedStrides(a.dims, out.dims);
    const std::vector<int64_t> bStrides = AlignedStrides(b.dims, out.dims);
    const float* x = static_cast<const float*>(a.data);
    const float* e = static_cast<const float*>(b.data);
    float* y = static_cast<float*>(out.data);
    for (int64_t i = 0; i < outCount; ++i) {
        int64_t remaining = i;
        int64_t aOffset = 0;
        int64_t bOffset = 0;
        for (std::size_t j = 0; j < out.dims.size(); ++j) {
            const int64_t coord = remaining / outStrides[j];
            remaining %= outStrides[j];
            aOffset += coord * aStrides[j];
            bOffset += coord * bStrides[j];
        }
        y[i] = PowOne(x[aOffset], e[bOffset]);
    }
}

} // namespace kernels
} // namespace engine
