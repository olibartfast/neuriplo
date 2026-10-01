// Mod: truncating integer remainder with NumPy multidirectional broadcasting.
//
// Only fmod=0 is supported: the C++ truncating remainder, whose sign follows
// the dividend. A zero divisor is rejected with node context (there is no
// integer inf/nan to fall back on).

#include "Kernels.hpp"

#include <algorithm>
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

// The NumPy multidirectional broadcast of the two input shapes. Throws when the
// shapes cannot be broadcast together.
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

// Row-major strides for a contiguous tensor.
std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

// For each output axis, the input stride that maps it (0 where the input axis
// is broadcast). Throws when the input shape cannot broadcast to the output.
std::vector<int64_t> AlignedStrides(const std::vector<int64_t>& inDims, const std::vector<int64_t>& outDims,
                                    const std::string& context) {
    const std::size_t outRank = outDims.size();
    const std::size_t inRank = inDims.size();
    const std::vector<int64_t> inStrides = Strides(inDims);
    std::vector<int64_t> aligned(outRank, 0);
    for (std::size_t j = 0; j < outRank; ++j) {
        if (j + inRank < outRank) {
            continue; // a missing leading axis broadcasts as a size-1 axis
        }
        const std::size_t k = j - (outRank - inRank);
        if (inDims[k] == outDims[j]) {
            aligned[j] = inStrides[k];
        } else if (inDims[k] == 1) {
            aligned[j] = 0; // broadcast
        } else {
            throw InferenceException(context + ": shapes are not broadcast-compatible");
        }
    }
    return aligned;
}

} // namespace

void Mod(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Mod (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "fmod") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'fmod' must be an integer");
            }
            if (*value != 0) {
                throw InferenceException(context + ": only fmod=0 (integer remainder) is supported");
            }
        }
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    if (a.dtype != DataType::Int64 || b.dtype != DataType::Int64 || out.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be int64");
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
    const std::vector<int64_t> aStrides = AlignedStrides(a.dims, out.dims, context);
    const std::vector<int64_t> bStrides = AlignedStrides(b.dims, out.dims, context);

    const int64_t* x = static_cast<const int64_t*>(a.data);
    const int64_t* z = static_cast<const int64_t*>(b.data);
    int64_t* y = static_cast<int64_t*>(out.data);
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
        if (z[bOffset] == 0) {
            throw InferenceException(context + ": division by zero");
        }
        y[i] = x[aOffset] % z[bOffset];
    }
}

} // namespace kernels
} // namespace engine
