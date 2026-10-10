// Tile: repeat the input along every axis by the int64 repeats input.
//
// Output dim i is input dim i times repeats[i]; each output coordinate reads
// the input at coordinate modulo the input dim. Data is float32, int64, or
// bool.

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

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

template <typename T>
void TileTyped(const T* x, T* y, const std::vector<int64_t>& inDims, const std::vector<int64_t>& outDims,
               int64_t outCount) {
    const std::vector<int64_t> inStrides = Strides(inDims);
    const std::vector<int64_t> outStrides = Strides(outDims);
    for (int64_t i = 0; i < outCount; ++i) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (std::size_t j = 0; j < outDims.size(); ++j) {
            const int64_t coord = remaining / outStrides[j];
            remaining %= outStrides[j];
            offset += (coord % inDims[j]) * inStrides[j];
        }
        y[i] = x[offset];
    }
}

} // namespace

void Tile(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Tile (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input, a repeats input, and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& repeats = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != out.dtype ||
        (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool)) {
        throw InferenceException(context + ": data must share one float32, int64, or bool dtype");
    }
    if (repeats.dtype != DataType::Int64 || repeats.dims.size() != 1 ||
        repeats.dims[0] != static_cast<int64_t>(in.dims.size())) {
        throw InferenceException(context + ": repeats must be a rank-1 int64 tensor with one entry per input axis");
    }
    (void)ElementCount(in.dims, context);
    const int64_t outCount = ElementCount(out.dims, context);
    if (in.data == nullptr || repeats.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const int64_t* reps = static_cast<const int64_t*>(repeats.data);
    if (out.dims.size() != in.dims.size()) {
        throw InferenceException(context + ": output rank does not match the input rank");
    }
    for (std::size_t j = 0; j < in.dims.size(); ++j) {
        if (reps[j] <= 0 || out.dims[j] != in.dims[j] * reps[j]) {
            throw InferenceException(context + ": output shape does not match input dims times repeats");
        }
    }

    if (in.dtype == DataType::Float32) {
        TileTyped(static_cast<const float*>(in.data), static_cast<float*>(out.data), in.dims, out.dims, outCount);
    } else if (in.dtype == DataType::Int64) {
        TileTyped(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), in.dims, out.dims, outCount);
    } else {
        TileTyped(static_cast<const bool*>(in.data), static_cast<bool*>(out.data), in.dims, out.dims, outCount);
    }
}

} // namespace kernels
} // namespace engine
