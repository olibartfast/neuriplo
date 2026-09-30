// Gemm: Y = alpha * op(A) * op(B) + beta * C, with optional transposes.
//
// Y is [M,N], op(A) is [M,K], and op(B) is [K,N]. The optional bias C is
// broadcast to the output from either [N] or [M,N]. Naive, float32,
// single-threaded.

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
                throw InferenceException("Gemm: attribute '" + name + "' must be an integer (node '" + node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

// Reads a float attribute, or returns the fallback when the attribute is absent.
float FloatAttribute(const Node& node, const std::string& name, float fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Gemm: attribute '" + name + "' must be a float (node '" + node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

} // namespace

void Gemm(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Gemm (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two or three inputs and one output");
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    const bool hasC = inputs.size() == 3;
    if (a.dtype != DataType::Float32 || b.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (hasC && inputs[2].dtype != DataType::Float32) {
        throw InferenceException(context + ": bias tensor must be float32");
    }
    if (a.dims.size() != 2 || b.dims.size() != 2) {
        throw InferenceException(context + ": A and B must be rank-2");
    }
    (void)ElementCount(a.dims, context);
    (void)ElementCount(b.dims, context);

    const int64_t transA = IntAttribute(node, "transA", 0);
    const int64_t transB = IntAttribute(node, "transB", 0);
    const float alpha = FloatAttribute(node, "alpha", 1.0f);
    const float beta = FloatAttribute(node, "beta", 1.0f);

    // Rows/columns of the operands after applying the transposes.
    const int64_t m = transA == 0 ? a.dims[0] : a.dims[1];
    const int64_t k = transA == 0 ? a.dims[1] : a.dims[0];
    const int64_t bK = transB == 0 ? b.dims[0] : b.dims[1];
    const int64_t n = transB == 0 ? b.dims[1] : b.dims[0];
    if (k != bK) {
        throw InferenceException(context + ": contraction dimension mismatch");
    }

    const std::vector<int64_t> expected = {m, n};
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the operands");
    }
    (void)ElementCount(out.dims, context);

    // Resolve the bias pointer and its rank; null when C is absent.
    const float* bias = nullptr;
    int64_t biasRank = 0;
    if (hasC) {
        const TensorView& c = inputs[2];
        const bool rankOne = c.dims.size() == 1 && c.dims[0] == n;
        const bool rankTwo = c.dims.size() == 2 && c.dims[0] == m && c.dims[1] == n;
        if (!rankOne && !rankTwo) {
            throw InferenceException(context + ": bias shape must be [N] or [M,N]");
        }
        (void)ElementCount(c.dims, context);
        bias = static_cast<const float*>(c.data);
        biasRank = static_cast<int64_t>(c.dims.size());
    }

    if (a.data == nullptr || b.data == nullptr || out.data == nullptr || (hasC && bias == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    const float* ap = static_cast<const float*>(a.data);
    const float* bp = static_cast<const float*>(b.data);
    float* y = static_cast<float*>(out.data);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                const float av = transA == 0 ? ap[i * k + kk] : ap[kk * m + i];
                const float bv = transB == 0 ? bp[kk * n + j] : bp[j * k + kk];
                acc += av * bv;
            }
            float value = alpha * acc;
            if (hasC) {
                value += beta * (biasRank == 1 ? bias[j] : bias[i * n + j]);
            }
            y[i * n + j] = value;
        }
    }
}

} // namespace kernels
} // namespace engine
