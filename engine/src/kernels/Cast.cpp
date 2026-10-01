// Cast: element conversion between float32, int64, and bool per the `to` attr.
//
// ONNX element-type codes: 1 = FLOAT, 7 = INT64, 9 = BOOL. Float-to-integer
// truncates toward zero; any nonzero value converts to boolean true.

#include "Kernels.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace engine {
namespace kernels {
namespace {

// Product of the dimensions; every dimension must be positive. An empty shape
// is a scalar and yields one element.
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

constexpr int64_t kElemFloat = 1;
constexpr int64_t kElemInt64 = 7;
constexpr int64_t kElemBool = 9;

// Reads the required integer `to` attribute.
int64_t ToAttribute(const Node& node, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "to") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'to' must be an integer");
            }
            return *value;
        }
    }
    throw InferenceException(context + ": missing required attribute 'to'");
}

} // namespace

void Cast(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Cast (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": input must be float32, int64, or bool");
    }

    const int64_t to = ToAttribute(node, context);
    DataType want = DataType::Unknown;
    if (to == kElemFloat) {
        want = DataType::Float32;
    } else if (to == kElemInt64) {
        want = DataType::Int64;
    } else if (to == kElemBool) {
        want = DataType::Bool;
    } else {
        throw InferenceException(context + ": attribute 'to' names an unsupported element type");
    }
    if (out.dtype != want) {
        throw InferenceException(context + ": output dtype does not match attribute 'to'");
    }

    const int64_t outCount = ElementCount(out.dims, context);
    const int64_t inCount = ElementCount(in.dims, context);
    if (inCount != outCount || in.dims != out.dims) {
        throw InferenceException(context + ": output shape must match the input shape");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    if (want == DataType::Float32) {
        float* y = static_cast<float*>(out.data);
        if (in.dtype == DataType::Float32) {
            const float* x = static_cast<const float*>(in.data);
            for (int64_t i = 0; i < outCount; ++i) {
                y[i] = x[i];
            }
        } else if (in.dtype == DataType::Int64) {
            const int64_t* x = static_cast<const int64_t*>(in.data);
            for (int64_t i = 0; i < outCount; ++i) {
                y[i] = static_cast<float>(x[i]);
            }
        } else {
            const bool* x = static_cast<const bool*>(in.data);
            for (int64_t i = 0; i < outCount; ++i) {
                y[i] = x[i] ? 1.0F : 0.0F;
            }
        }
    } else if (want == DataType::Int64) {
        int64_t* y = static_cast<int64_t*>(out.data);
        if (in.dtype == DataType::Float32) {
            const float* x = static_cast<const float*>(in.data);
            for (int64_t i = 0; i < outCount; ++i) {
                y[i] = static_cast<int64_t>(x[i]);
            }
        } else if (in.dtype == DataType::Int64) {
            const int64_t* x = static_cast<const int64_t*>(in.data);
            for (int64_t i = 0; i < outCount; ++i) {
                y[i] = x[i];
            }
        } else {
            const bool* x = static_cast<const bool*>(in.data);
            for (int64_t i = 0; i < outCount; ++i) {
                y[i] = x[i] ? 1 : 0;
            }
        }
    } else {
        bool* y = static_cast<bool*>(out.data);
        if (in.dtype == DataType::Float32) {
            const float* x = static_cast<const float*>(in.data);
            for (int64_t i = 0; i < outCount; ++i) {
                y[i] = x[i] != 0.0F;
            }
        } else if (in.dtype == DataType::Int64) {
            const int64_t* x = static_cast<const int64_t*>(in.data);
            for (int64_t i = 0; i < outCount; ++i) {
                y[i] = x[i] != 0;
            }
        } else {
            const bool* x = static_cast<const bool*>(in.data);
            for (int64_t i = 0; i < outCount; ++i) {
                y[i] = x[i];
            }
        }
    }
}

} // namespace kernels
} // namespace engine
