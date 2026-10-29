#pragma once

#include "InferenceInterface.hpp"

#include <string>
#include <torch/torch.h>
#include <vector>

// Maps a TorchScript graph scalar type to the metadata datatype published for
// runtime input packing. Every backend-supported element kind is covered; a
// graph type with no neuriplo equivalent throws instead of silently collapsing
// to FP32, so auxiliary INT32/INT64/BOOL inputs keep their actual types.
inline TensorDataType libtorch_scalar_to_datatype(c10::ScalarType scalar) {
    switch (scalar) {
    case c10::ScalarType::Float:
        return TensorDataType::Float32;
    case c10::ScalarType::Int:
        return TensorDataType::Int32;
    case c10::ScalarType::Long:
        return TensorDataType::Int64;
    case c10::ScalarType::Byte:
        return TensorDataType::UInt8;
    case c10::ScalarType::Char:
        return TensorDataType::Int8;
    case c10::ScalarType::Bool:
        return TensorDataType::Bool;
    default:
        throw std::runtime_error("Unsupported LibTorch input scalar type for metadata datatype: " +
                                 std::to_string(static_cast<int>(scalar)));
    }
}

// Copies a CPU, contiguous output tensor into neuriplo's element vector. Shared
// by the TorchScript and AOTInductor adapters so both report outputs the same
// way. Only float32 and int64 are supported, as before; anything else throws.
inline std::vector<TensorElement> libtorch_tensor_to_elements(const torch::Tensor& tensor) {
    std::vector<TensorElement> tensor_data;
    tensor_data.reserve(tensor.numel());

    const auto data_type = tensor.scalar_type();
    switch (data_type) {
    case torch::kFloat32: {
        const float* output_data = tensor.data_ptr<float>();
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            tensor_data.emplace_back(output_data[i]);
        }
        break;
    }
    case torch::kInt64: {
        const int64_t* output_data = tensor.data_ptr<int64_t>();
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            tensor_data.emplace_back(output_data[i]);
        }
        break;
    }
    default:
        LOG(ERROR) << "Unsupported tensor type: " << data_type;
        throw InferenceExecutionException("Unsupported output tensor type for LibTorch: " +
                                          std::to_string(static_cast<int>(data_type)));
    }
    return tensor_data;
}
