#pragma once

#include "InferenceInterface.hpp"

#include <string>
#include <torch/torch.h>
#include <vector>

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
