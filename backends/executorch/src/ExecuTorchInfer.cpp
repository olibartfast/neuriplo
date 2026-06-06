#include "ExecuTorchInfer.hpp"

#include <cstring>
#include <glog/logging.h>
#include <numeric>

using executorch::aten::ScalarType;
using executorch::extension::make_tensor_ptr;
using executorch::extension::TensorPtr;

namespace {

template <typename Container> std::vector<int64_t> to_int64_dims(const Container& dims) {
    std::vector<int64_t> result;
    result.reserve(dims.size());
    for (const auto dim : dims) {
        result.push_back(static_cast<int64_t>(dim));
    }
    return result;
}

std::vector<executorch::aten::SizesType> to_executorch_dims(const std::vector<int64_t>& dims) {
    std::vector<executorch::aten::SizesType> result;
    result.reserve(dims.size());
    for (const auto dim : dims) {
        result.push_back(static_cast<executorch::aten::SizesType>(dim));
    }
    return result;
}

size_t num_elements(const std::vector<int64_t>& dims) {
    if (dims.empty()) {
        return 0;
    }

    return std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1),
                           [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });
}

} // namespace

ExecuTorchInfer::ExecuTorchInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                                 const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}, module_(model_path) {
    LOG(INFO) << "ExecuTorch backend configured with delegate: " << NEURIPLO_EXECUTORCH_DELEGATE;

    // The delegate is selected when the .pte is exported and is consumed by
    // whichever backend library neuriplo was linked against. XNNPACK is an
    // optimized CPU delegate, so no configuration reports a GPU device here;
    // hardware-accelerated delegates (QNN, Vulkan, ...) are not built into this
    // neuriplo configuration.
    gpu_available_ = false;
    if (use_gpu) {
        LOG(WARNING) << "ExecuTorch backend: hardware-accelerated delegates are not built into this "
                        "neuriplo configuration; using the linked '"
                     << NEURIPLO_EXECUTORCH_DELEGATE << "' delegate";
    }

    const auto method_meta = module_.method_meta("forward");
    if (!method_meta.ok()) {
        throw ModelLoadException("ExecuTorch forward() metadata is unavailable");
    }

    for (size_t i = 0; i < method_meta->num_inputs(); ++i) {
        const auto input_meta = method_meta->input_tensor_meta(i);
        if (!input_meta.ok()) {
            throw ModelLoadException("ExecuTorch input metadata is unavailable for input " + std::to_string(i));
        }

        const auto shape = resolve_shape(to_int64_dims(input_meta->sizes()), input_sizes, i);
        inference_metadata_.addInput("input_" + std::to_string(i), shape, batch_size_);
        input_types_.push_back(input_meta->scalar_type());
    }

    for (size_t i = 0; i < method_meta->num_outputs(); ++i) {
        const auto output_meta = method_meta->output_tensor_meta(i);
        if (!output_meta.ok()) {
            inference_metadata_.addOutput("output_" + std::to_string(i), {-1}, batch_size_);
            continue;
        }

        inference_metadata_.addOutput("output_" + std::to_string(i), to_int64_dims(output_meta->sizes()), batch_size_);
    }

    state_ = BackendState::Ready;
}

std::vector<int64_t> ExecuTorchInfer::resolve_shape(const std::vector<int64_t>& metadata_shape,
                                                    const std::vector<std::vector<int64_t>>& input_sizes,
                                                    size_t index) const {
    const bool has_dynamic_dims = metadata_shape.empty() || std::any_of(metadata_shape.begin(), metadata_shape.end(),
                                                                        [](int64_t dim) { return dim <= 0; });

    if (has_dynamic_dims) {
        if (index >= input_sizes.size() || input_sizes[index].empty()) {
            throw ModelLoadException("ExecuTorch input " + std::to_string(index) +
                                     " requires explicit input_sizes because the exported shape is dynamic");
        }
        return input_sizes[index];
    }

    if (index < input_sizes.size() && !input_sizes[index].empty()) {
        return input_sizes[index];
    }

    return metadata_shape;
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
ExecuTorchInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    validate_input(input_tensors);

    if (input_tensors.size() != 1) {
        throw InferenceExecutionException(
            "ExecuTorch backend currently supports single-input forward() execution in neuriplo");
    }

    const auto& input_meta = inference_metadata_.getInputs().at(0);
    const auto et_shape = to_executorch_dims(input_meta.shape);
    const auto expected_numel = num_elements(input_meta.shape);
    const auto input_type = input_types_.empty() ? ScalarType::Float : input_types_.front();

    TensorPtr input_tensor;

    switch (input_type) {
    case ScalarType::Float: {
        const auto expected_bytes = expected_numel * sizeof(float);
        if (input_tensors[0].size() != expected_bytes) {
            throw InferenceExecutionException("ExecuTorch float input byte size mismatch");
        }
        std::vector<float> values(expected_numel);
        std::memcpy(values.data(), input_tensors[0].data(), expected_bytes);
        input_tensor = make_tensor_ptr(et_shape, values);
        break;
    }
    case ScalarType::Int: {
        const auto expected_bytes = expected_numel * sizeof(int32_t);
        if (input_tensors[0].size() != expected_bytes) {
            throw InferenceExecutionException("ExecuTorch int32 input byte size mismatch");
        }
        std::vector<int32_t> values(expected_numel);
        std::memcpy(values.data(), input_tensors[0].data(), expected_bytes);
        input_tensor = make_tensor_ptr(et_shape, values);
        break;
    }
    case ScalarType::Long: {
        const auto expected_bytes = expected_numel * sizeof(int64_t);
        if (input_tensors[0].size() != expected_bytes) {
            throw InferenceExecutionException("ExecuTorch int64 input byte size mismatch");
        }
        std::vector<int64_t> values(expected_numel);
        std::memcpy(values.data(), input_tensors[0].data(), expected_bytes);
        input_tensor = make_tensor_ptr(et_shape, values);
        break;
    }
    case ScalarType::Byte: {
        if (input_tensors[0].size() != expected_numel) {
            throw InferenceExecutionException("ExecuTorch uint8 input byte size mismatch");
        }
        std::vector<uint8_t> values = input_tensors[0];
        input_tensor = make_tensor_ptr(et_shape, values);
        break;
    }
    default:
        throw InferenceExecutionException("ExecuTorch input scalar type is not supported by neuriplo");
    }

    start_timer();
    const auto result = module_.forward(input_tensor);
    end_timer();

    if (!result.ok()) {
        throw InferenceExecutionException("ExecuTorch forward() failed");
    }

    std::vector<std::vector<TensorElement>> output_vectors;
    std::vector<std::vector<int64_t>> shape_vectors;

    for (size_t i = 0; i < result->size(); ++i) {
        const auto output_tensor = result->at(i).toTensor();
        const auto output_shape = to_int64_dims(output_tensor.sizes());
        const auto output_numel = static_cast<size_t>(output_tensor.numel());

        std::vector<TensorElement> output_values;
        output_values.reserve(output_numel);

        switch (output_tensor.scalar_type()) {
        case ScalarType::Float: {
            const float* data = output_tensor.const_data_ptr<float>();
            for (size_t j = 0; j < output_numel; ++j)
                output_values.emplace_back(data[j]);
            break;
        }
        case ScalarType::Int: {
            const int32_t* data = output_tensor.const_data_ptr<int32_t>();
            for (size_t j = 0; j < output_numel; ++j)
                output_values.emplace_back(data[j]);
            break;
        }
        case ScalarType::Long: {
            const int64_t* data = output_tensor.const_data_ptr<int64_t>();
            for (size_t j = 0; j < output_numel; ++j)
                output_values.emplace_back(data[j]);
            break;
        }
        case ScalarType::Byte: {
            const uint8_t* data = output_tensor.const_data_ptr<uint8_t>();
            for (size_t j = 0; j < output_numel; ++j)
                output_values.emplace_back(data[j]);
            break;
        }
        default:
            throw InferenceExecutionException("ExecuTorch output scalar type is not supported by neuriplo");
        }

        output_vectors.push_back(std::move(output_values));
        shape_vectors.push_back(output_shape);
    }

    return {std::move(output_vectors), std::move(shape_vectors)};
}
