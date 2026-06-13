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

std::vector<int64_t> shape_with_batch(const std::vector<int64_t>& shape, size_t batch_size) {
    if (shape.empty()) {
        return {static_cast<int64_t>(batch_size)};
    }
    std::vector<int64_t> result = shape;
    result[0] = static_cast<int64_t>(batch_size);
    return result;
}

std::string ecdet_input_name(size_t index) { return index == 0 ? "images" : "orig_target_sizes"; }

std::string ecdet_output_name(size_t index) {
    static const char* names[] = {"labels", "boxes", "scores"};
    if (index < 3) {
        return names[index];
    }
    return "output_" + std::to_string(index);
}

TensorPtr make_input_tensor(ScalarType input_type, const std::vector<int64_t>& shape,
                            const std::vector<uint8_t>& bytes) {
    const auto et_shape = to_executorch_dims(shape);
    const auto expected_numel = num_elements(shape);

    switch (input_type) {
    case ScalarType::Float: {
        const auto expected_bytes = expected_numel * sizeof(float);
        if (bytes.size() != expected_bytes) {
            throw InferenceExecutionException("ExecuTorch float input byte size mismatch");
        }
        std::vector<float> values(expected_numel);
        std::memcpy(values.data(), bytes.data(), expected_bytes);
        return make_tensor_ptr(et_shape, std::move(values));
    }
    case ScalarType::Int: {
        const auto expected_bytes = expected_numel * sizeof(int32_t);
        if (bytes.size() != expected_bytes) {
            throw InferenceExecutionException("ExecuTorch int32 input byte size mismatch");
        }
        std::vector<int32_t> values(expected_numel);
        std::memcpy(values.data(), bytes.data(), expected_bytes);
        return make_tensor_ptr(et_shape, std::move(values));
    }
    case ScalarType::Long: {
        const auto expected_bytes = expected_numel * sizeof(int64_t);
        if (bytes.size() != expected_bytes) {
            throw InferenceExecutionException("ExecuTorch int64 input byte size mismatch");
        }
        std::vector<int64_t> values(expected_numel);
        std::memcpy(values.data(), bytes.data(), expected_bytes);
        return make_tensor_ptr(et_shape, std::move(values));
    }
    case ScalarType::Byte: {
        if (bytes.size() != expected_numel) {
            throw InferenceExecutionException("ExecuTorch uint8 input byte size mismatch");
        }
        return make_tensor_ptr(et_shape, bytes);
    }
    default:
        throw InferenceExecutionException("ExecuTorch input scalar type is not supported by neuriplo");
    }
}

} // namespace

TensorDataType ExecuTorchInfer::inputTensorDataType(ScalarType type) {
    switch (type) {
    case ScalarType::Float:
        return TensorDataType::Float32;
    case ScalarType::Int:
        return TensorDataType::Int32;
    case ScalarType::Long:
        return TensorDataType::Int64;
    case ScalarType::Byte:
        return TensorDataType::UInt8;
    case ScalarType::Char:
        return TensorDataType::Int8;
    case ScalarType::Bool:
        return TensorDataType::Bool;
    default:
        throw std::runtime_error("Unsupported ExecuTorch input tensor scalar type for metadata datatype");
    }
}

TensorDataType ExecuTorchInfer::outputTensorDataType(ScalarType type) {
    switch (type) {
    case ScalarType::Float:
        return TensorDataType::Float32;
    case ScalarType::Int:
        return TensorDataType::Int32;
    case ScalarType::Long:
        return TensorDataType::Int64;
    case ScalarType::Byte:
        return TensorDataType::UInt8;
    default:
        throw std::runtime_error("Unsupported ExecuTorch output tensor scalar type for metadata datatype");
    }
}

TensorDtype ExecuTorchInfer::scalarTypeToRawDtype(ScalarType type) {
    switch (type) {
    case ScalarType::Float:
        return TensorDtype::FP32;
    case ScalarType::Int:
        return TensorDtype::INT32;
    case ScalarType::Long:
        return TensorDtype::INT64;
    case ScalarType::Byte:
        return TensorDtype::UINT8;
    default:
        throw InferenceExecutionException("ExecuTorch output scalar type is not supported by neuriplo");
    }
}

ExecuTorchInfer::ExecuTorchInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                                 const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}, module_(model_path) {
    LOG(INFO) << "ExecuTorch backend configured with delegate: " << NEURIPLO_EXECUTORCH_DELEGATE;

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

    const bool ecdet_contract = method_meta->num_inputs() == 2 && method_meta->num_outputs() == 3;

    for (size_t i = 0; i < method_meta->num_inputs(); ++i) {
        const auto input_meta = method_meta->input_tensor_meta(i);
        if (!input_meta.ok()) {
            throw ModelLoadException("ExecuTorch input metadata is unavailable for input " + std::to_string(i));
        }

        const auto shape =
            shape_with_batch(resolve_shape(to_int64_dims(input_meta->sizes()), input_sizes, i), batch_size_);
        const std::string name = ecdet_contract ? ecdet_input_name(i) : ("input_" + std::to_string(i));
        inference_metadata_.addInput(name, shape, batch_size_, inputTensorDataType(input_meta->scalar_type()));
        input_types_.push_back(input_meta->scalar_type());
    }

    for (size_t i = 0; i < method_meta->num_outputs(); ++i) {
        const auto output_meta = method_meta->output_tensor_meta(i);
        if (!output_meta.ok()) {
            const std::string name = ecdet_contract ? ecdet_output_name(i) : ("output_" + std::to_string(i));
            inference_metadata_.addOutput(name, {-1}, batch_size_);
            continue;
        }

        const auto shape = shape_with_batch(to_int64_dims(output_meta->sizes()), batch_size_);
        const std::string name = ecdet_contract ? ecdet_output_name(i) : ("output_" + std::to_string(i));
        inference_metadata_.addOutput(name, shape, batch_size_, outputTensorDataType(output_meta->scalar_type()));
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
        std::vector<int64_t> shape = input_sizes[index];
        if (shape.size() == metadata_shape.size() - 1) {
            shape.insert(shape.begin(), static_cast<int64_t>(batch_size_));
        }
        return shape;
    }

    if (index < input_sizes.size() && !input_sizes[index].empty()) {
        std::vector<int64_t> shape = input_sizes[index];
        if (shape.size() == metadata_shape.size() - 1) {
            shape.insert(shape.begin(), static_cast<int64_t>(batch_size_));
        }
        return shape;
    }

    return metadata_shape;
}

executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
ExecuTorchInfer::run_forward(const std::vector<std::vector<uint8_t>>& input_tensors) {
    validate_input(input_tensors);

    const auto& inputs = inference_metadata_.getInputs();
    if (input_tensors.size() != inputs.size()) {
        throw InferenceExecutionException("ExecuTorch input tensor count mismatch. Expected " +
                                          std::to_string(inputs.size()) + ", got " +
                                          std::to_string(input_tensors.size()));
    }

    bound_input_tensors_.clear();
    bound_input_values_.clear();
    bound_input_tensors_.reserve(inputs.size());
    bound_input_values_.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto input_type = i < input_types_.size() ? input_types_[i] : ScalarType::Float;
        bound_input_tensors_.push_back(make_input_tensor(input_type, inputs[i].shape, input_tensors[i]));
        bound_input_values_.emplace_back(*bound_input_tensors_.back());
    }

    start_timer();
    auto result = module_.forward(bound_input_values_);
    end_timer();
    return std::move(result);
}

void ExecuTorchInfer::append_output_tensors(const std::vector<executorch::runtime::EValue>& result_values,
                                            std::vector<std::vector<TensorElement>>& output_vectors,
                                            std::vector<std::vector<int64_t>>& shape_vectors) const {
    for (size_t i = 0; i < result_values.size(); ++i) {
        const auto output_tensor = result_values[i].toTensor();
        const auto output_shape = to_int64_dims(output_tensor.sizes());
        const auto output_numel = static_cast<size_t>(output_tensor.numel());

        std::vector<TensorElement> output_values;
        output_values.reserve(output_numel);

        switch (output_tensor.scalar_type()) {
        case ScalarType::Float: {
            const float* data = output_tensor.const_data_ptr<float>();
            for (size_t j = 0; j < output_numel; ++j) {
                output_values.emplace_back(data[j]);
            }
            break;
        }
        case ScalarType::Int: {
            const int32_t* data = output_tensor.const_data_ptr<int32_t>();
            for (size_t j = 0; j < output_numel; ++j) {
                output_values.emplace_back(data[j]);
            }
            break;
        }
        case ScalarType::Long: {
            const int64_t* data = output_tensor.const_data_ptr<int64_t>();
            for (size_t j = 0; j < output_numel; ++j) {
                output_values.emplace_back(data[j]);
            }
            break;
        }
        case ScalarType::Byte: {
            const uint8_t* data = output_tensor.const_data_ptr<uint8_t>();
            for (size_t j = 0; j < output_numel; ++j) {
                output_values.emplace_back(data[j]);
            }
            break;
        }
        default:
            throw InferenceExecutionException("ExecuTorch output scalar type is not supported by neuriplo");
        }

        output_vectors.push_back(std::move(output_values));
        shape_vectors.push_back(output_shape);
    }
}

std::vector<RawOutputTensor>
ExecuTorchInfer::raw_outputs_from_values(const std::vector<executorch::runtime::EValue>& result_values) const {
    std::vector<RawOutputTensor> raw_outputs;
    raw_outputs.reserve(result_values.size());

    for (const auto& value : result_values) {
        const auto output_tensor = value.toTensor();
        const auto output_shape = to_int64_dims(output_tensor.sizes());
        const auto output_numel = static_cast<size_t>(output_tensor.numel());
        const auto output_type = output_tensor.scalar_type();

        RawOutputTensor raw;
        raw.shape = output_shape;
        raw.dtype = scalarTypeToRawDtype(output_type);

        switch (output_type) {
        case ScalarType::Float: {
            const auto* data = output_tensor.const_data_ptr<float>();
            raw.bytes.assign(reinterpret_cast<const uint8_t*>(data),
                             reinterpret_cast<const uint8_t*>(data) + output_numel * sizeof(float));
            break;
        }
        case ScalarType::Int: {
            const auto* data = output_tensor.const_data_ptr<int32_t>();
            raw.bytes.assign(reinterpret_cast<const uint8_t*>(data),
                             reinterpret_cast<const uint8_t*>(data) + output_numel * sizeof(int32_t));
            break;
        }
        case ScalarType::Long: {
            const auto* data = output_tensor.const_data_ptr<int64_t>();
            raw.bytes.assign(reinterpret_cast<const uint8_t*>(data),
                             reinterpret_cast<const uint8_t*>(data) + output_numel * sizeof(int64_t));
            break;
        }
        case ScalarType::Byte: {
            const auto* data = output_tensor.const_data_ptr<uint8_t>();
            raw.bytes.assign(data, data + output_numel);
            break;
        }
        default:
            throw InferenceExecutionException("ExecuTorch output scalar type is not supported by neuriplo");
        }

        raw_outputs.push_back(std::move(raw));
    }

    return raw_outputs;
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
ExecuTorchInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    const auto result = run_forward(input_tensors);
    if (!result.ok()) {
        throw InferenceExecutionException("ExecuTorch forward() failed");
    }

    std::vector<std::vector<TensorElement>> output_vectors;
    std::vector<std::vector<int64_t>> shape_vectors;
    append_output_tensors(*result, output_vectors, shape_vectors);
    return {std::move(output_vectors), std::move(shape_vectors)};
}

std::vector<RawOutputTensor>
ExecuTorchInfer::get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) {
    const auto result = run_forward(input_tensors);
    if (!result.ok()) {
        throw InferenceExecutionException("ExecuTorch forward() failed");
    }

    return raw_outputs_from_values(*result);
}
