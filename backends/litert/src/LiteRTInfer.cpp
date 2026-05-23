#include "LiteRTInfer.hpp"

#include <cstring>
#include <numeric>
#include <stdexcept>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

namespace {

std::string tensor_type_name(TfLiteType type) {
    switch (type) {
    case kTfLiteFloat32:
        return "float32";
    case kTfLiteUInt8:
        return "uint8";
    case kTfLiteInt32:
        return "int32";
    case kTfLiteInt64:
        return "int64";
    default:
        return "unsupported";
    }
}

std::vector<int64_t> shape_from_dims(const TfLiteIntArray* dims) {
    std::vector<int64_t> shape;
    if (dims == nullptr) {
        return shape;
    }
    shape.reserve(static_cast<size_t>(dims->size));
    for (int i = 0; i < dims->size; ++i) {
        shape.push_back(dims->data[i]);
    }
    return shape;
}

template <typename T> void append_tensor_data(std::vector<TensorElement>& output, const T* data, size_t count) {
    output.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        output.emplace_back(data[i]);
    }
}

size_t element_count_from_shape(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, [](size_t total, int64_t dim) {
        if (dim <= 0) {
            throw InferenceExecutionException("Tensor has invalid runtime dimension: " + std::to_string(dim));
        }
        return total * static_cast<size_t>(dim);
    });
}

} // namespace

LiteRTInfer::LiteRTInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                         const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
    if (use_gpu) {
        LOG(WARNING) << "LiteRT backend currently uses the CPU interpreter; GPU/delegate selection is not wired";
    }

    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_) {
        throw ModelLoadException("Unable to load LiteRT flatbuffer: " + model_path);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    if (builder(&interpreter_) != kTfLiteOk || !interpreter_) {
        throw ModelLoadException("Unable to create LiteRT interpreter for: " + model_path);
    }

    const auto& inputs = interpreter_->inputs();
    for (size_t i = 0; i < input_sizes.size() && i < inputs.size(); ++i) {
        const std::vector<int> dims = makeInputDims(inputs[i], input_sizes[i]);
        if (interpreter_->ResizeInputTensor(inputs[i], dims) != kTfLiteOk) {
            throw ModelLoadException("Unable to resize LiteRT input tensor at index " + std::to_string(i));
        }
    }

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        throw ModelLoadException("Unable to allocate LiteRT tensors for: " + model_path);
    }

    refreshMetadata();
}

LiteRTInfer::~LiteRTInfer() = default;

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
LiteRTInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    validate_input(input_tensors);

    const auto& input_indices = interpreter_->inputs();
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        TfLiteTensor* input = interpreter_->tensor(input_indices[i]);
        if (input == nullptr) {
            throw InferenceExecutionException("LiteRT input tensor is null at index " + std::to_string(i));
        }
        if (input_tensors[i].size() != input->bytes) {
            throw InferenceExecutionException("LiteRT input tensor byte size mismatch at index " + std::to_string(i) +
                                              ": expected " + std::to_string(input->bytes) + ", got " +
                                              std::to_string(input_tensors[i].size()));
        }
        std::memcpy(input->data.raw, input_tensors[i].data(), input->bytes);
    }

    start_timer();
    if (interpreter_->Invoke() != kTfLiteOk) {
        throw InferenceExecutionException("LiteRT interpreter invocation failed");
    }
    end_timer();

    std::vector<std::vector<TensorElement>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    const auto& output_indices = interpreter_->outputs();
    outputs.reserve(output_indices.size());
    shapes.reserve(output_indices.size());

    for (int tensor_index : output_indices) {
        const TfLiteTensor* output = interpreter_->tensor(tensor_index);
        if (output == nullptr) {
            throw InferenceExecutionException("LiteRT output tensor is null");
        }

        std::vector<int64_t> shape = shape_from_dims(output->dims);
        const size_t element_count = element_count_from_shape(shape);
        std::vector<TensorElement> tensor_data;

        switch (output->type) {
        case kTfLiteFloat32:
            append_tensor_data(tensor_data, output->data.f, element_count);
            break;
        case kTfLiteUInt8:
            append_tensor_data(tensor_data, output->data.uint8, element_count);
            break;
        case kTfLiteInt32:
            append_tensor_data(tensor_data, output->data.i32, element_count);
            break;
        case kTfLiteInt64:
            append_tensor_data(tensor_data, output->data.i64, element_count);
            break;
        default:
            throw InferenceExecutionException("Unsupported LiteRT output tensor type: " +
                                              tensor_type_name(output->type));
        }

        shapes.push_back(std::move(shape));
        outputs.push_back(std::move(tensor_data));
    }

    return std::make_tuple(outputs, shapes);
}

std::vector<int> LiteRTInfer::makeInputDims(int tensor_index, const std::vector<int64_t>& requested_shape) const {
    const TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
    if (tensor == nullptr || tensor->dims == nullptr) {
        throw ModelLoadException("LiteRT input tensor metadata is unavailable");
    }

    std::vector<int> dims;
    dims.reserve(static_cast<size_t>(tensor->dims->size));

    if (requested_shape.size() == static_cast<size_t>(tensor->dims->size)) {
        for (int64_t dim : requested_shape) {
            dims.push_back(static_cast<int>(dim));
        }
    } else if (requested_shape.size() + 1 == static_cast<size_t>(tensor->dims->size)) {
        dims.push_back(static_cast<int>(batch_size_));
        for (int64_t dim : requested_shape) {
            dims.push_back(static_cast<int>(dim));
        }
    } else {
        throw ModelLoadException("LiteRT input shape rank mismatch: model rank " + std::to_string(tensor->dims->size) +
                                 ", requested rank " + std::to_string(requested_shape.size()));
    }

    return dims;
}

std::vector<int64_t> LiteRTInfer::tensorShape(int tensor_index) const {
    const TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
    if (tensor == nullptr) {
        return {};
    }
    return shape_from_dims(tensor->dims);
}

void LiteRTInfer::refreshMetadata() {
    const auto& inputs = interpreter_->inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const TfLiteTensor* tensor = interpreter_->tensor(inputs[i]);
        const std::string name =
            tensor != nullptr && tensor->name != nullptr ? tensor->name : "input" + std::to_string(i);
        inference_metadata_.addInput(name, tensorShape(inputs[i]), batch_size_);
    }

    const auto& outputs = interpreter_->outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        const TfLiteTensor* tensor = interpreter_->tensor(outputs[i]);
        const std::string name =
            tensor != nullptr && tensor->name != nullptr ? tensor->name : "output" + std::to_string(i);
        inference_metadata_.addOutput(name, tensorShape(outputs[i]), batch_size_);
    }
}
