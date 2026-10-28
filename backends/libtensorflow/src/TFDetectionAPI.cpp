#include "TFDetectionAPI.hpp"

enum class CHW { C = 1, H, W };

TFDetectionAPI::TFDetectionAPI(const std::string& model_path, bool use_gpu, size_t batch_size,
                               const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    tensorflow::Status status = LoadSavedModel(session_options, run_options, model_path, {"serve"}, &bundle_);

    if (!status.ok()) {
        LOG(ERROR) << "Error loading the model: " << status.ToString();
        throw std::runtime_error("Failed to load TensorFlow model: " + status.ToString());
    }

    // session_ is not needed since we can use bundle_.GetSession() directly

    // Get the SignatureDef
    const auto& signature_def = bundle_.GetSignatures().at("serving_default");

    // Get input tensor info
    const auto& inputs = signature_def.inputs();
    if (inputs.empty()) {
        LOG(ERROR) << "No inputs found in the model";
        throw std::runtime_error("No inputs found in TensorFlow model");
    }
    input_info_ = inputs.begin()->second;
    input_name_ = input_info_.name();
    LOG(INFO) << "Tensor Input name: " << input_name_;

    if (input_info_.tensor_shape().dim_size() != 4) {
        throw std::runtime_error("TensorFlow image backend requires a rank-four NHWC input");
    }
    // Extract dimensions and convert from NHWC to NCHW layout (excluding batch)
    std::vector<int64_t> input_shape;
    const auto& dim = input_info_.tensor_shape().dim();

    // NHWC: [Batch, Height, Width, Channels]
    // NCHW: [Batch, Channels, Height, Width]
    // Extract Channels (dim 3), Height (dim 1), Width (dim 2)
    input_shape.push_back(dim[3].size()); // Channels
    input_shape.push_back(dim[1].size()); // Height
    input_shape.push_back(dim[2].size()); // Width

    LOG(INFO) << "Reshaped Tensor (NCHW order, excluding batch): " << "[" << input_shape[0] << ", " << input_shape[1]
              << ", " << input_shape[2] << "]";

    TensorDataType input_type;
    switch (input_info_.dtype()) {
    case tensorflow::DT_FLOAT:
        input_type = TensorDataType::Float32;
        break;
    case tensorflow::DT_UINT8:
        input_type = TensorDataType::UInt8;
        break;
    case tensorflow::DT_INT32:
        input_type = TensorDataType::Int32;
        break;
    default:
        throw std::runtime_error("Unsupported TensorFlow input datatype");
    }
    inference_metadata_.addInput(input_name_, input_shape, batch_size, input_type);

    // Get output tensor names and shapes (excluding batch size)
    LOG(INFO) << "Tensor output names and shapes:";
    for (const auto& output : signature_def.outputs()) {
        if (!output.second.has_name()) {
            LOG(WARNING) << "Output tensor missing name, skipping";
            continue;
        }

        std::string output_name = output.second.name();
        output_names_.push_back(output_name);
        LOG(INFO) << output_name;

        std::vector<int64_t> output_shape;
        if (output.second.has_tensor_shape()) {
            const auto& tensor_shape = output.second.tensor_shape();
            for (int i = 1; i < tensor_shape.dim_size(); ++i) { // Start from index 1 to skip batch size
                output_shape.push_back(tensor_shape.dim(i).size());
            }
        }
        inference_metadata_.addOutput(output_name, output_shape, batch_size);
    }

    state_ = BackendState::Ready;
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
TFDetectionAPI::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {

    // TensorFlow backend currently supports only single input models
    if (input_tensors.size() != 1) {
        throw std::runtime_error("TensorFlow backend currently supports only single input models, got " +
                                 std::to_string(input_tensors.size()) + " inputs");
    }

    const std::vector<uint8_t>& input_data = input_tensors[0];

    const int batch_size = batch_size_;
    const auto& shape = inference_metadata_.getInputs()[0].shape;
    const int channels = shape[0];
    const int height = shape[1];
    const int width = shape[2];

    // Create tensor with proper shape for NHWC format
    tensorflow::Tensor input_tensor(input_info_.dtype(),
                                    tensorflow::TensorShape({batch_size, height, width, channels}));

    if (input_data.size() != input_tensor.TotalBytes()) {
        throw std::runtime_error("TensorFlow input tensor byte size mismatch: expected " +
                                 std::to_string(input_tensor.TotalBytes()) + ", got " +
                                 std::to_string(input_data.size()));
    }

    // Transpose helper
    auto transpose_nchw_to_nhwc = [&](auto* dest_ptr, const auto* src_ptr) {
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        int nchw_idx = b * channels * height * width + c * height * width + h * width + w;
                        int nhwc_idx = b * height * width * channels + h * width * channels + w * channels + c;
                        dest_ptr[nhwc_idx] = src_ptr[nchw_idx];
                    }
                }
            }
        }
    };

    switch (input_info_.dtype()) {
    case tensorflow::DataType::DT_FLOAT: {
        transpose_nchw_to_nhwc(input_tensor.flat<float>().data(), reinterpret_cast<const float*>(input_data.data()));
        break;
    }
    case tensorflow::DataType::DT_UINT8: {
        transpose_nchw_to_nhwc(input_tensor.flat<uint8_t>().data(),
                               reinterpret_cast<const uint8_t*>(input_data.data()));
        break;
    }
    case tensorflow::DataType::DT_INT32: {
        transpose_nchw_to_nhwc(input_tensor.flat<int32_t>().data(),
                               reinterpret_cast<const int32_t*>(input_data.data()));
        break;
    }
    default:
        throw std::runtime_error("Unsupported input data type in TFDetectionAPI");
    }

    // Prepare inputs for running the session
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_for_session = {{input_name_, input_tensor}};

    // Run the inference
    std::vector<tensorflow::Tensor> outputs;
    auto status = bundle_.GetSession()->Run(inputs_for_session, output_names_, {}, &outputs);
    if (!status.ok()) {
        LOG(ERROR) << "Error running session: " << status.ToString();
        throw std::runtime_error("Failed to run TensorFlow session: " + status.ToString());
    }

    std::vector<std::vector<TensorElement>> convertedOutputs;
    std::vector<std::vector<int64_t>> shapes;
    for (const auto& tensor : outputs) {
        std::vector<TensorElement> outputData;
        auto numDims = tensor.dims();
        std::vector<int64_t> outputShape(numDims);

        for (int i = 0; i < numDims; ++i) {
            outputShape[i] = tensor.dim_size(i);
        }

        shapes.push_back(outputShape);

        if (tensor.dtype() == tensorflow::DataType::DT_FLOAT) {
            for (int i = 0; i < tensor.NumElements(); ++i) {
                outputData.emplace_back(tensor.flat<float>()(i)); // Use emplace_back for variant
            }
        } else if (tensor.dtype() == tensorflow::DataType::DT_INT32) {
            for (int i = 0; i < tensor.NumElements(); ++i) {
                outputData.emplace_back(tensor.flat<int32_t>()(i));
            }
        } else if (tensor.dtype() == tensorflow::DataType::DT_INT64) {
            for (int i = 0; i < tensor.NumElements(); ++i) {
                outputData.emplace_back(tensor.flat<int64_t>()(i));
            }
        } else {
            throw std::runtime_error("Unsupported output data type encountered.");
        }

        convertedOutputs.push_back(outputData);
    }
    return std::make_tuple(convertedOutputs, shapes);
}