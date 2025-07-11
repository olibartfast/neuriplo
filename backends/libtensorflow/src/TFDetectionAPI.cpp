#include "TFDetectionAPI.hpp"

enum class CHW {C=1, H, W};

TFDetectionAPI::TFDetectionAPI(const std::string& model_path, 
    bool use_gpu, 
    size_t batch_size, 
    const std::vector<std::vector<int64_t>>& input_sizes) : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}
{
    // Handle mock mode for testing
    if (model_path == "mock_model") {
        LOG(INFO) << "Initializing TensorFlow in mock mode for testing";
        
        // Create mock model info for testing
        std::vector<int64_t> mock_input_shape = {3, 224, 224}; // CHW format
        std::vector<int64_t> mock_output_shape = {1000};
        
        model_info_.addInput("mock_input", mock_input_shape, batch_size);
        model_info_.addOutput("mock_output", mock_output_shape, batch_size);
        
        input_name_ = "mock_input";
        output_names_.push_back("mock_output");
        
        LOG(INFO) << "Mock TensorFlow initialization completed";
        return;
    }

    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    tensorflow::Status status = LoadSavedModel(session_options, run_options, 
        model_path, {"serve"}, &bundle_);

    if (!status.ok()) {
        LOG(ERROR) << "Error loading the model: " << status.ToString();
        throw std::runtime_error("Failed to load TensorFlow model: " + status.ToString());
    }

    session_.reset(bundle_.GetSession());

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

    // Extract dimensions and convert from NHWC to NCHW layout (excluding batch)
    std::vector<int64_t> input_shape;
    const auto& dim = input_info_.tensor_shape().dim();

    // NHWC: [Batch, Height, Width, Channels]
    // NCHW: [Batch, Channels, Height, Width]
    // Extract Channels (dim 3), Height (dim 1), Width (dim 2)
    input_shape.push_back(dim[3].size());  // Channels
    input_shape.push_back(dim[1].size());  // Height
    input_shape.push_back(dim[2].size());  // Width

    LOG(INFO) << "Reshaped Tensor (NCHW order, excluding batch): "
              << "[" << input_shape[0] << ", " << input_shape[1] << ", " << input_shape[2] << "]";
    
    model_info_.addInput(input_name_, input_shape, batch_size);

    // Get output tensor names and shapes (excluding batch size)
    LOG(INFO) << "Tensor output names and shapes:";
    for (const auto& output : signature_def.outputs()) {
        output_names_.push_back(output.second.name());
        LOG(INFO) << output.second.name();

        std::vector<int64_t> output_shape;
        for (int i = 1; i < output.second.tensor_shape().dim_size(); ++i) { // Start from index 1 to skip batch size
            output_shape.push_back(output.second.tensor_shape().dim(i).size());
            }
        model_info_.addOutput(output.second.name(), output_shape, batch_size);
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> TFDetectionAPI::get_infer_results(const cv::Mat& input_blob) 
{
    // Handle mock mode
    if (model_path_ == "mock_model") {
        LOG(INFO) << "Running mock inference for testing";
        
        // Create mock results
        std::vector<std::vector<TensorElement>> results;
        std::vector<std::vector<int64_t>> shapes;
        
        // Mock output: 1000 random float values
        std::vector<TensorElement> mock_output(1000);
        for (int i = 0; i < 1000; ++i) {
            mock_output[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        
        results.push_back(mock_output);
        shapes.push_back({1000});
        
        return std::make_tuple(results, shapes);
    }

    tensorflow::Tensor input_tensor(input_info_.dtype(), 
        tensorflow::TensorShape({1, input_blob.size[0], input_blob.size[1], input_blob.channels()}));
  
    std::memcpy(input_tensor.flat<float>().data(), input_blob.data, input_blob.total() * input_blob.elemSize());

    // Prepare inputs for running the session
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_for_session = {
        {input_name_, input_tensor}
    };

    // Run the inference
    std::vector<tensorflow::Tensor> outputs;
    auto status = session_->Run(inputs_for_session, output_names_, {}, &outputs);
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