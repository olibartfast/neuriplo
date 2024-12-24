#include "ORTInfer.hpp"
  

ORTInfer::ORTInfer(const std::string& model_path, bool use_gpu) : InferenceInterface{model_path, "", use_gpu}
{
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Onnx Runtime Inference");
    Ort::SessionOptions session_options;

    if (use_gpu)
    {
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        LOG(INFO) << "Available providers:";
        bool is_found = false;
        for (const auto& p : providers)
        {
            LOG(INFO) << p;
            if (p.find("CUDA") != std::string::npos)
            {
                LOG(INFO) << "Using CUDA GPU";
                OrtCUDAProviderOptions cuda_options;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                is_found = true;
                break;
            }
        }
        if (!is_found)
        {
            LOG(INFO) << "CUDA GPU not available, falling back to CPU";
            session_options = Ort::SessionOptions();
        }
    }
    else
    {
        LOG(INFO) << "Using CPU";
        session_options = Ort::SessionOptions();
    }

    try
    {
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
    }
    catch (const Ort::Exception& ex)
    {
        LOG(ERROR) << "Failed to load the ONNX model: " << ex.what();
        std::exit(1);
    }

    Ort::AllocatorWithDefaultOptions allocator;
    LOG(INFO) << "Input Node Name/Shape (" << session_.GetInputCount() << "):";
    
    // Process inputs
    for (std::size_t i = 0; i < session_.GetInputCount(); i++)
    {
        auto name = session_.GetInputNameAllocated(i, allocator).get();
        auto shapes = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        auto input_type = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        
        // Handle dynamic batch size
        shapes[0] = shapes[0] == -1 ? 1 : shapes[0];
        
        LOG(INFO) << "\t" << name << " : " << print_shape(shapes);
        model_info_.addInput(name, shapes);

        std::string input_type_str = getDataTypeString(input_type);
        LOG(INFO) << "\tData Type: " << input_type_str;
    }

    // Log network dimensions from first input
    const auto& first_input = model_info_.getInputs()[0].shape;
    const auto channels = static_cast<int>(first_input[1]);
    const auto network_height = static_cast<int>(first_input[2]);
    const auto network_width = static_cast<int>(first_input[3]);

    LOG(INFO) << "channels " << channels;
    LOG(INFO) << "width " << network_width;
    LOG(INFO) << "height " << network_height;

    // Process outputs
    LOG(INFO) << "Output Node Name/Shape (" << session_.GetOutputCount() << "):";
    for (std::size_t i = 0; i < session_.GetOutputCount(); i++)
    {
        auto name = session_.GetOutputNameAllocated(i, allocator).get();
        auto shapes = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        LOG(INFO) << "\t" << name << " : " << print_shape(shapes);
        model_info_.addOutput(name, shapes);
    }
}


ModelInfo ORTInfer::get_model_info()
{
    return model_info_;
}

std::string ORTInfer::getDataTypeString(ONNXTensorElementDataType type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "Float";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "Int64";
        default:
            return "Unknown";
    }
}

std::string ORTInfer::print_shape(const std::vector<std::int64_t>& v)
{
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

size_t ORTInfer::getSizeByDim(const std::vector<int64_t>& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.size(); ++i)
    {
        if (dims[i] == -1 || dims[i] == 0)
        {
            continue;
        }
        size *= dims[i];
    }
    return size;
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> 
ORTInfer::get_infer_results(const cv::Mat& preprocessed_img)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(preprocessed_img, blob, 1.0, cv::Size(), cv::Scalar(), false, false);
    
    const auto& inputs = model_info_.getInputs();
    const auto& outputs = model_info_.getOutputs();
    
    std::vector<std::vector<float>> input_tensors(inputs.size());
    std::vector<Ort::Value> in_ort_tensors;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Process first input (image blob)
    input_tensors[0] = blob2vec(blob);
    in_ort_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensors[0].data(),
        getSizeByDim(inputs[0].shape),
        inputs[0].shape.data(),
        inputs[0].shape.size()
    ));

    // Handle RTDETR case with two inputs
    if (inputs.size() > 1)
    {
        std::vector<int64_t> orig_target_sizes = { 
            static_cast<int64_t>(blob.size[2]), 
            static_cast<int64_t>(blob.size[3]) 
        };
        in_ort_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info,
            orig_target_sizes.data(),
            getSizeByDim(orig_target_sizes),
            inputs[1].shape.data(),
            inputs[1].shape.size()
        ));
    }

    // Prepare input/output names
    std::vector<const char*> input_names_char;
    input_names_char.reserve(inputs.size());
    for (const auto& input : inputs) {
        input_names_char.push_back(input.name.c_str());
    }

    std::vector<const char*> output_names_char;
    output_names_char.reserve(outputs.size());
    for (const auto& output : outputs) {
        output_names_char.push_back(output.name.c_str());
    }

    // Run inference
    std::vector<Ort::Value> output_ort_tensors = session_.Run(
        Ort::RunOptions{ nullptr },
        input_names_char.data(),
        in_ort_tensors.data(),
        in_ort_tensors.size(),
        output_names_char.data(),
        output_names_char.size()
    );

    // Process outputs
    assert(output_ort_tensors.size() == outputs.size());
    
    std::vector<std::vector<TensorElement>> tensor_outputs;
    std::vector<std::vector<int64_t>> shapes;
    tensor_outputs.reserve(output_ort_tensors.size());
    shapes.reserve(output_ort_tensors.size());

    for (const Ort::Value& output_tensor : output_ort_tensors)
    {
        auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
        const auto& shape = tensor_info.GetShape();
        size_t num_elements = getSizeByDim(shape);

        std::vector<TensorElement> tensor_data;
        tensor_data.reserve(num_elements);

        switch (tensor_info.GetElementType()) {
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                processTensorData(tensor_data, output_tensor.GetTensorData<float>(), num_elements);
                break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                processTensorData(tensor_data, output_tensor.GetTensorData<int64_t>(), num_elements);
                break;
            default:
                LOG(ERROR) << "Unsupported tensor type: " << tensor_info.GetElementType();
                std::exit(1);
        }

        tensor_outputs.emplace_back(std::move(tensor_data));
        shapes.emplace_back(shape);
    }

    return {tensor_outputs, shapes};
}

