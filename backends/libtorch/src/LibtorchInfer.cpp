#include "LibtorchInfer.hpp"
#include <sstream>

std::string LibtorchInfer::print_shape(const std::vector<int64_t>& shape)
{
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        ss << shape[i];
        if (i < shape.size() - 1)
        {
            ss << ", ";
        }
    }
    ss << ")";
    return ss.str();
}

LibtorchInfer::LibtorchInfer(const std::string& model_path, bool use_gpu, size_t batch_size, const std::vector<std::vector<int64_t>>& input_sizes) 
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}
{
    if (use_gpu && torch::cuda::is_available())
    {
        device_ = torch::kCUDA;
        LOG(INFO) << "Using CUDA GPU";
    }
    else
    {
        device_ = torch::kCPU;
        LOG(INFO) << "Using CPU";
    }

    try
    {
        module_ = torch::jit::load(model_path, device_);
    }
    catch (const c10::Error& e)
    {
        LOG(ERROR) << "Failed to load the LibTorch model: " << e.what();
        std::exit(1);
    }

    // Process inputs
    LOG(INFO) << "Input Node Name/Shape:";
    auto method = module_.get_method("forward");
    auto graph = method.graph();
    auto inputs = graph->inputs();
    
    // Skip the first input as it's usually the self/module input
    for (size_t i = 1; i < inputs.size(); ++i)
    {
        auto input = inputs[i];
        std::string name = input->debugName();
        auto type = input->type()->cast<c10::TensorType>();
        if (!type)
        {
            LOG(WARNING) << "Input " << name << " is not a tensor. Skipping.";
            continue;
        }

        auto shapes = type->sizes().concrete_sizes();
        if (!shapes)
        {
            if (input_sizes.empty() || (i-1) >= input_sizes.size())
            {
                throw std::runtime_error("LibtorchInfer Initialitazion Error: Dynamic shapes found but no input sizes provided for input '" + name + "'");
            }
            shapes = input_sizes[i-1];
        }

        std::vector<int64_t> final_shape = *shapes;

        LOG(INFO) << "\t" << name << " : " << print_shape(final_shape);
        model_info_.addInput(name, final_shape, batch_size);

        std::string input_type_str = type->scalarType().has_value() ? toString(type->scalarType().value()) : "Unknown";
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
    LOG(INFO) << "Output Node Name/Shape:";
    auto outputs = graph->outputs();
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto output = outputs[i];
        std::string name = output->debugName();
        auto type = output->type()->cast<c10::TensorType>();
        if (!type)
        {
            LOG(WARNING) << "Output " << name << " is not a tensor. Skipping.";
            continue;
        }

        auto shapes = type->sizes().concrete_sizes();
        if (!shapes)
        {
            LOG(WARNING) << "Output " << name << " has dynamic shape. Using (-1) as placeholder.";
            shapes = std::vector<int64_t>(type->dim().value_or(1), -1);
        }

        std::vector<int64_t> final_shape = *shapes;
        final_shape[0] = batch_size; // Set batch size

        LOG(INFO) << "\t" << name << " : " << print_shape(final_shape);
        model_info_.addOutput(name, final_shape, batch_size);
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> 
LibtorchInfer::get_infer_results(const cv::Mat& preprocessed_img)
{
    // Convert the input image to a blob swapping channels order from hwc to chw    
    cv::Mat blob;
    cv::dnn::blobFromImage(preprocessed_img, blob, 1.0, cv::Size(), cv::Scalar(), false, false);
    // Convert the input tensor to a Torch tensor
    torch::Tensor input = torch::from_blob(blob.data, 
        { 1, blob.size[1], blob.size[2], blob.size[3] }, 
        torch::kFloat32);
    input = input.to(device_);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto output = module_.forward(inputs);

    std::vector<std::vector<TensorElement>> output_vectors;
    std::vector<std::vector<int64_t>> shape_vectors;

    // Helper function to process a single tensor
    auto process_tensor = [](const torch::Tensor& tensor) {
        std::vector<TensorElement> tensor_data;
        tensor_data.reserve(tensor.numel());
        
        const auto data_type = tensor.scalar_type();
        switch (data_type) {
            case torch::kFloat32: {
                const float* output_data = tensor.data_ptr<float>();
                for (size_t i = 0; i < tensor.numel(); ++i) {
                    tensor_data.emplace_back(output_data[i]);
                }
                break;
            }
            case torch::kInt64: {
                const int64_t* output_data = tensor.data_ptr<int64_t>();
                for (size_t i = 0; i < tensor.numel(); ++i) {
                    tensor_data.emplace_back(output_data[i]);
                }
                break;
            }
            default:
                LOG(ERROR) << "Unsupported tensor type: " << data_type;
                std::exit(1);
        }
        return tensor_data;
    };

    if (output.isTuple()) {
        // Handle tuple output
        auto tuple_outputs = output.toTuple()->elements();
        for (const auto& output_tensor : tuple_outputs) {
            if (!output_tensor.isTensor()) {
                continue;
            }
            
            torch::Tensor tensor = output_tensor.toTensor()
                                              .to(torch::kCPU)
                                              .contiguous();
            
            output_vectors.push_back(process_tensor(tensor));
            shape_vectors.push_back(tensor.sizes().vec());
        }
    } else if (output.isTensor()) {
        // Handle single tensor output
        torch::Tensor tensor = output.toTensor()
                                    .to(torch::kCPU)
                                    .contiguous();
        
        output_vectors.push_back(process_tensor(tensor));
        shape_vectors.push_back(tensor.sizes().vec());
    } else {
        LOG(ERROR) << "Unsupported output type: neither tensor nor tuple";
        std::exit(1);
    }

    return std::make_tuple(output_vectors, shape_vectors);
}