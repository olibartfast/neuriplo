#include "LibtorchInfer.hpp"


LibtorchInfer::LibtorchInfer(const std::string& model_path, bool use_gpu, size_t batch_size, const std::vector<std::vector<int64_t>>& input_sizes) : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}
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

    module_ = torch::jit::load(model_path, device_);

}

ModelInfo LibtorchInfer::get_model_info()
{
    return model_info_;
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