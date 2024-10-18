#include "OVInfer.hpp" 

OVInfer::OVInfer(const std::string& model_path, const std::string& model_config, bool use_gpu) : 
    InferenceInterface{model_path, model_config, use_gpu}
{

    model_ = core_.read_model(model_config);
    compiled_model_ = core_.compile_model(model_);
    infer_request_ = compiled_model_.create_infer_request();
    ov::Shape s = compiled_model_.input().get_shape();
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> OVInfer::get_infer_results(const cv::Mat& input_blob) 
{
    std::vector<std::vector<TensorElement>> outputs;
    std::vector<std::vector<int64_t>> shapes;

    // Create input tensor
    ov::Tensor input_tensor(compiled_model_.input().get_element_type(), 
                            compiled_model_.input().get_shape(), 
                            input_blob.data);

    // Set input tensor for model with one input
    infer_request_.set_input_tensor(input_tensor);    
    infer_request_.infer();  // Perform inference

    // Get output tensor
    auto output_tensor = infer_request_.get_output_tensor();
    const float *output_buffer = output_tensor.data<const float>();  // Get pointer to output buffer
    std::size_t output_size = output_tensor.get_size();  // Get the total size of the output tensor

    // Extract the shape of the output tensor
    std::vector<int64_t> output_shape(output_tensor.get_shape().begin(), 
                                      output_tensor.get_shape().end());

    // Extract the data and store it as TensorElement (std::variant)
    std::vector<TensorElement> output;
    for (std::size_t i = 0; i < output_size; ++i) {
        output.push_back(static_cast<float>(output_buffer[i]));  // Wrap each float in TensorElement
    }

    // Store the output and shape
    outputs.emplace_back(output);
    shapes.emplace_back(output_shape);

    return std::make_tuple(outputs, shapes);  // Return tuple of outputs and shapes
}
