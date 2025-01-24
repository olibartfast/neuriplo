#include "OVInfer.hpp" 
#include <filesystem>

OVInfer::OVInfer(const std::string& model_path, bool use_gpu, size_t batch_size, const std::vector<std::vector<int64_t>>& input_sizes) : 
    InferenceInterface{model_path, use_gpu, batch_size, input_sizes}
{
    std::filesystem::path fs_path(model_path);
    std::string basename = fs_path.stem().string();
    const std::string model_config = basename + ".xml";    
    if (!std::filesystem::exists(fs_path)) {
        throw std::runtime_error("XML file must have same name as model binary");
    }    

    try {
        model_ = core_.read_model(model_config);

        // Set up device
        std::string device = use_gpu ? "GPU" : "CPU";
        LOG(INFO) << "Using device: " << device;
        compiled_model_ = core_.compile_model(model_, device);
        infer_request_ = compiled_model_.create_infer_request();

        // Process inputs
        LOG(INFO) << "Input Node Name/Shape (" << model_->inputs().size() << "):";
        for (size_t i = 0; i < model_->inputs().size(); ++i) {
            auto input = model_->input(i);
            std::string name = input.get_any_name();
            ov::Shape shape = input.get_shape();

            // Handle dynamic shapes
            bool has_dynamic = false;
            for (size_t j = 1; j < shape.size(); j++) {
                if (shape[j] == ov::Dimension::dynamic()) {
                    has_dynamic = true;
                    break;
                }
            }

            if (has_dynamic) {
                if (input_sizes.empty() || i >= input_sizes.size()) {
                    throw std::runtime_error("Dynamic shapes found but no input sizes provided for input '" + name + "'");
                }
                
                const auto& provided_shape = input_sizes[i];
                size_t provided_idx = 0;
                for (size_t j = 1; j < shape.size(); j++) {
                    if (shape[j] == ov::Dimension::dynamic()) {
                        if (provided_idx >= provided_shape.size()) {
                            throw std::runtime_error("Insufficient input sizes provided for dynamic dimensions in input '" + name + "'");
                        }
                        shape[j] = provided_shape[provided_idx++];
                    }
                }
            }

            // Set batch size
            shape[0] = batch_size;

            LOG(INFO) << "\t" << name << " : " << print_shape(shape);
            model_info_.addInput(name, shape, batch_size);

            ov::element::Type input_type = input.get_element_type();
            LOG(INFO) << "\tData Type: " << input_type.get_type_name();
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
        LOG(INFO) << "Output Node Name/Shape (" << model_->outputs().size() << "):";
        for (size_t i = 0; i < model_->outputs().size(); ++i) {
            auto output = model_->output(i);
            std::string name = output.get_any_name();
            ov::Shape shape = output.get_shape();

            // Set batch size for output
            shape[0] = batch_size;

            LOG(INFO) << "\t" << name << " : " << print_shape(shape);
            model_info_.addOutput(name, shape, batch_size);
        }
    }
    catch (const ov::Exception& e) {
        LOG(ERROR) << "Failed to load or process the OpenVINO model: " << e.what();
        std::exit(1);
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> OVInfer::get_infer_results(const cv::Mat& preprocessed_img) 
{
    std::vector<std::vector<TensorElement>> outputs;
    std::vector<std::vector<int64_t>> shapes;

    // Convert the input image to a blob swapping channels order from hwc to chw    
    cv::Mat blob;
    cv::dnn::blobFromImage(preprocessed_img, blob, 1.0, cv::Size(), cv::Scalar(), false, false);

    ov::Tensor input_tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), blob.data);
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
