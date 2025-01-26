#include "OVInfer.hpp" 
#include <filesystem>
#include <sstream>
#include <numeric>

// Helper function to print ov::Shape and ov::PartialShape
template <typename ShapeType>
std::string OVInfer::print_shape(const ShapeType& shape) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if constexpr (std::is_same<ShapeType, ov::PartialShape>::value) { // Note the 'constexpr'
            if (shape[i].is_dynamic()) {
                ss << "?";
            } else {
                ss << shape[i].get_length();
            }
        } else { // It's ov::Shape
            ss << shape[i]; 
        }

        if (i < shape.size() - 1) {
            ss << ",";
        }
    }
    ss << "]";
    return ss.str();
}

OVInfer::OVInfer(const std::string& model_path, bool use_gpu, size_t batch_size, const std::vector<std::vector<int64_t>>& input_sizes) : 
    InferenceInterface{model_path, use_gpu, batch_size, input_sizes}
{
    std::filesystem::path fs_path(model_path);
    std::string basename = fs_path.stem().string();
    const std::string model_config = model_path.substr(0, model_path.find(".bin")) + ".xml";   
    if (!std::filesystem::exists(fs_path)) {
        throw std::runtime_error("XML file must have same name as model binary");
    }    

    try {
        model_ = core_.read_model(model_config);

        // --- Handle dynamic shapes before compiling the model ---
        std::map<ov::Output<ov::Node>, ov::PartialShape> all_shapes; // Store dynamic shapes for all inputs

        LOG(INFO) << "Input Node Name/Shape (" << model_->inputs().size() << "):";
        for (size_t i = 0; i < model_->inputs().size(); ++i) {
            auto input = model_->input(i);
            std::string name = input.get_any_name();
            ov::PartialShape partial_shape = input.get_partial_shape();

            // Check for dynamic dimensions
            bool has_dynamic = false;
            for (size_t j = 0; j < partial_shape.size(); ++j) {
                if (partial_shape[j].is_dynamic()) {
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
                for (size_t j = 0; j < partial_shape.size(); ++j) {
                    if (partial_shape[j].is_dynamic()) {
                        if (provided_idx >= provided_shape.size()) {
                            throw std::runtime_error("Insufficient input sizes provided for dynamic dimensions in input '" + name + "'");
                        }
                        partial_shape[j] = provided_shape[provided_idx++];
                    }
                }
            }
            
            // Set batch size as dynamic if it was dynamic, otherwise static
            if (partial_shape[0].is_dynamic()){
                partial_shape[0] = ov::Dimension(1, -1); // Allow batch sizes from 1 upwards
            }
            else{
                partial_shape[0] = batch_size;
            }
            

            // Store the potentially updated shape
            all_shapes[input] = partial_shape;
        }

        if (!all_shapes.empty()){
          // Reshape the model with the gathered partial shapes
          model_->reshape(all_shapes);
        }
        

        // Set up device
        std::string device = use_gpu ? "GPU" : "CPU";
        LOG(INFO) << "Using device: " << device;
        compiled_model_ = core_.compile_model(model_, device);
        infer_request_ = compiled_model_.create_infer_request();

        // --- Process inputs after compilation ---
        for (size_t i = 0; i < model_->inputs().size(); ++i) {
            auto input = model_->input(i);
            std::string name = input.get_any_name();
            ov::Shape shape = input.get_shape();

            // Convert ov::Shape to std::vector<int64_t>
            std::vector<int64_t> shape_vec(shape.begin() + 1, shape.end());

            LOG(INFO) << "\t" << name << " : " << print_shape(shape);
            model_info_.addInput(name, shape_vec, batch_size); // Pass the converted shape_vec

            ov::element::Type input_type = input.get_element_type();
            LOG(INFO) << "\tData Type: " << input_type.get_type_name();
        }

        // Log network dimensions from first input
        const auto& first_input_shape_vec = model_info_.getInputs()[0].shape;
        const auto channels = static_cast<int>(first_input_shape_vec[1]);
        const auto network_height = static_cast<int>(first_input_shape_vec[2]);
        const auto network_width = static_cast<int>(first_input_shape_vec[3]);

        LOG(INFO) << "channels " << channels;
        LOG(INFO) << "width " << network_width;
        LOG(INFO) << "height " << network_height;

        // Process outputs
        LOG(INFO) << "Output Node Name/Shape (" << model_->outputs().size() << "):";
        for (size_t i = 0; i < model_->outputs().size(); ++i) {
            auto output = model_->output(i);
            std::string name;
            if (auto node = output.get_node()) {
                name = node->get_friendly_name();
            } else {
                name = "Unnamed Output"; // Default name if no friendly name is found
            }

            ov::Shape shape = output.get_shape();

            // Convert ov::Shape to std::vector<int64_t>
            std::vector<int64_t> shape_vec(shape.begin() + 1, shape.end());

            LOG(INFO) << "\t" << name << " : " << print_shape(shape);
            model_info_.addOutput(name, shape_vec, batch_size); // Pass the converted shape_vec
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