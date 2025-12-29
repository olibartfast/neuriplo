#include "TVMInfer.hpp"
#include <sstream>
#include <fstream>

std::string TVMInfer::print_shape(const std::vector<int64_t>& shape)
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

TVMInfer::TVMInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                   const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}, 
      module_handle_(nullptr), model_loaded_(false)
{
    // Determine device
    if (use_gpu)
    {
        device_.device_type = kDLCUDA;
        device_.device_id = 0;
        LOG(INFO) << "Using CUDA GPU for TVM inference";
    }
    else
    {
        device_.device_type = kDLCPU;
        device_.device_id = 0;
        LOG(INFO) << "Using CPU for TVM inference";
    }

    try
    {
        // For this simplified implementation, just validate that the model file exists
        std::string lib_path = model_path;
        if (lib_path.find(".so") == std::string::npos) {
            lib_path += ".so";
        }

        std::ifstream file_check(lib_path);
        if (!file_check.is_open())
        {
            throw std::runtime_error("TVM model file not found: " + lib_path);
        }
        file_check.close();

        LOG(INFO) << "TVM model file found: " << lib_path;
        model_loaded_ = true;

        // Set default shapes if not provided
        if (input_sizes.empty())
        {
            input_shapes_ = {{static_cast<int64_t>(batch_size), 3, 224, 224}};
            LOG(WARNING) << "No input shapes provided, using default: " << print_shape(input_shapes_[0]);
        }
        else
        {
            input_shapes_ = input_sizes;
        }

        // Set default output shapes
        output_shapes_ = {{static_cast<int64_t>(batch_size), 1000}};
        
        num_inputs_ = static_cast<int>(input_shapes_.size());
        num_outputs_ = static_cast<int>(output_shapes_.size());

        LOG(INFO) << "TVM model initialized successfully";
        LOG(INFO) << "Number of inputs: " << num_inputs_;
        LOG(INFO) << "Number of outputs: " << num_outputs_;

        // Process input information
        LOG(INFO) << "Input Node Name/Shape:";
        for (int i = 0; i < num_inputs_; ++i)
        {
            std::string input_name = "input_" + std::to_string(i);
            LOG(INFO) << "\t" << input_name << " : " << print_shape(input_shapes_[i]);
            inference_metadata_.addInput(input_name, input_shapes_[i], batch_size);
        }

        // Process output information
        LOG(INFO) << "Output Node Name/Shape:";
        for (int i = 0; i < num_outputs_; ++i)
        {
            std::string output_name = "output_" + std::to_string(i);
            LOG(INFO) << "\t" << output_name << " : " << print_shape(output_shapes_[i]);
            inference_metadata_.addOutput(output_name, output_shapes_[i], batch_size);
        }
    }
    catch (const std::exception& e)
    {
        LOG(ERROR) << "Failed to load TVM model: " << e.what();
        throw ModelLoadException(e.what());
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
TVMInfer::get_infer_results(const std::vector<cv::Mat>& input_images)
{
    validate_input(input_images);
    start_timer();

    if (!model_loaded_) {
        LOG(ERROR) << "TVM model not loaded";
        throw InferenceExecutionException("TVM model not loaded");
    }
    
    // TVM backend currently supports only single input models
    if (input_images.size() != 1) {
        throw std::runtime_error("TVM backend currently supports only single input models, got " + std::to_string(input_images.size()) + " inputs");
    }
    
    const cv::Mat& preprocessed_img = input_images[0];

    try
    {
        LOG(INFO) << "TVM inference requested - returning dummy results";
        LOG(INFO) << "Input image size: " << preprocessed_img.rows << "x" << preprocessed_img.cols;

        // Return dummy results for now
        std::vector<std::vector<TensorElement>> output_vectors;
        std::vector<std::vector<int64_t>> shape_vectors;

        for (int i = 0; i < num_outputs_; ++i)
        {
            const auto& output_shape = output_shapes_[i];
            
            // Calculate number of elements
            size_t num_elements = 1;
            for (auto dim : output_shape)
            {
                num_elements *= dim;
            }

            // Create dummy output data (random values for classification)
            std::vector<TensorElement> output_data;
            output_data.reserve(num_elements);
            
            for (size_t j = 0; j < num_elements; ++j)
            {
                // Generate dummy classification scores
                float score = static_cast<float>(rand()) / RAND_MAX;
                output_data.emplace_back(score);
            }

            output_vectors.push_back(output_data);
            shape_vectors.push_back(output_shape);
        }

        end_timer();
        LOG(INFO) << "TVM inference completed (dummy implementation)";
        return std::make_tuple(output_vectors, shape_vectors);
    }
    catch (const std::exception& e)
    {
        LOG(ERROR) << "TVM inference failed: " << e.what();
        throw InferenceExecutionException(e.what());
    }
}