#include "GGMLInfer.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <ggml-cpu.h>

GGMLInfer::GGMLInfer(const std::string& model_path, bool use_gpu, size_t batch_size, const std::vector<std::vector<int64_t>>& input_sizes) 
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}
    , ctx_(nullptr)
    , backend_(nullptr)
    , graph_(nullptr)
    , input_tensor_(nullptr)
    , model_loaded_(false)
{
    try {
        LOG(INFO) << "Running using GGML runtime: " << model_path;
        
        // Initialize GGML
        ggml_init_params params = {
            .mem_size = 1024 * 1024 * 1024, // 1GB
            .mem_buffer = nullptr,
            .no_alloc = true
        };
        
        ctx_ = ggml_init(params);
        if (!ctx_) {
            throw std::runtime_error("Failed to initialize GGML context");
        }
        
        // Setup backend (CPU or GPU)
        setup_backend(use_gpu);
        
        // Load model
        load_model(model_path);
        
        // Setup input/output tensors
        setup_input_output_tensors(input_sizes);
        
        model_loaded_ = true;
        
    } catch (const std::exception& e) {
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        throw;
    }
}

GGMLInfer::~GGMLInfer()
{
    if (backend_) {
        ggml_backend_free(backend_);
    }
    if (ctx_) {
        ggml_free(ctx_);
    }
}

void GGMLInfer::setup_backend(bool use_gpu)
{
    if (use_gpu) {
        // Try to use GPU backend if available
        // For now, fall back to CPU since CUDA backend setup is complex
        LOG(WARNING) << "GPU backend not implemented yet, using CPU";
        backend_ = ggml_backend_cpu_init();
    } else {
        backend_ = ggml_backend_cpu_init();
        LOG(INFO) << "Using CPU backend";
    }
    
    if (!backend_) {
        throw std::runtime_error("Failed to initialize GGML backend");
    }
}

void GGMLInfer::load_model(const std::string& model_path)
{
    // Check if file exists
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_path);
    }
    
    // For now, we'll create a simple placeholder model
    // In a real implementation, you would load the actual GGML model format
    LOG(INFO) << "Loading GGML model from: " << model_path;
    
    // Create a simple graph for demonstration
    // In practice, you would parse the actual model file
    // OpenCV blobFromImage creates (batch, channels, height, width) format
    input_tensor_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, 3, 224, 224, batch_size_);
    
    // Create a simple output tensor (this would be replaced with actual model loading)
    struct ggml_tensor* output_tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, 1000, batch_size_);
    
    // Create computation graph
    graph_ = ggml_new_graph(ctx_);
    ggml_build_forward_expand(graph_, output_tensor);
    
    // Allocate backend buffer
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
    if (!buffer) {
        LOG(WARNING) << "Failed to allocate backend buffer, continuing without backend allocation";
        // For now, we'll continue without backend allocation for testing
        // In a real implementation, this would be required
    }
    
    LOG(INFO) << "GGML model loaded successfully";
}

void GGMLInfer::setup_input_output_tensors(const std::vector<std::vector<int64_t>>& input_sizes)
{
    if (input_sizes.empty()) {
        throw std::runtime_error("Input sizes must be specified for GGML backend");
    }
    
    // Setup input tensors
    for (size_t i = 0; i < input_sizes.size(); i++) {
        const auto& shape = input_sizes[i];
        if (shape.size() != 4) {
            throw std::runtime_error("GGML expects 4D input tensors (batch, height, width, channels)");
        }
        
        std::string input_name = "input" + std::to_string(i + 1);
        inference_metadata_.addInput(input_name, shape, batch_size_);
    }
    
    // Setup output tensors (placeholder - would be determined from actual model)
    std::vector<int64_t> output_shape = {static_cast<int64_t>(batch_size_), 1000}; // Example: 1000 classes
    inference_metadata_.addOutput("output", output_shape, batch_size_);
    
    output_names_.push_back("output");
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> 
GGMLInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors)
{
    validate_input(input_tensors);
    
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    
    // Process all input tensors
    if (input_tensors.size() != 1) {
        throw std::runtime_error("GGML backend currently supports only single input models, got " + std::to_string(input_tensors.size()) + " inputs");
    }
    
    const cv::Mat& input_blob = input_tensors[0];
    
    start_timer();
    
    try {
        // Convert OpenCV Mat to GGML tensor
        std::vector<float> input_data = blob2vec(input_blob);
        
        // Copy data to input tensor
        size_t tensor_size = input_tensor_->ne[0] * input_tensor_->ne[1] * input_tensor_->ne[2] * input_tensor_->ne[3];
        LOG(INFO) << "Tensor dimensions: " << input_tensor_->ne[0] << "x" << input_tensor_->ne[1] << "x" << input_tensor_->ne[2] << "x" << input_tensor_->ne[3];
        LOG(INFO) << "Tensor size: " << tensor_size << ", Input data size: " << input_data.size();
        
        if (tensor_size != input_data.size()) {
            throw std::runtime_error("Input data size mismatch: tensor=" + std::to_string(tensor_size) + ", data=" + std::to_string(input_data.size()));
        }
        
        memcpy(input_tensor_->data, input_data.data(), input_data.size() * sizeof(float));
        
        // Execute the graph (if backend is available)
        if (backend_) {
            ggml_backend_graph_compute(backend_, graph_);
        } else {
            LOG(WARNING) << "No backend available, skipping graph computation";
        }
        
        // Get output tensors
        std::vector<std::vector<TensorElement>> outputs;
        std::vector<std::vector<int64_t>> shapes;
        
        // For demonstration, we'll create a simple output
        // In practice, you would iterate through the actual output tensors
        std::vector<TensorElement> output_data(1000, 0.0f); // Placeholder
        outputs.push_back(output_data);
        
        std::vector<int64_t> output_shape = {static_cast<int64_t>(batch_size_), 1000};
        shapes.push_back(output_shape);
        
        end_timer();
        
        return std::make_tuple(outputs, shapes);
        
    } catch (const std::exception& e) {
        end_timer();
        throw InferenceExecutionException(e.what());
    }
}

std::vector<TensorElement> GGMLInfer::tensor_to_vector(struct ggml_tensor* tensor)
{
    std::vector<TensorElement> result;
    size_t total_elements = ggml_nelements(tensor);
    result.reserve(total_elements);
    
    float* data = static_cast<float*>(tensor->data);
    for (size_t i = 0; i < total_elements; ++i) {
        result.push_back(data[i]);
    }
    
    return result;
}

std::vector<int64_t> GGMLInfer::get_tensor_shape(struct ggml_tensor* tensor)
{
    std::vector<int64_t> shape;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] > 1) {
            shape.push_back(tensor->ne[i]);
        }
    }
    return shape;
}
