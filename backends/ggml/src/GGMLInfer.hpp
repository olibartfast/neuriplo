#pragma once
#include "InferenceInterface.hpp"
#include <ggml.h>
#include <ggml-backend.h>

class GGMLInfer : public InferenceInterface
{
private:
    struct ggml_context* ctx_;
    struct ggml_backend* backend_;
    struct ggml_cgraph* graph_;
    struct ggml_tensor* input_tensor_;
    std::vector<struct ggml_tensor*> output_tensors_;
    std::vector<std::string> output_names_;
    bool model_loaded_;
    
public:
    GGMLInfer(const std::string& model_path, 
        bool use_gpu = false, 
        size_t batch_size = 1, 
        const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    ~GGMLInfer();

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> get_infer_results(const std::vector<cv::Mat>& input_tensors) override;

private:
    void load_model(const std::string& model_path);
    void setup_backend(bool use_gpu);
    void setup_input_output_tensors(const std::vector<std::vector<int64_t>>& input_sizes);
    std::vector<TensorElement> tensor_to_vector(struct ggml_tensor* tensor);
    std::vector<int64_t> get_tensor_shape(struct ggml_tensor* tensor);
};
