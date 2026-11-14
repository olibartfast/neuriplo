#pragma once
#include "InferenceInterface.hpp"

#include <ggml-backend.h>
#include <ggml.h>

// Adapter: exposes the GGML runtime through the common InferenceInterface contract.
class GGMLInfer : public InferenceInterface {
  private:
    struct ggml_context* ctx_;
    struct ggml_backend* backend_;
    ggml_backend_buffer_t buffer_;
    struct ggml_cgraph* graph_;
    struct ggml_tensor* feat_tensor_;
    std::vector<struct ggml_tensor*> output_tensors_;
    std::vector<std::string> output_names_;
    bool model_loaded_;
    // Fixed per-model geometry (set in load_model; validated at inference).
    // ResNet: in 1x3x224x224, out {N,1000}. YOLOv8n: in 1x3x640x640,
    // out {1,84,8400}.
    int64_t input_height_ = 224;
    int64_t input_width_ = 224;
    std::vector<int64_t> output_shape_ = {1, 1000};
    bool is_yolo_ = false;

  public:
    GGMLInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
              const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    ~GGMLInfer();

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    void load_model(const std::string& model_path);
    void setup_backend(bool use_gpu);
    void setup_input_output_tensors(const std::vector<std::vector<int64_t>>& input_sizes);
    std::vector<TensorElement> tensor_to_vector(struct ggml_tensor* tensor);
    std::vector<int64_t> get_tensor_shape(struct ggml_tensor* tensor);
};
