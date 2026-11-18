#pragma once
#include "InferenceInterface.hpp"

#include <ggml-backend.h>
#include <ggml.h>

// Adapter: exposes the GGML runtime through the common InferenceInterface contract.
// Strict-GGUF-only: weights come from scripts/convert_onnx_to_gguf.py
// (.gguf F32, ggml-order dims + neuriplo.rank.* KVs). No ONNX and no
// legacy NGGL path at runtime.
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
    // ResNet: in 1x3x224x224, out {N,1000}. YOLOv8n detection: in
    // 1x3x640x640, out {1,84,8400}. YOLOv8n segmentation: same input, out
    // {1,116,8400} (boxes + classes + mask coeffs) then {1,32,160,160}
    // (prototype masks).
    int64_t input_height_ = 224;
    int64_t input_width_ = 224;
    std::vector<std::vector<int64_t>> output_shapes_ = {{1, 1000}};
    bool is_yolo_ = false;
    bool is_yolo_seg_ = false;
    bool is_yolo11_ = false;
    // Debug tensors for YOLO11
    struct ggml_tensor* debug_reg_ = nullptr;
    struct ggml_tensor* debug_dm_ = nullptr;
    struct ggml_tensor* debug_d4_ = nullptr;
    struct ggml_tensor* debug_feat_[3] = {nullptr, nullptr, nullptr};
    // Backbone/neck debug
    struct ggml_tensor* debug_m4_ = nullptr;
    struct ggml_tensor* debug_m6_ = nullptr;
    struct ggml_tensor* debug_p5_ = nullptr;
    struct ggml_tensor* debug_up1_ = nullptr;
    struct ggml_tensor* debug_n13_ = nullptr;
    struct ggml_tensor* debug_up2_ = nullptr;
    struct ggml_tensor* debug_d5_ = nullptr;
    struct ggml_tensor* debug_d6_ = nullptr;
    // Early layers debug
    struct ggml_tensor* debug_m3_ = nullptr;
    struct ggml_tensor* debug_m4_cv1_ = nullptr;
    struct ggml_tensor* debug_m4_m0_cv1_ = nullptr;
    struct ggml_tensor* debug_m4_m0_cv2_ = nullptr;
    struct ggml_tensor* debug_m4_m0_add_ = nullptr;
    struct ggml_tensor* debug_m4_cv2_ = nullptr;
    struct ggml_tensor* debug_m2_ = nullptr;
    struct ggml_tensor* debug_m1_ = nullptr;
    struct ggml_tensor* debug_m0_ = nullptr;

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
