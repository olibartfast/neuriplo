#pragma once
#include "InferenceInterface.hpp"

#include <llama.h>
#include <mtmd-helper.h>
#include <mtmd.h>

// Adapter: exposes the llama.cpp runtime through the common InferenceInterface contract.
class LlamaCppInfer : public InferenceInterface {
  private:
    llama_model* model_;
    llama_context* ctx_llama_;
    const llama_vocab* vocab_;
    mtmd_context* ctx_mtmd_;
    bool model_loaded_;

  public:
    LlamaCppInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                  const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    ~LlamaCppInfer();

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    std::string bytes_to_prompt(const std::vector<uint8_t>& data) const;
    std::vector<TensorElement> response_to_tensor(const std::string& response) const;
    std::string apply_chat_template(const std::string& user_prompt) const;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    infer_text_only(const std::string& raw_prompt);

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    infer_multimodal(const std::string& raw_prompt, const std::vector<uint8_t>& image_bytes);

    std::string autoregressiveGenerate(llama_pos n_past);
};
