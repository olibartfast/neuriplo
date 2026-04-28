#pragma once
#include "InferenceInterface.hpp"

#include <llama.h>

class LlamaCppInfer : public InferenceInterface {
  private:
    llama_model* model_;
    llama_context* ctx_llama_;
    const llama_vocab* vocab_;
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
};
