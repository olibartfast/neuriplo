#pragma once
#include "InferenceInterface.hpp"

#include <cactus.h>

class CactusInfer : public InferenceInterface {
  private:
    cactus_model_t model_;
    bool model_loaded_;
    size_t vocab_size_;

  public:
    CactusInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    ~CactusInfer();

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    std::string bytes_to_prompt(const std::vector<uint8_t>& data) const;
    std::vector<TensorElement> response_to_tensor(const std::string& response) const;
};
