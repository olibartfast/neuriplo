#pragma once
#include "InferenceInterface.hpp"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

class ExecuTorchInfer : public InferenceInterface {
  public:
    ExecuTorchInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                    const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    std::vector<int64_t> resolve_shape(const std::vector<int64_t>& metadata_shape,
                                       const std::vector<std::vector<int64_t>>& input_sizes, size_t index) const;

    executorch::extension::Module module_;
    std::vector<executorch::aten::ScalarType> input_types_;
};
