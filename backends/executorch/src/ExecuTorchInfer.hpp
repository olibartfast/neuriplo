#pragma once
#include "InferenceInterface.hpp"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <vector>

// Adapter: exposes the ExecuTorch runtime through the common InferenceInterface contract.
class ExecuTorchInfer : public InferenceInterface {
  public:
    ExecuTorchInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                    const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;
    std::vector<RawOutputTensor> get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    std::vector<int64_t> resolve_shape(const std::vector<int64_t>& metadata_shape,
                                       const std::vector<std::vector<int64_t>>& input_sizes, size_t index) const;

    static TensorDataType inputTensorDataType(executorch::aten::ScalarType type);
    static TensorDataType outputTensorDataType(executorch::aten::ScalarType type);
    static TensorDtype scalarTypeToRawDtype(executorch::aten::ScalarType type);

    executorch::extension::Module module_;
    std::vector<executorch::aten::ScalarType> input_types_;
    std::vector<executorch::extension::TensorPtr> bound_input_tensors_;
    std::vector<executorch::runtime::EValue> bound_input_values_;

    executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
    run_forward(const std::vector<std::vector<uint8_t>>& input_tensors);
    void append_output_tensors(const std::vector<executorch::runtime::EValue>& result_values,
                               std::vector<std::vector<TensorElement>>& output_vectors,
                               std::vector<std::vector<int64_t>>& shape_vectors) const;
    std::vector<RawOutputTensor>
    raw_outputs_from_values(const std::vector<executorch::runtime::EValue>& result_values) const;
};
