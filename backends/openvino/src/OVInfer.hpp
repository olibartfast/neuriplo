#pragma once
#include "InferenceInterface.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/core.hpp"

#include <sstream>

// Adapter: exposes the OpenVINO runtime through the common InferenceInterface contract.
class OVInfer : public InferenceInterface {
  public:
    OVInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
            const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;
    std::vector<RawOutputTensor> get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    // Helper function to print ov::Shape and ov::PartialShape
    template <typename ShapeType> std::string print_shape(const ShapeType& shape);

    void bind_inputs_and_infer(const std::vector<std::vector<uint8_t>>& input_tensors);

    static TensorDataType inputTensorDataType(ov::element::Type type);
    static TensorDataType outputTensorDataType(ov::element::Type type);

    ov::Core core_;
    ov::Tensor input_tensor_;
    ov::InferRequest infer_request_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
};
