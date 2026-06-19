#pragma once

#include "InferenceInterface.hpp"

#include <memory>
#include <string>
#include <vector>

// TensorFlow Lite exposes these as public aliases, so they cannot be forward-declared.
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

// Adapter: exposes the LiteRT (TensorFlow Lite) runtime through the common InferenceInterface contract.
class LiteRTInfer : public InferenceInterface {
  public:
    LiteRTInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());
    ~LiteRTInfer() override;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;

    std::vector<int> makeInputDims(int tensor_index, const std::vector<int64_t>& requested_shape) const;
    std::vector<int64_t> tensorShape(int tensor_index) const;
    void refreshMetadata();
};
