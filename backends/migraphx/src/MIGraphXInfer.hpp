#pragma once
#include "InferenceInterface.hpp"

#include <glog/logging.h>
#include <migraphx/migraphx.hpp>

class MIGraphXInfer : public InferenceInterface {
  public:
    MIGraphXInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                  const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    migraphx::program program_;
    bool use_gpu_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    // Temporary storage used only during constructor to discover output shapes
    std::vector<std::vector<uint8_t>> dummy_buf_;
};
