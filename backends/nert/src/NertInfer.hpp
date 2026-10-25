#pragma once

#include "InferenceInterface.hpp"
#include "nert/executor.hpp"
#include "nert/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// Adapter exposing the first-party `nert` runtime through the shared
// InferenceInterface contract. The device follows the GPU request: CPU
// reference device by default, CUDA device when use_gpu is set and the
// engine was built with CUDA support (NEURIPLO_NERT_CUDA).
class NertInfer : public InferenceInterface {
  public:
    NertInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
              const std::vector<std::vector<int64_t>>& input_sizes = {});

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

    std::vector<RawOutputTensor> get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    // Shared input-marshalling + engine execution path for both output shapes.
    // Each input byte buffer is read in its graph input's dtype (float32,
    // int64, or bool); every output comes back in its own dtype.
    nert::HostTensorMap run_engine(const std::vector<std::vector<uint8_t>>& input_tensors);

    nert::Graph graph_;
    std::unique_ptr<nert::Model> model_;
};
