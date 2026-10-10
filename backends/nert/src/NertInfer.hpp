#pragma once

#include "InferenceInterface.hpp"
#include "engine/Executor.hpp"
#include "engine/Graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// Adapter exposing the first-party `engine` runtime through the shared
// InferenceInterface contract. Runs on the CPU reference spine only: the
// device layer currently supplies just CpuDevice(), so a GPU request is
// rejected rather than silently downgraded.
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
    engine::HostTensorMap run_engine(const std::vector<std::vector<uint8_t>>& input_tensors);

    engine::Graph graph_;
    std::unique_ptr<engine::Model> model_;
};
