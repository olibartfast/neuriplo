#pragma once
#include "InferenceInterface.hpp"

#include <memory>
#include <torch/torch.h>

namespace torch::inductor {
class AOTIModelPackageLoader;
}

// Adapter: exposes an AOTInductor package (.pt2, produced by
// torch._inductor.aoti_compile_and_package) through the common
// InferenceInterface contract.
//
// Unlike TorchScript, a .pt2 package carries no input shape or dtype
// introspection, so input_sizes are required (full shapes, batch included) and
// inputs are float32. Output shapes are discovered with one zero-filled probe
// run at load time. The package is compiled for one device; asking for another
// device fails instead of silently running elsewhere.
class LibtorchAOTIInfer : public InferenceInterface {
  public:
    LibtorchAOTIInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                      const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());
    ~LibtorchAOTIInfer() override;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    torch::Device device_{torch::kCPU};
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
};
