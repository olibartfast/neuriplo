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
// The exporter records input dtypes and static shapes inside package metadata.
// input_sizes remain required (full shapes, batch included) and are checked
// against that contract before a typed probe discovers output shapes. Device
// mismatches fail instead of silently running on another device.
class LibtorchAOTIInfer : public InferenceInterface {
  public:
    LibtorchAOTIInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                      const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());
    ~LibtorchAOTIInfer() override;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    torch::Device device_{torch::kCPU};
    std::vector<c10::ScalarType> input_types_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
};
