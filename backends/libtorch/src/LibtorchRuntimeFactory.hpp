#pragma once

#include "HostTensorConverter.hpp"
#include "IAllocator.hpp"
#include "IBackendRuntimeFactory.hpp"
#include "ITensorConverter.hpp"
#include "LibtorchAOTIInfer.hpp"
#include "LibtorchInfer.hpp"
#include "LibtorchModelFormat.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Abstract Factory for the LibTorch runtime family. The backend product depends
// on the model format: ".pt2" packages go to the AOTInductor adapter, anything
// else to the TorchScript adapter. Both share the host allocator and converter.
class LibtorchRuntimeFactory : public IBackendRuntimeFactory {
  public:
    std::unique_ptr<InferenceInterface> create_backend(const std::string& model_path, bool use_gpu, size_t batch_size,
                                                       const std::vector<std::vector<int64_t>>& input_sizes) override {
        switch (detect_libtorch_model_format(model_path)) {
        case LibtorchModelFormat::AOTInductor:
            return std::make_unique<LibtorchAOTIInfer>(model_path, use_gpu, batch_size, input_sizes);
        case LibtorchModelFormat::TorchScript:
            break;
        }
        return std::make_unique<LibtorchInfer>(model_path, use_gpu, batch_size, input_sizes);
    }

    std::unique_ptr<IAllocator> create_allocator() override { return std::make_unique<HostAllocator>(); }
    std::unique_ptr<ITensorConverter> create_converter() override { return std::make_unique<HostTensorConverter>(); }

    const char* name() const noexcept override { return "LibtorchRuntimeFactory"; }
};
