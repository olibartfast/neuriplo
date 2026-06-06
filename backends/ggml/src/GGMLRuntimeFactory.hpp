#pragma once

#include "GGMLInfer.hpp"
#include "HostTensorConverter.hpp"
#include "IAllocator.hpp"
#include "IBackendRuntimeFactory.hpp"
#include "ITensorConverter.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Abstract Factory for the GGML runtime family: pairs the GGMLInfer adapter with
// the default host allocator and tensor converter.
class GGMLRuntimeFactory : public IBackendRuntimeFactory {
  public:
    std::unique_ptr<InferenceInterface> create_backend(const std::string& model_path, bool use_gpu, size_t batch_size,
                                                       const std::vector<std::vector<int64_t>>& input_sizes) override {
        return std::make_unique<GGMLInfer>(model_path, use_gpu, batch_size, input_sizes);
    }

    std::unique_ptr<IAllocator> create_allocator() override { return std::make_unique<HostAllocator>(); }
    std::unique_ptr<ITensorConverter> create_converter() override { return std::make_unique<HostTensorConverter>(); }

    const char* name() const noexcept override { return "GGMLRuntimeFactory"; }
};
