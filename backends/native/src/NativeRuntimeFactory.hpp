#pragma once

#include "HostTensorConverter.hpp"
#include "IAllocator.hpp"
#include "IBackendRuntimeFactory.hpp"
#include "ITensorConverter.hpp"
#include "NativeInfer.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Abstract Factory for the first-party engine family: pairs the NativeInfer
// adapter with the default host allocator and tensor converter.
class NativeRuntimeFactory : public IBackendRuntimeFactory {
  public:
    std::unique_ptr<InferenceInterface> create_backend(const std::string& model_path, bool use_gpu, size_t batch_size,
                                                       const std::vector<std::vector<int64_t>>& input_sizes) override {
        if (use_gpu) {
            throw ::InferenceException("NATIVE runs on CPU only in this phase");
        }
        return std::make_unique<NativeInfer>(model_path, use_gpu, batch_size, input_sizes);
    }

    std::unique_ptr<IAllocator> create_allocator() override { return std::make_unique<HostAllocator>(); }
    std::unique_ptr<ITensorConverter> create_converter() override { return std::make_unique<HostTensorConverter>(); }

    const char* name() const noexcept override { return "NativeRuntimeFactory"; }
};
