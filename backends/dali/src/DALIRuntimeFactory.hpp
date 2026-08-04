#pragma once

#include "DALIInfer.hpp"
#include "HostTensorConverter.hpp"
#include "IAllocator.hpp"
#include "IBackendRuntimeFactory.hpp"
#include "ITensorConverter.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Abstract Factory for the DALI runtime family: pairs the DALI GPU
// preprocessing adapter with the default host allocator and tensor converter.
// The pipeline runs on the GPU; outputs are handed back through host memory, so
// the host converter is the right pairing until a device-to-device path exists.
class DALIRuntimeFactory : public IBackendRuntimeFactory {
  public:
    std::unique_ptr<InferenceInterface> create_backend(const std::string& model_path, bool use_gpu, size_t batch_size,
                                                       const std::vector<std::vector<int64_t>>& input_sizes) override {
        return std::make_unique<DALIInfer>(model_path, use_gpu, batch_size, input_sizes);
    }

    std::unique_ptr<IAllocator> create_allocator() override { return std::make_unique<HostAllocator>(); }
    std::unique_ptr<ITensorConverter> create_converter() override { return std::make_unique<HostTensorConverter>(); }

    const char* name() const noexcept override { return "DALIRuntimeFactory"; }
};
