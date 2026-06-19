#pragma once

#include "IAllocator.hpp"
#include "ITensorConverter.hpp"
#include "InferenceInterface.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Abstract Factory for a coherent backend runtime family. A concrete factory
// produces an inference backend together with the allocator and tensor
// converter that belong to the same runtime, so callers never mix components
// from different backends. Interface only: no concrete factories in this step.
class IBackendRuntimeFactory {
  public:
    virtual ~IBackendRuntimeFactory() = default;

    virtual std::unique_ptr<InferenceInterface>
    create_backend(const std::string& model_path, bool use_gpu, size_t batch_size,
                   const std::vector<std::vector<int64_t>>& input_sizes) = 0;

    virtual std::unique_ptr<IAllocator> create_allocator() = 0;
    virtual std::unique_ptr<ITensorConverter> create_converter() = 0;

    virtual const char* name() const noexcept = 0;
};
