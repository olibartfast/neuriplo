#pragma once

#include "InferenceInterface.hpp"
#include "TensorDataType.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

// Pure-virtual abstraction over the raw-bytes <-> typed-tensor conversion that
// backends (e.g. ORTInfer) currently inline. Interface only: this step defines
// the contract without prescribing any conversion math or behavior.
class ITensorConverter {
  public:
    virtual ~ITensorConverter() = default;

    // Raw input bytes -> backend-ready typed buffer description.
    virtual std::vector<TensorElement> to_typed(const std::vector<uint8_t>& raw_bytes, TensorDataType type) const = 0;

    // Backend output buffer -> neuriplo TensorElement vector.
    virtual std::vector<TensorElement> from_backend(const void* data, std::size_t num_elements,
                                                    TensorDataType type) const = 0;

    virtual const char* name() const noexcept = 0;
};
