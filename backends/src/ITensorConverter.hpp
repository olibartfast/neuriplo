#pragma once

#include "InferenceInterface.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

// Element kinds handled across the codebase when translating between raw input
// bytes and neuriplo's TensorElement variant. Mirrors the types backends
// currently branch on when interpreting tensor data.
enum class TensorDataType { Float32, Int32, Int64, UInt8, Int8, Bool };

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
