#pragma once
#include <cstddef>
#include <cstdint>

// Element type tags for typed tensor buffers and metadata. Values mirror
// neuriplo_dtype_t in include/neuriplo/plugin_abi.h. Distinct from
// TensorDataType (ITensorConverter.hpp), which describes wire formats that
// widen on decode (Int8 -> Int32, Bool -> UInt8).
//
// Lives in its own header so both InferenceMetadata.hpp (LayerInfo) and
// InferenceInterface.hpp (RawOutputTensor) can carry a typed datatype without
// an include cycle.
enum class TensorDtype : uint8_t { FP32 = 0, INT32 = 1, INT64 = 2, UINT8 = 3 };

constexpr size_t tensor_dtype_size(TensorDtype dtype) noexcept {
    switch (dtype) {
    case TensorDtype::FP32:
    case TensorDtype::INT32:
        return 4;
    case TensorDtype::INT64:
        return 8;
    case TensorDtype::UINT8:
        return 1;
    }
    return 0;
}
