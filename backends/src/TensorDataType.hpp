#pragma once

// Element kinds handled across the codebase when translating between raw input
// bytes and neuriplo's TensorElement variant. This is the full set of element
// types neuriplo backends branch on when interpreting tensor data; Int8 and Bool
// widen on decode (Int8 -> Int32, Bool -> UInt8).
//
// Lives in its own header so InferenceMetadata.hpp (LayerInfo) and
// ITensorConverter.hpp can share it without an include cycle, and so tensor
// metadata can carry any backend-supported element type rather than collapsing
// everything to FP32.
enum class TensorDataType { Float32, Int32, Int64, UInt8, Int8, Bool };
