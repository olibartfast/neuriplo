#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace neuriplo::testing {

// Byte-for-byte what cv::dnn::blobFromImage produced from a zeroed 224x224x3
// image: an all-zero NCHW float32 blob. The tests only ever fed zeros, so the
// scale factor, mean and swapRB the old call applied were all no-ops.
inline std::vector<uint8_t> zero_blob_bytes(int n = 1, int c = 3, int h = 224, int w = 224) {
    const std::size_t elements = static_cast<std::size_t>(n) * c * h * w;
    return std::vector<uint8_t>(elements * sizeof(float), 0);
}

// Single-input convenience wrapper matching the get_infer_results() signature.
inline std::vector<std::vector<uint8_t>> zero_blob_tensors(int n = 1, int c = 3, int h = 224, int w = 224) {
    return {zero_blob_bytes(n, c, h, w)};
}

} // namespace neuriplo::testing
