#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace neuriplo::testing {

// An NCHW float32 blob with every element set to `value` -- what
// cv::dnn::blobFromImage produced from a uniformly-filled CV_8UC3 image.
inline std::vector<uint8_t> constant_blob_bytes(float value, int n = 1, int c = 3, int h = 224, int w = 224) {
    const std::size_t elements = static_cast<std::size_t>(n) * c * h * w;
    std::vector<uint8_t> bytes(elements * sizeof(float));
    auto* typed = reinterpret_cast<float*>(bytes.data());
    std::fill(typed, typed + elements, value);
    return bytes;
}

inline std::vector<std::vector<uint8_t>> constant_blob_tensors(float value, int n = 1, int c = 3, int h = 224,
                                                               int w = 224) {
    return {constant_blob_bytes(value, n, c, h, w)};
}

// What blobFromImage(image, blob, 1.f / 255.f, ...) yielded for the gray
// cv::Mat::ones(rows, cols, CV_8UC3) * 128 the shared test template fed in.
// Named because the value is the whole point: feeding zeros instead would
// exercise a different path through any backend that special-cases them.
inline constexpr float kGray128Normalized = 128.0F / 255.0F;

// Byte-for-byte what cv::dnn::blobFromImage produced from a zeroed 224x224x3
// image: an all-zero NCHW float32 blob. The per-backend tests only ever fed
// zeros, so the scale factor, mean and swapRB the old call applied were all
// no-ops there.
inline std::vector<uint8_t> zero_blob_bytes(int n = 1, int c = 3, int h = 224, int w = 224) {
    const std::size_t elements = static_cast<std::size_t>(n) * c * h * w;
    return std::vector<uint8_t>(elements * sizeof(float), 0);
}

// Single-input convenience wrapper matching the get_infer_results() signature.
inline std::vector<std::vector<uint8_t>> zero_blob_tensors(int n = 1, int c = 3, int h = 224, int w = 224) {
    return {zero_blob_bytes(n, c, h, w)};
}

} // namespace neuriplo::testing
