#pragma once

#include "InferenceInterface.hpp"

#include <cstring>

namespace neuriplo::detail {

inline void transpose_nchw_to_nhwc(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int batch, int channels,
                                   int height, int width) {
    if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0) {
        throw InferenceExecutionException("LiteRT transpose requires positive dimensions");
    }
    const size_t elements = static_cast<size_t>(batch) * channels * height * width;
    if (elements == 0 || src.size() != dst.size() || src.size() % elements != 0) {
        throw InferenceExecutionException("LiteRT transpose input byte size mismatch");
    }
    const size_t element_bytes = src.size() / elements;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    const int nchw_idx = ((b * channels + c) * height + h) * width + w;
                    const int nhwc_idx = ((b * height + h) * width + w) * channels + c;
                    std::memcpy(dst.data() + static_cast<size_t>(nhwc_idx) * element_bytes,
                                src.data() + static_cast<size_t>(nchw_idx) * element_bytes, element_bytes);
                }
            }
        }
    }
}

} // namespace neuriplo::detail
