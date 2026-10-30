#pragma once

#include "InferenceInterface.hpp"

#include <c10/core/ScalarType.h>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

inline constexpr const char* kTorchScriptInputDtypesFile = "neuriplo_input_dtypes.txt";
inline constexpr const char* kAotiInputDtypesKey = "NEURIPLO_INPUT_DTYPES";
inline constexpr const char* kAotiInputShapesKey = "NEURIPLO_INPUT_SHAPES";

inline std::vector<c10::ScalarType> parse_libtorch_input_dtypes(const std::string& text, size_t expected) {
    if (text.empty()) {
        return {};
    }
    std::istringstream stream(text);
    std::string header;
    size_t count = 0;
    if (!(stream >> header >> count) || header != "NEURIPLO_INPUT_DTYPES_V1" || count != expected) {
        throw ModelLoadException("Invalid LibTorch input datatype metadata header or input count");
    }
    std::vector<c10::ScalarType> types;
    types.reserve(expected);
    for (size_t i = 0; i < expected; ++i) {
        std::string tag;
        if (!(stream >> tag)) {
            throw ModelLoadException("Missing LibTorch input datatype entry " + std::to_string(i));
        }
        if (tag == "FP32") {
            types.push_back(c10::ScalarType::Float);
        } else if (tag == "INT32") {
            types.push_back(c10::ScalarType::Int);
        } else if (tag == "INT64") {
            types.push_back(c10::ScalarType::Long);
        } else if (tag == "UINT8") {
            types.push_back(c10::ScalarType::Byte);
        } else if (tag == "INT8") {
            types.push_back(c10::ScalarType::Char);
        } else if (tag == "BOOL") {
            types.push_back(c10::ScalarType::Bool);
        } else {
            throw ModelLoadException("Unsupported LibTorch input datatype: " + tag);
        }
    }
    std::string trailing;
    if (stream >> trailing) {
        throw ModelLoadException("Unexpected trailing LibTorch input datatype metadata");
    }
    return types;
}

inline size_t libtorch_input_bytes(const std::vector<int64_t>& shape, c10::ScalarType type) {
    size_t bytes = c10::elementSize(type);
    for (const auto dimension : shape) {
        if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max() ||
            bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension)) {
            throw std::runtime_error("Invalid or overflowing LibTorch input shape");
        }
        bytes *= static_cast<size_t>(dimension);
    }
    return bytes;
}

// AOTInductor's optional native runtime checks cannot be relied on to protect
// raw buffers. Verify the compiler-recorded static shapes before any probe.
inline void validate_aoti_input_shapes(const std::string& text, const std::vector<std::vector<int64_t>>& provided) {
    std::istringstream stream(text);
    std::string header;
    size_t count = 0;
    if (!(stream >> header >> count) || header != "NEURIPLO_INPUT_SHAPES_V1" || count != provided.size()) {
        throw ModelLoadException("Missing or invalid AOTInductor input shape metadata; re-export the package");
    }
    for (const auto& shape : provided) {
        size_t rank = 0;
        if (!(stream >> rank) || rank != shape.size()) {
            throw ModelLoadException("AOTInductor input rank does not match the compiled package");
        }
        for (const auto dimension : shape) {
            int64_t expected = 0;
            if (!(stream >> expected) || expected <= 0 || expected != dimension) {
                throw ModelLoadException("AOTInductor input shape does not match the compiled package");
            }
        }
    }
    if (stream >> header) {
        throw ModelLoadException("Unexpected trailing AOTInductor input shape metadata");
    }
}
