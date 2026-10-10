#include "NativeInfer.hpp"
#include "ORTInfer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace {

constexpr const char* kManifestEnv = "NEURIPLO_MODEL_ZOO_MANIFEST";
constexpr const char* kOnlyEnv = "NEURIPLO_MODEL_ZOO_ONLY";

// One manifest row: `name  path  dims[;dims...]  mode  atol  [input.bin]`,
// whitespace separated. `dims` is one `AxBxC` shape per graph input. `mode` is
// `parity` (run both backends, compare every output), `e2e` (end-to-end
// detection output [1,N,C>=6] with the score in column 4: compare only rows ONNX Runtime scores above 0.25,
// since near-tied background rows select backend-noise-dependent cells), or
// `load` (NATIVE must load the model and plan it; nothing runs). The optional
// input file replaces the seeded input for a single-input model. Each element must satisfy
// |native - ort| <= atol * (1 + |ort|).
struct ZooEntry {
    std::string name;
    std::string path;
    std::vector<std::vector<int64_t>> input_sizes;
    std::string mode;
    double atol = 1e-4;
    std::string input_file; // optional raw little-endian float32 input
};

std::vector<int64_t> parse_dims(const std::string& text) {
    std::vector<int64_t> dims;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, 'x')) {
        dims.push_back(std::stoll(part));
    }
    return dims;
}

std::vector<ZooEntry> read_manifest(const std::string& path) {
    std::vector<ZooEntry> entries;
    std::ifstream stream(path);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream fields(line);
        ZooEntry entry;
        std::string shapes;
        fields >> entry.name >> entry.path >> shapes >> entry.mode >> entry.atol >> entry.input_file;
        if (entry.name.empty()) {
            continue;
        }
        std::stringstream shape_stream(shapes);
        std::string one;
        while (std::getline(shape_stream, one, ';')) {
            entry.input_sizes.push_back(parse_dims(one));
        }
        entries.push_back(entry);
    }
    return entries;
}

// Deterministic uniform [0, 1) input, identical for both backends.
std::vector<std::vector<uint8_t>> make_inputs(const std::vector<std::vector<int64_t>>& sizes) {
    std::vector<std::vector<uint8_t>> inputs;
    uint32_t state = 0x9E3779B9U;
    for (const auto& dims : sizes) {
        size_t count = 1;
        for (int64_t dim : dims) {
            count *= static_cast<size_t>(dim);
        }
        std::vector<uint8_t> bytes(count * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            state = state * 1664525U + 1013904223U;
            const float value = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
            std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(float));
        }
        inputs.push_back(std::move(bytes));
    }
    return inputs;
}

double as_double(const TensorElement& element) {
    return std::visit([](auto value) { return static_cast<double>(value); }, element);
}

void check_entry(const ZooEntry& entry) {
    SCOPED_TRACE(entry.name);
    ASSERT_TRUE(std::filesystem::exists(entry.path)) << "model does not exist: " << entry.path;

    if (entry.mode == "load") {
        try {
            NativeInfer native(entry.path, /*use_gpu=*/false, 1, entry.input_sizes);
            std::cout << "ZOO " << entry.name << " LOADED" << std::endl;
        } catch (const std::exception& ex) {
            FAIL() << entry.name << ": NATIVE load failed: " << ex.what();
        }
        return;
    }
    ASSERT_TRUE(entry.mode == "parity" || entry.mode == "e2e") << "unknown mode " << entry.mode;

    auto inputs = make_inputs(entry.input_sizes);
    if (!entry.input_file.empty()) {
        std::ifstream stream(entry.input_file, std::ios::binary);
        ASSERT_TRUE(stream.good()) << "cannot open input file " << entry.input_file;
        stream.read(reinterpret_cast<char*>(inputs[0].data()), static_cast<std::streamsize>(inputs[0].size()));
        ASSERT_EQ(stream.gcount(), static_cast<std::streamsize>(inputs[0].size())) << "input file is truncated";
    }
    std::vector<std::vector<TensorElement>> native_out;
    std::vector<std::vector<int64_t>> native_shapes;
    try {
        NativeInfer native(entry.path, /*use_gpu=*/false, 1, entry.input_sizes);
        std::tie(native_out, native_shapes) = native.get_infer_results(inputs);
    } catch (const std::exception& ex) {
        FAIL() << entry.name << ": NATIVE failed: " << ex.what();
    }
    std::vector<std::vector<TensorElement>> ort_out;
    std::vector<std::vector<int64_t>> ort_shapes;
    try {
        // ORTInfer takes per-input dims without the batch axis, plus the batch
        // size; NativeInfer takes full shapes.
        std::vector<std::vector<int64_t>> ort_sizes;
        size_t batch = 1;
        for (const auto& dims : entry.input_sizes) {
            ort_sizes.emplace_back(dims.begin() + 1, dims.end());
            batch = static_cast<size_t>(dims.front());
        }
        ORTInfer ort(entry.path, /*use_gpu=*/false, batch, ort_sizes);
        std::tie(ort_out, ort_shapes) = ort.get_infer_results(inputs);
    } catch (const std::exception& ex) {
        FAIL() << entry.name << ": ONNX_RUNTIME failed: " << ex.what();
    }

    ASSERT_EQ(native_out.size(), ort_out.size()) << "output count differs";
    if (entry.mode == "e2e") {
        ASSERT_EQ(ort_shapes[0].size(), 3U);
        ASSERT_GE(ort_shapes[0][2], 6);
        const size_t rows = static_cast<size_t>(ort_shapes[0][1]);
        const size_t cols = static_cast<size_t>(ort_shapes[0][2]);
        size_t confident = 0;
        double max_scaled = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            if (as_double(ort_out[0][r * cols + 4]) <= 0.25) {
                continue;
            }
            ++confident;
            for (size_t c = 0; c < cols; ++c) {
                const double ref = as_double(ort_out[0][r * cols + c]);
                const double diff = std::fabs(as_double(native_out[0][r * cols + c]) - ref);
                max_scaled = std::max(max_scaled, diff / (1.0 + std::fabs(ref)));
            }
        }
        std::cout << "ZOO " << entry.name << " confident_rows=" << confident << " max_scaled_diff=" << max_scaled
                  << " atol=" << entry.atol << std::endl;
        EXPECT_GT(confident, 0U) << "no confident detections; the input is not a real image";
        EXPECT_LE(max_scaled, entry.atol) << "confident rows exceed tolerance";
        return;
    }
    for (size_t o = 0; o < ort_out.size(); ++o) {
        ASSERT_EQ(native_shapes[o], ort_shapes[o]) << "output " << o << " shape differs";
        ASSERT_EQ(native_out[o].size(), ort_out[o].size()) << "output " << o << " size differs";
        double max_ref = 0.0;
        double max_diff = 0.0;
        double max_scaled = 0.0;
        for (size_t i = 0; i < ort_out[o].size(); ++i) {
            const double ref = as_double(ort_out[o][i]);
            const double diff = std::fabs(as_double(native_out[o][i]) - ref);
            max_ref = std::max(max_ref, std::fabs(ref));
            max_diff = std::max(max_diff, diff);
            max_scaled = std::max(max_scaled, diff / (1.0 + std::fabs(ref)));
        }
        std::cout << "ZOO " << entry.name << " output" << o << " max_abs_diff=" << max_diff << " max_ref=" << max_ref
                  << " max_scaled_diff=" << max_scaled << " atol=" << entry.atol << std::endl;
        EXPECT_LE(max_scaled, entry.atol) << "output " << o << " exceeds tolerance";
    }
}

} // namespace

// Gate check for the neuriplo-tasks operator coverage packet: every model in
// the manifest loads on NATIVE, and `parity` rows match ONNX_RUNTIME on the
// same seeded input. A missing manifest fails, never skips.
TEST(ModelZooParityTest, ManifestModelsMatchOnnxRuntime) {
    const char* manifest = std::getenv(kManifestEnv);
    ASSERT_TRUE(manifest != nullptr && manifest[0] != '\0') << kManifestEnv << " is not set";
    const std::vector<ZooEntry> entries = read_manifest(manifest);
    ASSERT_FALSE(entries.empty()) << "manifest has no entries: " << manifest;

    const char* only = std::getenv(kOnlyEnv);
    for (const ZooEntry& entry : entries) {
        if (only != nullptr && only[0] != '\0' && entry.name != only) {
            continue;
        }
        check_entry(entry);
    }
}
