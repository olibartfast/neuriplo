#include "NeuriploEngineInfer.hpp"
#include "ORTInfer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kFixtureEnv = "NEURIPLO_YOLO_PARITY_FIXTURE";
constexpr const char* kInputEnv = "NEURIPLO_YOLO_PARITY_INPUT";
constexpr size_t kInputElements = 1u * 3u * 640u * 640u;
constexpr size_t kOutputElements = 1u * 300u * 6u;
constexpr size_t kRows = kOutputElements / 6u;
constexpr double kConfidenceThreshold = 0.25;
constexpr double kMinIou = 0.999;
constexpr double kMaxAllowedAbsDiff = 1e-4;

// The fixture path and the preprocessed tensor path are supplied by ctest
// through the environment. A missing or absent fixture is a hard failure:
// never a skip, never a mock.
std::string resolve_env(const char* name) {
    const char* from_env = std::getenv(name);
    if (from_env != nullptr && from_env[0] != '\0') {
        return from_env;
    }
    return "";
}

// Packet Y11: the input is the deterministic tensor provisioned from the real
// image (PIL resize + /255, see provision_yolo_fixture.py), read as raw
// little-endian float32. Both backends receive the same buffer.
std::vector<std::vector<uint8_t>> load_input(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return {};
    }
    std::vector<uint8_t> bytes(kInputElements * sizeof(float));
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (stream.gcount() != static_cast<std::streamsize>(bytes.size())) {
        return {};
    }
    uint8_t probe[4] = {0, 0, 0x80, 0x3E}; // little-endian 0.25f
    float probe_value = 0.0F;
    std::memcpy(&probe_value, probe, sizeof(float));
    if (probe_value != 0.25F) {
        return {};
    }
    return {std::move(bytes)};
}

double iou(const std::array<float, 6>& a, const std::array<float, 6>& b) {
    const double left = std::max<double>(a[0], b[0]);
    const double top = std::max<double>(a[1], b[1]);
    const double right = std::min<double>(a[2], b[2]);
    const double bottom = std::min<double>(a[3], b[3]);
    const double intersection = std::max(0.0, right - left) * std::max(0.0, bottom - top);
    const double area_a =
        std::max(0.0, static_cast<double>(a[2] - a[0])) * std::max(0.0, static_cast<double>(a[3] - a[1]));
    const double area_b =
        std::max(0.0, static_cast<double>(b[2] - b[0])) * std::max(0.0, static_cast<double>(b[3] - b[1]));
    const double total = area_a + area_b - intersection;
    if (total <= 0.0) {
        return 0.0;
    }
    return intersection / total;
}

int row_class(const std::array<float, 6>& row) { return static_cast<int>(row[5]); }

double row_score(const std::array<float, 6>& row) { return static_cast<double>(row[4]); }

std::vector<std::array<float, 6>> confident_rows(const float* flat) {
    std::vector<std::array<float, 6>> rows;
    for (size_t r = 0; r < kRows; ++r) {
        std::array<float, 6> row{};
        for (int c = 0; c < 6; ++c) {
            row[static_cast<size_t>(c)] = flat[r * 6u + static_cast<size_t>(c)];
        }
        if (row_score(row) > kConfidenceThreshold) {
            rows.push_back(row);
        }
    }
    return rows;
}

std::string dump_row(const char* side, size_t index, const std::array<float, 6>& row) {
    return std::string(side) + " row " + std::to_string(index) + ": x1=" + std::to_string(row[0]) +
           " y1=" + std::to_string(row[1]) + " x2=" + std::to_string(row[2]) + " y2=" + std::to_string(row[3]) +
           " score=" + std::to_string(row[4]) + " class=" + std::to_string(row[5]) + "\n";
}

std::string dump_rows(const std::vector<std::array<float, 6>>& rows, const char* side) {
    std::string text;
    for (size_t i = 0; i < rows.size(); ++i) {
        text += dump_row(side, i, rows[i]);
    }
    return text;
}

std::vector<float> flatten_floats(const std::vector<std::vector<TensorElement>>& outputs) {
    std::vector<float> values;
    values.reserve(outputs[0].size());
    for (const TensorElement& element : outputs[0]) {
        values.push_back(std::get<float>(element));
    }
    return values;
}

} // namespace

// Packet Y11: the identical real-image tensor through NEURIPLO_ENGINE and
// ONNX_RUNTIME. Confidence-gated comparison: confident rows only (score >
// 0.25), matched one-to-one by class and IoU. Replaces the Y9 noise-input
// elementwise gates: background TopK tie selection is backend-noise-dependent
// and is explicitly out of scope.
TEST(YoloParityTest, NeuriploEngineAndOnnxRuntimeConfidentRowsAgree) {
    const std::string fixture = resolve_env(kFixtureEnv);
    ASSERT_FALSE(fixture.empty()) << kFixtureEnv
                                  << " is not set; the YOLO parity fixture must be provisioned before this test runs";
    ASSERT_TRUE(std::filesystem::exists(fixture)) << "YOLO parity fixture does not exist: " << fixture;

    const std::string input_path = resolve_env(kInputEnv);
    ASSERT_FALSE(input_path.empty())
        << kInputEnv << " is not set; the YOLO parity input tensor must be provisioned before this test runs";
    ASSERT_TRUE(std::filesystem::exists(input_path)) << "YOLO parity input tensor does not exist: " << input_path;

    const std::vector<std::vector<uint8_t>> input = load_input(input_path);
    ASSERT_FALSE(input.empty()) << "YOLO parity input tensor is unreadable, truncated, or not little-endian float32: "
                                << input_path;
    ASSERT_EQ(input[0].size(), kInputElements * sizeof(float));

    std::vector<float> native_output;
    try {
        NeuriploEngineInfer native(fixture, /*use_gpu=*/false);
        auto [outputs, shapes] = native.get_infer_results(input);
        ASSERT_EQ(outputs.size(), 1u) << "NEURIPLO_ENGINE returned " << outputs.size() << " outputs, expected 1";
        ASSERT_EQ(shapes.size(), 1u) << "NEURIPLO_ENGINE returned " << shapes.size() << " shapes, expected 1";
        ASSERT_EQ(outputs[0].size(), kOutputElements)
            << "NEURIPLO_ENGINE returned " << outputs[0].size() << " elements, expected " << kOutputElements;
        native_output = flatten_floats(outputs);
    } catch (const std::exception& ex) {
        FAIL() << "NEURIPLO_ENGINE inference failed: " << ex.what();
    }

    std::vector<float> ort_output;
    try {
        ORTInfer ort(fixture, /*use_gpu=*/false);
        auto [outputs, shapes] = ort.get_infer_results(input);
        ASSERT_EQ(outputs.size(), 1u) << "ONNX_RUNTIME returned " << outputs.size() << " outputs, expected 1";
        ASSERT_EQ(shapes.size(), 1u) << "ONNX_RUNTIME returned " << shapes.size() << " shapes, expected 1";
        ASSERT_EQ(outputs[0].size(), kOutputElements)
            << "ONNX_RUNTIME returned " << outputs[0].size() << " elements, expected " << kOutputElements;
        ort_output = flatten_floats(outputs);
    } catch (const std::exception& ex) {
        FAIL() << "ONNX_RUNTIME inference failed: " << ex.what();
    }

    const std::vector<std::array<float, 6>> native_rows = confident_rows(native_output.data());
    const std::vector<std::array<float, 6>> ort_rows = confident_rows(ort_output.data());

    const size_t confident_rows_count = native_rows.size();
    std::cout << "YOLO_PARITY_CONFIDENT_ROWS=" << confident_rows_count << std::endl;

    // Greedy one-to-one matching: each ORT confident row is matched to the
    // best-IoU still-unmatched NEURIPLO_ENGINE confident row of the same class.
    std::vector<bool> native_matched(native_rows.size(), false);
    double min_iou = 0.0;
    double max_score_diff = 0.0;
    bool have_match_metric = false;
    std::vector<std::string> failures;

    for (size_t o = 0; o < ort_rows.size(); ++o) {
        const std::array<float, 6>& ort_row = ort_rows[o];
        int best_native = -1;
        double best_iou = 0.0;
        for (size_t n = 0; n < native_rows.size(); ++n) {
            if (native_matched[n] || row_class(native_rows[n]) != row_class(ort_row)) {
                continue;
            }
            const double value = iou(native_rows[n], ort_row);
            if (value > best_iou) {
                best_iou = value;
                best_native = static_cast<int>(n);
            }
        }
        if (best_native < 0 || best_iou < kMinIou) {
            failures.push_back("no NEURIPLO_ENGINE match (same class, IoU >= " + std::to_string(kMinIou) +
                               ") for ORT row; best IoU " + std::to_string(best_iou));
            failures.push_back(dump_row("ORT", o, ort_row));
            continue;
        }
        native_matched[static_cast<size_t>(best_native)] = true;
        const double score_diff =
            std::fabs(row_score(native_rows[static_cast<size_t>(best_native)]) - row_score(ort_row));
        if (!have_match_metric) {
            min_iou = best_iou;
            max_score_diff = score_diff;
            have_match_metric = true;
        } else {
            min_iou = std::min(min_iou, best_iou);
            max_score_diff = std::max(max_score_diff, score_diff);
        }
        if (score_diff > kMaxAllowedAbsDiff) {
            failures.push_back("score diff " + std::to_string(score_diff) + " exceeds " +
                               std::to_string(kMaxAllowedAbsDiff));
            failures.push_back(dump_row("NEURIPLO_ENGINE", static_cast<size_t>(best_native),
                                        native_rows[static_cast<size_t>(best_native)]));
            failures.push_back(dump_row("ORT", o, ort_row));
        }
    }

    std::cout << "YOLO_PARITY_MIN_IOU=" << min_iou << std::endl;
    std::cout << "YOLO_PARITY_MAX_SCORE_DIFF=" << max_score_diff << std::endl;

    EXPECT_FALSE(native_rows.empty())
        << "zero confident NEURIPLO_ENGINE rows: the image did not produce confident detections "
        << "(score > " << kConfidenceThreshold << ")";
    EXPECT_FALSE(ort_rows.empty())
        << "zero confident ONNX_RUNTIME rows: the image did not produce confident detections " << "(score > "
        << kConfidenceThreshold << ")";
    EXPECT_EQ(native_rows.size(), ort_rows.size())
        << "confident-row count mismatch: NEURIPLO_ENGINE has " << native_rows.size() << ", ONNX_RUNTIME has "
        << ort_rows.size() << "\n"
        << dump_rows(native_rows, "NEURIPLO_ENGINE") << dump_rows(ort_rows, "ORT");
    EXPECT_TRUE(failures.empty());

    for (const std::string& failure : failures) {
        ADD_FAILURE() << failure;
    }
}
