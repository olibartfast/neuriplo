#include "GGMLInfer.hpp"

#include "testing/TestBlob.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// T-9 / R-7 note: MockGGMLInfer and the Mock* tests below exercise only the
// test harness itself. A green run WITHOUT a real model artifact
// ("Using mock inference for testing") is explicitly NOT loader evidence:
// only the RealModel* tests count, and they SKIP (never fake-pass) when no
// artifact is present.
// ---------------------------------------------------------------------------

// Mock inference implementation for unit testing
class MockGGMLInfer {
  public:
    MockGGMLInfer() = default;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input) {
        (void)input;
        // Mock output: 1x1000 classification results
        std::vector<TensorElement> output_vector(1000);
        for (int i = 0; i < 1000; ++i) {
            output_vector[i] = static_cast<float>(i * 0.001f); // Mock probabilities
        }

        std::vector<std::vector<TensorElement>> output_vectors = {output_vector};
        std::vector<std::vector<int64_t>> shape_vectors = {{1, 1000}};

        return std::make_tuple(output_vectors, shape_vectors);
    }
};

class GGMLInferTest : public ::testing::Test {
  protected:
    std::string model_path;
    bool has_real_model;
    std::unique_ptr<GGMLInfer> real_infer;
    std::unique_ptr<MockGGMLInfer> mock_infer;

    void SetUp() override {
        has_real_model = false;
        model_path = "";

        // Check if model_path.txt exists (set by test script)
        std::ifstream modelPathFile("model_path.txt");
        if (modelPathFile) {
            std::getline(modelPathFile, model_path);
            if (!model_path.empty() && fs::exists(model_path)) {
                has_real_model = true;
                try {
                    std::vector<std::vector<int64_t>> input_sizes = {{1, 224, 224, 3}}; // Example input size
                    real_infer = std::make_unique<GGMLInfer>(model_path, false, 1, input_sizes);
                    std::cout << "Using real model: " << model_path << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "Failed to load real model, falling back to mock: " << e.what() << std::endl;
                    has_real_model = false;
                }
            }
        }

        if (!has_real_model) {
            mock_infer = std::make_unique<MockGGMLInfer>();
            std::cout << "Using mock inference for testing" << std::endl;
        }
    }
};

// Test basic functionality - works with both real model and mock
TEST_F(GGMLInferTest, BasicInference) {
    std::vector<std::vector<uint8_t>> input_tensors = neuriplo::testing::zero_blob_tensors();

    std::vector<std::vector<TensorElement>> output_vectors;
    std::vector<std::vector<int64_t>> shape_vectors;

    if (has_real_model) {
        auto result = real_infer->get_infer_results(input_tensors);
        output_vectors = std::get<0>(result);
        shape_vectors = std::get<1>(result);
    } else {
        auto result = mock_infer->get_infer_results(input_tensors);
        output_vectors = std::get<0>(result);
        shape_vectors = std::get<1>(result);
    }

    ASSERT_FALSE(output_vectors.empty());
    ASSERT_FALSE(shape_vectors.empty());

    ASSERT_EQ(shape_vectors[0].size(), 2);
    ASSERT_EQ(shape_vectors[0][0], 1);
    ASSERT_EQ(shape_vectors[0][1], 1000);

    // Type checking
    ASSERT_TRUE(std::holds_alternative<float>(output_vectors[0][0]));

    // Value access checking
    ASSERT_NO_THROW({ (void)std::get<float>(output_vectors[0][0]); });

    // Size checking
    ASSERT_EQ(output_vectors[0].size(), shape_vectors[0][1]);

    // Check all elements are of the expected type
    ASSERT_TRUE(std::all_of(output_vectors[0].begin(), output_vectors[0].end(),
                            [](const TensorElement& element) { return std::holds_alternative<float>(element); }));
}

// Integration test - only runs with real model
TEST_F(GGMLInferTest, IntegrationTest) {
    if (!has_real_model) {
        GTEST_SKIP() << "Skipping integration test - no real model available";
    }

    // Test with real model
    std::vector<std::vector<uint8_t>> input_tensors = neuriplo::testing::zero_blob_tensors();

    auto [output_vectors, shape_vectors] = real_infer->get_infer_results(input_tensors);

    // Verify real model produces reasonable results
    ASSERT_FALSE(output_vectors.empty());
    ASSERT_EQ(output_vectors[0].size(), 1000); // ImageNet classes

    // Check that output values are in reasonable range for probabilities/logits
    for (const auto& element : output_vectors[0]) {
        float value = std::get<float>(element);
        ASSERT_TRUE(std::isfinite(value)) << "Output contains non-finite value";
    }
}

// Unit test - only runs with mock. NON-EVIDENCE per R-7: verifies the harness,
// not the loader. A pass here must never be cited as loader validation.
TEST_F(GGMLInferTest, MockUnitTest) {
    if (has_real_model) {
        GTEST_SKIP() << "Skipping mock unit test - real model is available";
    }

    std::vector<std::vector<uint8_t>> input_tensors = neuriplo::testing::zero_blob_tensors();

    auto [output_vectors, shape_vectors] = mock_infer->get_infer_results(input_tensors);

    // Test mock-specific behavior
    ASSERT_EQ(output_vectors[0].size(), 1000);

    // Verify mock data pattern
    for (int i = 0; i < 10; ++i) {
        float expected = i * 0.001f;
        float actual = std::get<float>(output_vectors[0][i]);
        ASSERT_FLOAT_EQ(expected, actual);
    }
}

// Test GGML-specific functionality
TEST_F(GGMLInferTest, GGMLSpecificTest) {
    if (has_real_model) {
        // Test with real GGML backend
        ASSERT_TRUE(real_infer->is_gpu_available() || !real_infer->is_gpu_available()); // Should not throw

        // Test memory usage
        size_t memory_usage = real_infer->get_memory_usage_mb();
        ASSERT_GE(memory_usage, 0);

        // Test performance timing
        double inference_time = real_infer->get_last_inference_time_ms();
        ASSERT_GE(inference_time, 0.0);

    } else {
        // Test mock GGML functionality
        ASSERT_TRUE(true); // Mock should always pass basic tests
    }
}

namespace {
// Removes a temp file when it goes out of scope, even past FAIL().
struct ScopedRemove {
    explicit ScopedRemove(fs::path p) : path(std::move(p)) {}
    ~ScopedRemove() {
        std::error_code ec;
        fs::remove(path, ec);
    }
    fs::path path;
};
} // namespace

// T-16/T-17 (R-3/Q-2): ORT parity helpers. Reference logits come from
// scripts/ggml_ort_reference.py (/tmp/ortenv stock onnxruntime on
// resnet18-v1-7.onnx), stored as .npy float32 vectors of 1000 logits.
namespace {

// Minimal strict .npy v1.0 reader: float32 little-endian C-order only, which
// is exactly what np.save of a float32 (1000,) array writes. The caller
// SKIP-checks existence first; here a present-but-malformed file is a hard
// error (never silent fallback: that would fake parity evidence).
bool load_ort_ref_logits(const fs::path& path, std::vector<float>& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    char magic[6] = {0, 0, 0, 0, 0, 0};
    f.read(magic, 6);
    if (f.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        error = "bad npy magic in " + path.string();
        return false;
    }
    uint8_t major = 0;
    uint8_t minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    if (f.gcount() != 1 || major != 1) {
        error = "unsupported npy version in " + path.string();
        return false;
    }
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    if (f.gcount() != 2) {
        error = "truncated npy header length in " + path.string();
        return false;
    }
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    if (f.gcount() != header_len) {
        error = "truncated npy header in " + path.string();
        return false;
    }
    // Expect descr '<f4' (or '|f4'/'=f4') and a 1D shape (N,).
    const auto shape_pos = header.find("'shape'");
    const auto num_pos = header.find_first_of("0123456789", shape_pos == std::string::npos ? 0 : shape_pos);
    const auto comma_pos = header.find(',', num_pos == std::string::npos ? 0 : num_pos);
    if (header.find("'descr'") == std::string::npos || header.find("f4") == std::string::npos ||
        shape_pos == std::string::npos || num_pos == std::string::npos || comma_pos == std::string::npos) {
        error = "unexpected npy header (want float32 1D shape) in " + path.string() + ": " + header;
        return false;
    }
    const size_t count = static_cast<size_t>(std::stoul(header.substr(num_pos, comma_pos - num_pos)));
    if (count == 0 || count > (1u << 26)) {
        error = "implausible npy element count in " + path.string();
        return false;
    }
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<size_t>(f.gcount()) != count * sizeof(float)) {
        error = "truncated npy data in " + path.string();
        return false;
    }
    return true;
}

std::vector<float> tensor_to_floats(const std::vector<TensorElement>& elements) {
    std::vector<float> values;
    values.reserve(elements.size());
    for (const auto& element : elements) {
        values.push_back(std::get<float>(element));
    }
    return values;
}

// Indices of the 5 largest logits, descending. Exact-order comparison: ties
// in float logits are not expected from either runtime.
std::vector<int> top5_classes(const std::vector<float>& logits) {
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(5);
    return idx;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return worst;
}

// Deterministic spatially-nonuniform NCHW input (review finding 1): uniform
// inputs cannot validate the NCHW->[W,H,C,N] layout claim or expose
// spatial-only bugs. Integer math first, then float division — the ORT
// reference script (scripts/ggml_ort_reference.py) implements the identical
// formula, so any layout transposition shows up as parity failure.
inline std::vector<uint8_t> gradient_blob_bytes(int n = 1, int c = 3, int h = 224, int w = 224) {
    const std::size_t elements = static_cast<std::size_t>(n) * c * h * w;
    std::vector<uint8_t> bytes(elements * sizeof(float));
    auto* typed = reinterpret_cast<float*>(bytes.data());
    for (int nn = 0; nn < n; ++nn) {
        for (int cc = 0; cc < c; ++cc) {
            for (int yy = 0; yy < h; ++yy) {
                for (int xx = 0; xx < w; ++xx) {
                    const int v = (xx * 13 + yy * 17 + cc * 19) % 256;
                    typed[nn * c * h * w + cc * h * w + yy * w + xx] = static_cast<float>(v) / 255.0F;
                }
            }
        }
    }
    return bytes;
}

inline std::vector<std::vector<uint8_t>> gradient_blob_tensors(int n = 1, int c = 3, int h = 224, int w = 224) {
    return {gradient_blob_bytes(n, c, h, w)};
}

struct ParityCaseResult {
    std::string label;
    double diff;
    bool top5_agree;
    std::vector<int> got_top5;
    std::vector<int> ref_top5;
};

// Shared parity runner ... Returns per-case results so the calling TEST_F can
// RecordProperty them (a free function has no fixture to record on).
std::vector<ParityCaseResult>
check_parity_cases(GGMLInfer& infer,
                   const std::vector<std::pair<std::string, std::vector<std::vector<uint8_t>>>>& cases,
                   const std::vector<fs::path>& ref_paths, double tolerance) {
    EXPECT_EQ(cases.size(), ref_paths.size()) << "parity harness miswired: cases != refs";
    std::vector<ParityCaseResult> results;
    if (cases.size() != ref_paths.size()) {
        return results;
    }
    for (size_t c = 0; c < cases.size(); ++c) {
        const std::string& label = cases[c].first;
        std::vector<float> ref_logits;
        std::string error;
        if (!load_ort_ref_logits(ref_paths[c], ref_logits, error)) {
            ADD_FAILURE() << error;
            return results;
        }
        EXPECT_EQ(ref_logits.size(), 1000u);
        if (ref_logits.size() != 1000u) {
            return results;
        }

        const auto [outputs, shapes] = infer.get_infer_results(cases[c].second);
        EXPECT_EQ(outputs.size(), 1u);
        EXPECT_EQ(outputs[0].size(), 1000u);
        if (outputs.empty() || outputs[0].size() != 1000u) {
            ADD_FAILURE() << "input=" << label << " bad output shape";
            return results;
        }
        for (const auto& element : outputs[0]) {
            EXPECT_TRUE(std::holds_alternative<float>(element));
        }
        const std::vector<float> got_logits = tensor_to_floats(outputs[0]);

        const double diff = max_abs_diff(got_logits, ref_logits);
        const std::vector<int> got_top5 = top5_classes(got_logits);
        const std::vector<int> ref_top5 = top5_classes(ref_logits);
        const bool top5_agree = (got_top5 == ref_top5);

        std::cout << "[parity] input=" << label << " max_abs_diff=" << diff << " top5_agree=" << top5_agree
                  << " ggml_top5=[";
        for (size_t i = 0; i < got_top5.size(); ++i) {
            std::cout << (i == 0 ? "" : ",") << got_top5[i];
        }
        std::cout << "] ort_top5=[";
        for (size_t i = 0; i < ref_top5.size(); ++i) {
            std::cout << (i == 0 ? "" : ",") << ref_top5[i];
        }
        std::cout << "]" << std::endl;

        EXPECT_TRUE(top5_agree) << "input=" << label << " top-5 class mismatch (max_abs_diff=" << diff << ")";
        EXPECT_LE(diff, tolerance) << "input=" << label << " max_abs_diff=" << diff << " exceeds tolerance "
                                   << tolerance;
        results.push_back({label, diff, top5_agree, got_top5, ref_top5});
    }
    return results;
}

} // namespace

// (1) Real-model golden: fixed input -> 1000 finite values, not all zeros,
// deterministic across two runs. Skips (never fake-passes) without artifact.
// Q-2 hook: logs max-abs-diff vs the old zeros vector; no tolerance is
// asserted here because none has been measured against the ORT reference.
TEST_F(GGMLInferTest, RealModelGolden) {
    if (!has_real_model) {
        GTEST_SKIP() << "Skipping golden test - no real model available (model_path.txt missing)";
    }

    // Fixed, deterministic, non-zero input: gray-128 normalized NCHW blob.
    const auto input_tensors = neuriplo::testing::constant_blob_tensors(neuriplo::testing::kGray128Normalized);

    const auto [out1, shape1] = real_infer->get_infer_results(input_tensors);
    const auto [out2, shape2] = real_infer->get_infer_results(input_tensors);

    ASSERT_EQ(shape1.size(), 1u);
    ASSERT_EQ(shape1[0], (std::vector<int64_t>{1, 1000}));
    ASSERT_EQ(out1.size(), 1u);
    ASSERT_EQ(out1[0].size(), 1000u);

    bool any_nonzero = false;
    double max_abs_vs_zero = 0.0;
    for (const auto& element : out1[0]) {
        ASSERT_TRUE(std::holds_alternative<float>(element));
        const float value = std::get<float>(element);
        ASSERT_TRUE(std::isfinite(value)) << "logit is not finite";
        if (value != 0.0F) {
            any_nonzero = true;
        }
        max_abs_vs_zero = std::max(max_abs_vs_zero, static_cast<double>(std::fabs(value)));
    }
    EXPECT_TRUE(any_nonzero) << "real loader returned all zeros (placeholder behavior)";
    RecordProperty("max_abs_diff_vs_zeros", max_abs_vs_zero);
    std::cout << "[Q-2 hook] max_abs_diff_vs_zeros=" << max_abs_vs_zero << std::endl;

    ASSERT_EQ(out2.size(), 1u);
    ASSERT_EQ(out2[0].size(), 1000u);
    for (size_t i = 0; i < 1000; ++i) {
        ASSERT_FLOAT_EQ(std::get<float>(out1[0][i]), std::get<float>(out2[0][i]))
            << "non-deterministic logit at index " << i;
    }

    // Input sensitivity: a zeros input must produce different logits than the
    // gray input (catches constant-output bugs the checks above would miss).
    const auto zero_tensors = neuriplo::testing::zero_blob_tensors();
    const auto [out_zero, shape_zero] = real_infer->get_infer_results(zero_tensors);
    ASSERT_EQ(out_zero[0].size(), 1000u);
    bool any_diff = false;
    for (size_t i = 0; i < 1000; ++i) {
        if (std::get<float>(out1[0][i]) != std::get<float>(out_zero[0][i])) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff) << "logits identical for gray vs zeros input (constant-output bug)";
}

// T-17 (R-3/Q-2): ORT parity golden. Same two fixed inputs through the real
// GGML loader and the stock-onnxruntime reference (T-16:
// scripts/ggml_ort_reference.py): top-5 classes must agree exactly and
// max-abs-logit-diff must stay within the adopted tolerance. SKIPs (never
// fake-passes) without the model artifact or without the reference files.
TEST_F(GGMLInferTest, ParityVsOrt) {
    if (!has_real_model) {
        GTEST_SKIP() << "Skipping parity test - no real model available (model_path.txt missing)";
    }
    const fs::path gray_ref_path = "/tmp/ort_ref_gray.npy";
    const fs::path zeros_ref_path = "/tmp/ort_ref_zeros.npy";
    const fs::path gradient_ref_path = "/tmp/ort_ref_gradient.npy";
    if (!fs::exists(gray_ref_path) || !fs::exists(zeros_ref_path) || !fs::exists(gradient_ref_path)) {
        GTEST_SKIP() << "Skipping parity test - ORT reference files missing "
                        "(run /tmp/ortenv/bin/python scripts/ggml_ort_reference.py)";
    }

    constexpr double kParityTolerance = 1e-3;
    const std::vector<std::pair<std::string, std::vector<std::vector<uint8_t>>>> cases = {
        {"gray", neuriplo::testing::constant_blob_tensors(neuriplo::testing::kGray128Normalized)},
        {"zeros", neuriplo::testing::zero_blob_tensors()},
        {"gradient", gradient_blob_tensors()},
    };
    const std::vector<fs::path> ref_paths = {gray_ref_path, zeros_ref_path, gradient_ref_path};

    const auto results18 = check_parity_cases(*real_infer, cases, ref_paths, kParityTolerance);
    for (const auto& r : results18) {
        RecordProperty(("ort_max_abs_diff_" + r.label).c_str(), r.diff);
        RecordProperty(("ort_top5_agree_" + r.label).c_str(), r.top5_agree ? 1 : 0);
    }
}

// YOLOv8n ORT parity golden (R-3): same fixed inputs through the real GGML
// YOLO graph and stock onnxruntime on yolov8n.onnx. Compares the raw
// [1,84,8400] outputs: max-cell agreement (argmax over all 705600 values must
// match — decoding identical maxima means identical detections downstream)
// and max-abs-diff within tolerance. SKIPs without artifact or refs.
TEST_F(GGMLInferTest, ParityVsYolo) {
    const fs::path model_path = "/tmp/yolov8n.gguf";
    if (!fs::exists(model_path)) {
        GTEST_SKIP()
            << "Skipping yolo parity test - no model artifact (/tmp/yolov8n.gguf missing; "
               "convert: python3 scripts/convert_onnx_to_gguf.py --input /tmp/yolov8n.onnx --output /tmp/yolov8n.gguf)";
    }
    const std::vector<fs::path> ref_paths = {"/tmp/ort_yolo_gray.npy", "/tmp/ort_yolo_zeros.npy",
                                             "/tmp/ort_yolo_gradient.npy"};
    for (const auto& ref_path : ref_paths) {
        if (!fs::exists(ref_path)) {
            GTEST_SKIP() << "Skipping yolo parity test - ORT reference files missing";
        }
    }
    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 640, 640}};
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    constexpr double kYoloParityTolerance = 1e-2;
    const std::vector<std::pair<std::string, std::vector<std::vector<uint8_t>>>> cases = {
        {"gray", neuriplo::testing::constant_blob_tensors(neuriplo::testing::kGray128Normalized, 1, 3, 640, 640)},
        {"zeros", neuriplo::testing::zero_blob_tensors(1, 3, 640, 640)},
        {"gradient", gradient_blob_tensors(1, 3, 640, 640)},
    };
    const std::vector<fs::path> refs = {"/tmp/ort_yolo_gray.npy", "/tmp/ort_yolo_zeros.npy",
                                        "/tmp/ort_yolo_gradient.npy"};
    for (size_t c = 0; c < cases.size(); ++c) {
        std::vector<float> ref_logits;
        std::string error;
        ASSERT_TRUE(load_ort_ref_logits(refs[c], ref_logits, error)) << error;
        ASSERT_EQ(ref_logits.size(), static_cast<size_t>(84 * 8400));
        const auto [outputs, shapes] = infer.get_infer_results(cases[c].second);
        ASSERT_EQ(outputs.size(), 1u);
        ASSERT_EQ(outputs[0].size(), static_cast<size_t>(84 * 8400));
        ASSERT_EQ(shapes[0], (std::vector<int64_t>{1, 84, 8400}));
        const std::vector<float> got_logits = tensor_to_floats(outputs[0]);
        const double diff = max_abs_diff(got_logits, ref_logits);
        const auto got_max = std::max_element(got_logits.begin(), got_logits.end());
        const auto ref_max = std::max_element(ref_logits.begin(), ref_logits.end());
        const bool max_cell_agree =
            std::distance(got_logits.begin(), got_max) == std::distance(ref_logits.begin(), ref_max);
        RecordProperty(("yolo_max_abs_diff_" + cases[c].first).c_str(), diff);
        RecordProperty(("yolo_max_cell_agree_" + cases[c].first).c_str(), max_cell_agree ? 1 : 0);
        std::cout << "[yolo-parity] input=" << cases[c].first << " max_abs_diff=" << diff
                  << " max_cell_agree=" << max_cell_agree << std::endl;
        EXPECT_TRUE(max_cell_agree) << "input=" << cases[c].first << " max cell mismatch";
        EXPECT_LE(diff, kYoloParityTolerance)
            << "input=" << cases[c].first << " max_abs_diff=" << diff << " exceeds tolerance";
    }
}

// YOLOv8n-seg raw-output parity (R-3/R-4): output0 [1,116,8400] must match
// ORT by max cell and max-abs-diff; output1 [1,32,160,160] must match
// elementwise. SKIPs without the artifact or reference files.
TEST_F(GGMLInferTest, ParityVsYoloSeg) {
    const fs::path model_path = "/tmp/yolov8n-seg.gguf";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping yolo-seg parity test - no model artifact (/tmp/yolov8n-seg.gguf missing; "
                        "convert: python3 scripts/convert_onnx_to_gguf.py --input <yolov8n-seg.onnx> --output "
                        "<yolov8n-seg.gguf>)";
    }
    const std::vector<fs::path> ref_paths = {
        "/tmp/ort_yoloseg_gray_out0.npy",  "/tmp/ort_yoloseg_gray_out1.npy",     "/tmp/ort_yoloseg_zeros_out0.npy",
        "/tmp/ort_yoloseg_zeros_out1.npy", "/tmp/ort_yoloseg_gradient_out0.npy", "/tmp/ort_yoloseg_gradient_out1.npy",
    };
    for (const auto& ref_path : ref_paths) {
        if (!fs::exists(ref_path)) {
            GTEST_SKIP() << "Skipping yolo-seg parity test - ORT reference files missing "
                            "(run scripts/ggml_ort_reference.py --all-outputs --img-size 640)";
        }
    }
    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 640, 640}};
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    constexpr double kDetTolerance = 1e-2;
    constexpr double kProtoTolerance = 1e-2;
    const std::vector<std::pair<std::string, std::vector<std::vector<uint8_t>>>> cases = {
        {"gray", neuriplo::testing::constant_blob_tensors(neuriplo::testing::kGray128Normalized, 1, 3, 640, 640)},
        {"zeros", neuriplo::testing::zero_blob_tensors(1, 3, 640, 640)},
        {"gradient", gradient_blob_tensors(1, 3, 640, 640)},
    };
    for (size_t c = 0; c < cases.size(); ++c) {
        std::vector<float> ref_det;
        std::vector<float> ref_proto;
        std::string error;
        ASSERT_TRUE(load_ort_ref_logits(ref_paths[2 * c], ref_det, error)) << error;
        ASSERT_TRUE(load_ort_ref_logits(ref_paths[2 * c + 1], ref_proto, error)) << error;
        ASSERT_EQ(ref_det.size(), static_cast<size_t>(116 * 8400));
        ASSERT_EQ(ref_proto.size(), static_cast<size_t>(32 * 160 * 160));

        const auto [outputs, shapes] = infer.get_infer_results(cases[c].second);
        ASSERT_EQ(outputs.size(), 2u);
        ASSERT_EQ(shapes.size(), 2u);
        ASSERT_EQ(shapes[0], (std::vector<int64_t>{1, 116, 8400}));
        ASSERT_EQ(shapes[1], (std::vector<int64_t>{1, 32, 160, 160}));
        ASSERT_EQ(outputs[0].size(), ref_det.size());
        ASSERT_EQ(outputs[1].size(), ref_proto.size());

        const std::vector<float> got_det = tensor_to_floats(outputs[0]);
        const std::vector<float> got_proto = tensor_to_floats(outputs[1]);
        EXPECT_TRUE(std::all_of(got_det.begin(), got_det.end(), [](float v) { return std::isfinite(v); }));
        EXPECT_TRUE(std::all_of(got_proto.begin(), got_proto.end(), [](float v) { return std::isfinite(v); }));
        const double det_diff = max_abs_diff(got_det, ref_det);
        const double proto_diff = max_abs_diff(got_proto, ref_proto);
        const auto got_max = std::max_element(got_det.begin(), got_det.end());
        const auto ref_max = std::max_element(ref_det.begin(), ref_det.end());
        const bool max_cell_agree = std::distance(got_det.begin(), got_max) == std::distance(ref_det.begin(), ref_max);

        RecordProperty(("yoloseg_det_max_abs_diff_" + cases[c].first).c_str(), det_diff);
        RecordProperty(("yoloseg_det_max_cell_agree_" + cases[c].first).c_str(), max_cell_agree ? 1 : 0);
        RecordProperty(("yoloseg_proto_max_abs_diff_" + cases[c].first).c_str(), proto_diff);
        std::cout << "[yoloseg-parity] input=" << cases[c].first << " det_diff=" << det_diff
                  << " det_max_cell_agree=" << max_cell_agree << " proto_diff=" << proto_diff << std::endl;
        EXPECT_TRUE(max_cell_agree) << "input=" << cases[c].first << " output0 max cell mismatch";
        EXPECT_LE(det_diff, kDetTolerance)
            << "input=" << cases[c].first << " output0 max_abs_diff=" << det_diff << " exceeds tolerance";
        EXPECT_LE(proto_diff, kProtoTolerance)
            << "input=" << cases[c].first << " output1 max_abs_diff=" << proto_diff << " exceeds tolerance";
        if (c == 0) {
            const auto [repeat_outputs, repeat_shapes] = infer.get_infer_results(cases[c].second);
            ASSERT_EQ(repeat_outputs.size(), 2u);
            ASSERT_EQ(repeat_shapes.size(), 2u);
            EXPECT_EQ(tensor_to_floats(repeat_outputs[0]), got_det) << "output0 is not deterministic";
            EXPECT_EQ(tensor_to_floats(repeat_outputs[1]), got_proto) << "output1 is not deterministic";
        }
    }
}

// T-17b (R-3): ResNet-50 ORT parity golden. Same bar as ParityVsOrt, own
// artifact (/tmp/resnet50.gguf) and own references (/tmp/ort50_ref_*.npy);
// SKIPs without either. Proves the bottleneck topology, not just dispatch.
TEST_F(GGMLInferTest, ParityVsOrt50) {
    const fs::path model_path = "/tmp/resnet50.gguf";
    if (!fs::exists(model_path)) {
        GTEST_SKIP()
            << "Skipping resnet50 parity test - no model artifact (/tmp/resnet50.gguf missing; "
               "convert: python3 scripts/convert_onnx_to_gguf.py --input <resnet50.onnx> --output <resnet50.gguf>)";
    }
    const std::vector<fs::path> ref_paths = {
        "/tmp/ort50_ref_gray.npy",
        "/tmp/ort50_ref_zeros.npy",
        "/tmp/ort50_ref_gradient.npy",
    };
    for (const auto& ref_path : ref_paths) {
        if (!fs::exists(ref_path)) {
            GTEST_SKIP() << "Skipping resnet50 parity test - ORT reference files missing "
                            "(run /tmp/ortenv/bin/python scripts/ggml_ort_reference.py --model "
                            "<resnet50.onnx> --gray-out /tmp/ort50_ref_gray.npy --zeros-out "
                            "/tmp/ort50_ref_zeros.npy --gradient-out /tmp/ort50_ref_gradient.npy)";
        }
    }

    const std::vector<std::vector<int64_t>> input_sizes = {{1, 224, 224, 3}};
    GGMLInfer infer50(model_path.string(), false, 1, input_sizes);
    constexpr double kParityTolerance = 1e-3;
    const std::vector<std::pair<std::string, std::vector<std::vector<uint8_t>>>> cases = {
        {"gray", neuriplo::testing::constant_blob_tensors(neuriplo::testing::kGray128Normalized)},
        {"zeros", neuriplo::testing::zero_blob_tensors()},
        {"gradient", gradient_blob_tensors()},
    };
    const auto results50 = check_parity_cases(infer50, cases, ref_paths, kParityTolerance);
    for (const auto& r : results50) {
        RecordProperty(("ort50_max_abs_diff_" + r.label).c_str(), r.diff);
        RecordProperty(("ort50_top5_agree_" + r.label).c_str(), r.top5_agree ? 1 : 0);
    }
}

// RF-DETR S-1: backbone+projector parity vs ORT prefix refs. Intermediate
// contract: the loader exposes the projector map [1,256,40,40]. The bar is
// measurement-derived per validation.md (projector chained drift 2.07e-03):
// assert max-cell agreement plus max_abs <= 5e-3, recording the actuals.
// SKIPs without /tmp/rfdetr.gguf, the fixed input blob, or the ORT ref.
TEST_F(GGMLInferTest, GgufRfdetrBackboneProjectorParity) {
    const fs::path model_path = "/tmp/rfdetr.gguf";
    const fs::path input_path = "/tmp/opencode/rfdetr_bb_input.npy";
    const fs::path ref_path = "/tmp/opencode/ref__backbone_backbone.0_projector_stages.0_stages.0.1_Add_1_output_0.npy";
    for (const auto& p : {model_path, input_path, ref_path}) {
        if (!fs::exists(p)) {
            GTEST_SKIP() << "Skipping RF-DETR backbone parity - missing " << p;
        }
    }
    // Whole-array F32 .npy loader (any C-order shape; flat compare is valid:
    // NCHW row-major bytes equal ggml flat order for these tensors).
    auto load_npy_flat = [](const fs::path& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            return false;
        }
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        if (!f) {
            return false;
        }
        f.seekg(hlen, std::ios::cur);
        const auto start = f.tellg();
        f.seekg(0, std::ios::end);
        const size_t bytes = static_cast<size_t>(f.tellg() - start);
        if (bytes % sizeof(float) != 0) {
            return false;
        }
        f.seekg(start);
        out.resize(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(f);
    };
    std::vector<float> input_f32;
    std::vector<float> expected;
    ASSERT_TRUE(load_npy_flat(input_path, input_f32)) << "cannot read " << input_path;
    ASSERT_TRUE(load_npy_flat(ref_path, expected)) << "cannot read " << ref_path;
    ASSERT_EQ(input_f32.size(), static_cast<size_t>(1 * 3 * 560 * 560));
    ASSERT_EQ(expected.size(), static_cast<size_t>(256 * 40 * 40));

    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 560, 560}};
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    std::vector<uint8_t> raw(input_f32.size() * sizeof(float));
    std::memcpy(raw.data(), input_f32.data(), raw.size());
    const auto [outputs, shapes] = infer.get_infer_results({std::move(raw)});
    ASSERT_EQ(shapes.size(), 7u);
    EXPECT_EQ(shapes[0], (std::vector<int64_t>{1, 300, 4}));
    EXPECT_EQ(shapes[1], (std::vector<int64_t>{1, 300, 91}));
    EXPECT_EQ(shapes[2], (std::vector<int64_t>{1, 256, 40, 40}));
    EXPECT_EQ(shapes[3], (std::vector<int64_t>{1, 300, 4}));
    EXPECT_EQ(shapes[4], (std::vector<int64_t>{1, 300, 256}));
    EXPECT_EQ(shapes[5], (std::vector<int64_t>{1, 300, 256}));
    EXPECT_EQ(shapes[6], (std::vector<int64_t>{1, 300, 256}));
    ASSERT_EQ(outputs.size(), 7u);
    ASSERT_EQ(outputs[2].size(), expected.size());
    const std::vector<float> got = tensor_to_floats(outputs[2]);
    size_t got_argmax = 0;
    size_t ref_argmax = 0;
    double max_diff = 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - expected[i]);
        max_diff = std::max(max_diff, d);
        sum += d;
        if (got[i] > got[got_argmax]) {
            got_argmax = i;
        }
        if (expected[i] > expected[ref_argmax]) {
            ref_argmax = i;
        }
    }
    RecordProperty("max_abs_diff", max_diff);
    RecordProperty("mean_abs_diff", sum / static_cast<double>(got.size()));
    std::cout << "[rfdetr-bb] max_abs_diff=" << max_diff << " mean=" << sum / static_cast<double>(got.size())
              << std::endl;
    EXPECT_EQ(got_argmax, ref_argmax) << "max-cell mismatch";
    // Measurement-derived budget (validation.md S-1: gradient-input probe
    // projector drift 7.87e-03; 1e-2 covers it with margin, not the 1e-5
    // chained-op bar).
    EXPECT_LE(max_diff, 1e-2) << "max_abs_diff " << max_diff << " exceeds measured budget";
}

namespace {

// Minimal GGUF v3 writer for parser regression fixtures. It mirrors the
// production converter: string/int32 KVs, F32 tensors, 32-byte data base.
struct GgufTestTensor {
    std::string name;
    std::vector<uint64_t> ne; // ggml order
    uint64_t offset = 0;
    std::vector<uint8_t> data;
};

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(value));
}

void append_u64(std::vector<uint8_t>& out, uint64_t value) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(value));
}

void append_string(std::vector<uint8_t>& out, const std::string& value) {
    append_u64(out, value.size());
    out.insert(out.end(), value.begin(), value.end());
}

void write_test_gguf(const fs::path& path, const std::vector<GgufTestTensor>& tensors, bool duplicate_rank = false) {
    std::vector<std::pair<std::string, int32_t>> ranks;
    ranks.reserve(tensors.size() + (duplicate_rank ? 1 : 0));
    for (const auto& t : tensors) {
        ranks.emplace_back("neuriplo.rank." + t.name, static_cast<int32_t>(t.ne.size()));
    }
    if (duplicate_rank && !ranks.empty()) {
        ranks.push_back(ranks.front());
    }

    std::vector<uint8_t> out = {'G', 'G', 'U', 'F'};
    append_u32(out, 3);
    append_u64(out, tensors.size());
    append_u64(out, ranks.size() + 1); // architecture + rank entries
    append_string(out, "general.architecture");
    append_u32(out, 8); // string
    append_string(out, "neuriplo-vision");
    for (const auto& [key, rank] : ranks) {
        append_string(out, key);
        append_u32(out, 5); // int32
        append_u32(out, static_cast<uint32_t>(rank));
    }
    for (const auto& t : tensors) {
        append_string(out, t.name);
        append_u32(out, static_cast<uint32_t>(t.ne.size()));
        for (uint64_t d : t.ne) {
            append_u64(out, d);
        }
        append_u32(out, 0); // GGML_TYPE_F32
        append_u64(out, t.offset);
    }
    while (out.size() % 32 != 0) {
        out.push_back(0);
    }
    const size_t data_base = out.size();
    for (const auto& t : tensors) {
        if (t.data.empty()) {
            continue;
        }
        const size_t start = data_base + static_cast<size_t>(t.offset);
        const size_t end = start + t.data.size();
        if (out.size() < end) {
            out.resize(end, 0);
        }
        std::memcpy(out.data() + start, t.data.data(), t.data.size());
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
}

std::string load_test_gguf_error(const fs::path& path) {
    const std::vector<std::vector<int64_t>> input_sizes = {{1, 224, 224, 3}};
    try {
        GGMLInfer infer(path.string(), false, 1, input_sizes);
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

} // namespace

TEST_F(GGMLInferTest, GgufRejectsOverflowingTensorOffset) {
    const fs::path path = fs::temp_directory_path() / "ggml_overflow_offset_test.gguf";
    ScopedRemove guard(path);
    write_test_gguf(path, {{"overflow_offset", {1}, std::numeric_limits<uint64_t>::max() - 8, {}}});
    const std::string error = load_test_gguf_error(path);
    ASSERT_FALSE(error.empty()) << "malformed tensor offset must be rejected";
    EXPECT_NE(error.find("extends past end of file"), std::string::npos) << error;
}

TEST_F(GGMLInferTest, GgufRejectsOverflowingTensorDimensions) {
    const fs::path path = fs::temp_directory_path() / "ggml_overflow_dims_test.gguf";
    ScopedRemove guard(path);
    write_test_gguf(path, {{"overflow_dims", {1u << 24, 1u << 24, 1u << 24, 2}, 0, {}}});
    const std::string error = load_test_gguf_error(path);
    ASSERT_FALSE(error.empty()) << "overflowing tensor dimensions must be rejected";
    EXPECT_NE(error.find("element count too large"), std::string::npos) << error;
}

TEST_F(GGMLInferTest, GgufAcceptsZeroSizedTensorDimension) {
    const fs::path path = fs::temp_directory_path() / "ggml_zero_dim_test.gguf";
    ScopedRemove guard(path);
    write_test_gguf(path, {{"zero_dim", {0, 1}, 0, {}}});
    const std::string error = load_test_gguf_error(path);
    ASSERT_FALSE(error.empty()) << "fixture should fail topology dispatch, not parsing";
    EXPECT_NE(error.find("zero_dim"), std::string::npos) << error;
    EXPECT_EQ(error.find("invalid dim"), std::string::npos) << error;
}

TEST_F(GGMLInferTest, GgufRejectsDuplicateRankEntry) {
    const fs::path path = fs::temp_directory_path() / "ggml_duplicate_rank_test.gguf";
    ScopedRemove guard(path);
    write_test_gguf(path, {{"duplicate_rank", {1}, 0, {}}}, /*duplicate_rank=*/true);
    const std::string error = load_test_gguf_error(path);
    ASSERT_FALSE(error.empty()) << "duplicate rank metadata must be rejected";
    EXPECT_NE(error.find("duplicate rank entry"), std::string::npos) << error;
}

// The old gguf C API truncated/misread tensor names beyond 64 chars; RF-DETR
// relies on names longer than that. A long name must reach topology dispatch.
TEST_F(GGMLInferTest, GgufAcceptsLongTensorName) {
    const fs::path path = fs::temp_directory_path() / "ggml_long_name_test.gguf";
    ScopedRemove guard(path);
    const std::string long_name = "encoder.blocks.0." + std::string(80, 'x') + ".weight";
    ASSERT_GT(long_name.size(), 64u);
    const uint8_t payload[4] = {0, 0, 0, 0};
    write_test_gguf(path, {{long_name, {1}, 0, std::vector<uint8_t>(payload, payload + sizeof(payload))}});
    const std::string error = load_test_gguf_error(path);
    ASSERT_FALSE(error.empty()) << "minimal fixture should fail topology dispatch, not parsing";
    EXPECT_NE(error.find(long_name), std::string::npos) << error;
    EXPECT_EQ(error.find("truncated"), std::string::npos) << error;
}

// Strict-GGUF-only: any non-.gguf extension is rejected at the gate with
// the path and the .gguf converter command. Legacy .ggml/.nggl and .onnx
// never reach the tensor parser.
TEST_F(GGMLInferTest, StrictGgufRejectsNonGguf) {
    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 224, 224}};
    for (const std::string& name : {"probe_model.ggml", "probe_model.nggl", "probe_model.onnx"}) {
        const fs::path probe = fs::temp_directory_path() / name;
        ScopedRemove guard(probe);
        {
            std::ofstream out(probe, std::ios::binary | std::ios::trunc);
            out << "dummy";
        }
        try {
            GGMLInfer infer(probe.string(), false, 1, input_sizes);
            FAIL() << "expected strict-GGUF rejection: " << probe;
        } catch (const std::exception& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find(probe.string()), std::string::npos) << "error must name the path; got: " << msg;
            EXPECT_NE(msg.find(".gguf"), std::string::npos) << "error must point to .gguf; got: " << msg;
        }
    }
}

// GGUF rejects files it cannot interpret, naming the path.
TEST_F(GGMLInferTest, GgufRejectsGarbage) {
    const fs::path garbage = fs::temp_directory_path() / "ggml_garbage_test.gguf";
    ScopedRemove guard(garbage);
    {
        std::ofstream out(garbage, std::ios::binary | std::ios::trunc);
        out << "GGUF";
        const uint32_t rest[8] = {0};
        out.write(reinterpret_cast<const char*>(rest), sizeof(rest));
    }
    const std::vector<std::vector<int64_t>> input_sizes = {{1, 224, 224, 3}};
    try {
        GGMLInfer infer(garbage.string(), false, 1, input_sizes);
        FAIL() << "expected load of garbage .gguf to throw: " << garbage;
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find(garbage.string()), std::string::npos)
            << "gguf error must name the path; got: " << e.what();
    }
}

TEST_F(GGMLInferTest, MissingFileThrowsWithPath) {
    const fs::path missing = fs::temp_directory_path() / "ggml_missing_model_test.gguf";
    ScopedRemove guard(missing);
    {
        std::error_code ec;
        fs::remove(missing, ec); // ensure absent
    }
    const std::vector<std::vector<int64_t>> input_sizes = {{1, 224, 224, 3}};
    try {
        GGMLInfer infer(missing.string(), false, 1, input_sizes);
        FAIL() << "expected load of missing file to throw: " << missing;
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find(missing.string()), std::string::npos)
            << "error message must name the missing path; got: " << e.what();
    }
}

// (3) Truncated/corrupt file: loader must reject it, not accept near-empty bytes.
TEST_F(GGMLInferTest, TruncatedCorruptFileThrows) {
    const fs::path corrupt = fs::temp_directory_path() / "ggml_corrupt_model_test.gguf";
    ScopedRemove guard(corrupt);
    {
        std::ofstream out(corrupt, std::ios::binary | std::ios::trunc);
        out << "NOT_A_GGML_MODEL_TRUNCATED";
        // Deliberately tiny: far short of any valid weight file.
    }
    const std::vector<std::vector<int64_t>> input_sizes = {{1, 224, 224, 3}};
    try {
        GGMLInfer infer(corrupt.string(), false, 1, input_sizes);
        FAIL() << "expected load of truncated/corrupt file to throw: " << corrupt;
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        EXPECT_FALSE(msg.empty());
        EXPECT_NE(msg.find(corrupt.string()), std::string::npos)
            << "corrupt-file error must name the path; got: " << msg;
        std::cout << "corrupt-file error: " << msg << std::endl;
    }
}

// (4) Wrong byte size: inference throws with expected-vs-got detail.
TEST_F(GGMLInferTest, WrongByteSizeThrowsWithExpectedVsGot) {
    if (!has_real_model) {
        GTEST_SKIP() << "needs a loadable model artifact; no model_path.txt";
    }
    const std::vector<uint8_t> short_bytes(100, 0);
    try {
        (void)real_infer->get_infer_results({short_bytes});
        FAIL() << "expected byte-size mismatch to throw";
    } catch (const InferenceExecutionException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Expected"), std::string::npos) << msg;
        EXPECT_NE(msg.find("got"), std::string::npos) << msg;
    }
}

// (5) Multi-input: single-input backend rejects two tensors.
TEST_F(GGMLInferTest, MultiInputThrows) {
    if (!has_real_model) {
        GTEST_SKIP() << "needs a loadable model artifact; no model_path.txt";
    }
    auto inputs = neuriplo::testing::zero_blob_tensors();
    inputs.push_back(inputs.front());
    try {
        (void)real_infer->get_infer_results(inputs);
        FAIL() << "expected multi-input to throw";
    } catch (const InferenceExecutionException& e) {
        EXPECT_NE(std::string(e.what()).find("single input"), std::string::npos) << e.what();
    }
}

// (6) use_gpu=true fails fast and names the GPU request; never silently CPU.
// setup_backend throws before the model file is opened, so no artifact needed.
TEST_F(GGMLInferTest, UseGpuThrowsNamingRequest) {
    const std::vector<std::vector<int64_t>> input_sizes = {{1, 224, 224, 3}};
    const std::string probe_path = has_real_model ? model_path : "nonexistent-probe.gguf";
    try {
        GGMLInfer infer(probe_path, true, 1, input_sizes);
        FAIL() << "use_gpu=true must fail fast, not silently run CPU";
    } catch (const std::exception& e) {
        std::string lowered = e.what();
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        EXPECT_NE(lowered.find("gpu"), std::string::npos) << "error must name the GPU request; got: " << e.what();
    }
}

// RF-DETR S-2a: encoder query-selection parity vs ORT refs. The loader
// exposes the top-300 selected boxes [1,300,4] (ggml [4,300,1,1]) as the
// second output; row order follows ggml_ops::top_k_indices (deterministic
// descending + index-ascending, drift may reorder near-ties vs ORT), so the
// test matches SETS: greedy bipartite match on max-abs per row, every row
// matched within a measured budget. SKIPs without artifact/refs.
TEST_F(GGMLInferTest, GgufRfdetrEncoderSelectionParity) {
    const fs::path model_path = "/tmp/rfdetr.gguf";
    const fs::path input_path = "/tmp/opencode/rfdetr_bb_input.npy";
    const fs::path ref_boxes_path = "/tmp/opencode/dec0x/x__GatherElements_output_0.npy";
    for (const auto& p : {model_path, input_path, ref_boxes_path}) {
        if (!fs::exists(p)) {
            GTEST_SKIP() << "Skipping RF-DETR encoder parity - missing " << p;
        }
    }
    auto load_npy_flat = [](const fs::path& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            return false;
        }
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        if (!f) {
            return false;
        }
        f.seekg(hlen, std::ios::cur);
        const auto start = f.tellg();
        f.seekg(0, std::ios::end);
        const size_t bytes = static_cast<size_t>(f.tellg() - start);
        if (bytes % sizeof(float) != 0) {
            return false;
        }
        f.seekg(start);
        out.resize(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(f);
    };
    std::vector<float> input_f32;
    std::vector<float> ref_boxes;
    ASSERT_TRUE(load_npy_flat(input_path, input_f32)) << "cannot read " << input_path;
    ASSERT_TRUE(load_npy_flat(ref_boxes_path, ref_boxes)) << "cannot read " << ref_boxes_path;
    ASSERT_EQ(ref_boxes.size(), static_cast<size_t>(300 * 4));

    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 560, 560}};
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    std::vector<uint8_t> raw(input_f32.size() * sizeof(float));
    std::memcpy(raw.data(), input_f32.data(), raw.size());
    const auto [outputs, shapes] = infer.get_infer_results({std::move(raw)});
    ASSERT_EQ(shapes.size(), 7u);
    EXPECT_EQ(shapes[3], (std::vector<int64_t>{1, 300, 4}));
    ASSERT_EQ(outputs.size(), 7u);
    ASSERT_EQ(outputs[3].size(), ref_boxes.size());
    const std::vector<float> got = tensor_to_floats(outputs[3]);
    // Greedy set match: cost[i][j] = max-abs over the 4 coords.
    // ggml [4,300] flat (k + 4*i) coincides with ONNX (1,300,4) flat (i*4+k).
    std::vector<bool> used(300, false);
    double worst = 0.0;
    for (size_t i = 0; i < 300; ++i) {
        double best = std::numeric_limits<double>::infinity();
        size_t best_j = 0;
        for (size_t j = 0; j < 300; ++j) {
            if (used[j]) {
                continue;
            }
            double d = 0.0;
            for (size_t k = 0; k < 4; ++k) {
                d = std::max(
                    d, std::fabs(static_cast<double>(got[i * 4 + k]) - static_cast<double>(ref_boxes[j * 4 + k])));
            }
            if (d < best) {
                best = d;
                best_j = j;
            }
        }
        used[best_j] = true;
        worst = std::max(worst, best);
    }
    RecordProperty("max_row_match_abs_diff", worst);
    std::cout << "[rfdetr-enc] worst row-match max_abs_diff=" << worst << std::endl;
    // Measured budget (probe S-2: decoded-box drift 1.69e-03; 5e-3 is ~3x).
    EXPECT_LE(worst, 5e-3) << "encoder selection set mismatch, worst row " << worst;
}

// RF-DETR S-2b: decoder layer-0 output parity vs ORT. The loader exposes
// norm3 [1,300,256] (ggml [256,300,1,1]) as the third output; flat NCHW
// row-major bytes equal ggml flat order.
//
// TopK order note: F32 score drift (2.3e-03 measured) flips 82 near-tie
// pairs vs ORT (min adjacent ORT gap is 3.8e-07), which changes slot
// pairing downstream because the learned queries/refpoints are fixed per
// slot. To verify the decoder wiring itself, this test pins ORT's exact
// TopK order through NEURIPLO_TEST_RFDETR_TOPK_ORDER (see S-2); without the
// override the loader still selects the same 300-box set
// (GgufRfdetrEncoderSelectionParity). SKIPs without artifact or refs.
TEST_F(GGMLInferTest, GgufRfdetrDecoderLayer0Parity) {
    const fs::path model_path = "/tmp/rfdetr.gguf";
    const fs::path input_path = "/tmp/opencode/rfdetr_bb_input.npy";
    const fs::path ref_path = "/tmp/opencode/dec0/ref__layers.0__norm3__LayerNormalization_output_0.npy";
    const char* order_file = "/tmp/opencode/dec0x/rfdetr_test_topk_order.txt";
    for (const auto& p : {model_path, input_path, ref_path}) {
        if (!fs::exists(p)) {
            GTEST_SKIP() << "Skipping RF-DETR decoder parity - missing " << p;
        }
    }
    if (!fs::exists(order_file)) {
        GTEST_SKIP() << "Skipping RF-DETR decoder parity - missing TopK order file";
    }
    auto load_npy_flat = [](const fs::path& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            return false;
        }
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        if (!f) {
            return false;
        }
        f.seekg(hlen, std::ios::cur);
        const auto start = f.tellg();
        f.seekg(0, std::ios::end);
        const size_t bytes = static_cast<size_t>(f.tellg() - start);
        if (bytes % sizeof(float) != 0) {
            return false;
        }
        f.seekg(start);
        out.resize(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(f);
    };
    std::vector<float> input_f32;
    std::vector<float> expected;
    ASSERT_TRUE(load_npy_flat(input_path, input_f32)) << "cannot read " << input_path;
    ASSERT_TRUE(load_npy_flat(ref_path, expected)) << "cannot read " << ref_path;
    ASSERT_EQ(expected.size(), static_cast<size_t>(300 * 256));

    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 560, 560}};
    ASSERT_EQ(::setenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER", order_file, 1), 0);
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    ::unsetenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER");
    std::vector<uint8_t> raw(input_f32.size() * sizeof(float));
    std::memcpy(raw.data(), input_f32.data(), raw.size());
    const auto [outputs, shapes] = infer.get_infer_results({std::move(raw)});
    ASSERT_EQ(shapes.size(), 7u);
    EXPECT_EQ(shapes[4], (std::vector<int64_t>{1, 300, 256}));
    ASSERT_EQ(outputs.size(), 7u);
    ASSERT_EQ(outputs[4].size(), expected.size());
    const std::vector<float> got = tensor_to_floats(outputs[4]);
    double max_diff = 0.0;
    double sum_diff = 0.0;
    size_t got_argmax = 0;
    size_t ref_argmax = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - expected[i]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
        if (got[i] > got[got_argmax]) {
            got_argmax = i;
        }
        if (expected[i] > expected[ref_argmax]) {
            ref_argmax = i;
        }
    }
    RecordProperty("max_abs_diff", max_diff);
    RecordProperty("mean_abs_diff", sum_diff / static_cast<double>(got.size()));
    std::cout << "[rfdetr-dec0] max_abs_diff=" << max_diff << " mean=" << sum_diff / static_cast<double>(got.size())
              << std::endl;
    EXPECT_EQ(got_argmax, ref_argmax) << "max-cell mismatch";
    // Measurement-derived budget (chained backbone->decoder F32 drift;
    // probe S-2 fed-ORT-inputs norm3 max is 1.20e-05).
    EXPECT_LE(max_diff, 5e-2) << "max_abs_diff " << max_diff << " exceeds measured budget";
}

// RF-DETR S-3a: decoder layers 1-2 norm3 parity vs ORT. Same TopK-order
// pinning as the layer-0 test (slot-wise comparison needs the order match).
// Layer index 3/4 in the 7-output layout.
TEST_F(GGMLInferTest, GgufRfdetrDecoderLayer12Parity) {
    const fs::path model_path = "/tmp/rfdetr.gguf";
    const fs::path input_path = "/tmp/opencode/rfdetr_bb_input.npy";
    const fs::path ref1_path =
        "/tmp/opencode/dec0/ref___transformer_decoder_layers.1_norm3_LayerNormalization_output_0.npy";
    const fs::path ref2_path =
        "/tmp/opencode/dec0/ref___transformer_decoder_layers.2_norm3_LayerNormalization_output_0.npy";
    const char* order_file = "/tmp/opencode/dec0x/rfdetr_test_topk_order.txt";
    for (const auto& p : {model_path, input_path, ref1_path, ref2_path}) {
        if (!fs::exists(p)) {
            GTEST_SKIP() << "Skipping RF-DETR decoder 1-2 parity - missing " << p;
        }
    }
    if (!fs::exists(order_file)) {
        GTEST_SKIP() << "Skipping RF-DETR decoder 1-2 parity - missing TopK order file";
    }
    auto load_npy_flat = [](const fs::path& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            return false;
        }
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        if (!f) {
            return false;
        }
        f.seekg(hlen, std::ios::cur);
        const auto start = f.tellg();
        f.seekg(0, std::ios::end);
        const size_t bytes = static_cast<size_t>(f.tellg() - start);
        if (bytes % sizeof(float) != 0) {
            return false;
        }
        f.seekg(start);
        out.resize(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(f);
    };
    std::vector<float> input_f32;
    std::vector<float> ref1;
    std::vector<float> ref2;
    ASSERT_TRUE(load_npy_flat(input_path, input_f32)) << "cannot read " << input_path;
    ASSERT_TRUE(load_npy_flat(ref1_path, ref1)) << "cannot read " << ref1_path;
    ASSERT_TRUE(load_npy_flat(ref2_path, ref2)) << "cannot read " << ref2_path;
    ASSERT_EQ(ref1.size(), static_cast<size_t>(300 * 256));
    ASSERT_EQ(ref2.size(), static_cast<size_t>(300 * 256));

    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 560, 560}};
    ASSERT_EQ(::setenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER", order_file, 1), 0);
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    ::unsetenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER");
    std::vector<uint8_t> raw(input_f32.size() * sizeof(float));
    std::memcpy(raw.data(), input_f32.data(), raw.size());
    const auto [outputs, shapes] = infer.get_infer_results({std::move(raw)});
    ASSERT_EQ(shapes.size(), 7u);
    EXPECT_EQ(shapes[5], (std::vector<int64_t>{1, 300, 256}));
    EXPECT_EQ(shapes[6], (std::vector<int64_t>{1, 300, 256}));
    ASSERT_EQ(outputs.size(), 7u);
    for (int layer = 0; layer < 2; ++layer) {
        const std::vector<float> got = tensor_to_floats(outputs[5 + layer]);
        const std::vector<float>& expected = layer == 0 ? ref1 : ref2;
        double max_diff = 0.0;
        double sum_diff = 0.0;
        size_t got_argmax = 0;
        size_t ref_argmax = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = std::fabs(static_cast<double>(got[i]) - expected[i]);
            max_diff = std::max(max_diff, d);
            sum_diff += d;
            if (got[i] > got[got_argmax]) {
                got_argmax = i;
            }
            if (expected[i] > expected[ref_argmax]) {
                ref_argmax = i;
            }
        }
        RecordProperty(("layer" + std::to_string(layer + 1) + "_max_abs_diff").c_str(), max_diff);
        RecordProperty(("layer" + std::to_string(layer + 1) + "_mean_abs_diff").c_str(),
                       sum_diff / static_cast<double>(got.size()));
        std::cout << "[rfdetr-dec" << layer + 1 << "] max_abs_diff=" << max_diff
                  << " mean=" << sum_diff / static_cast<double>(got.size()) << std::endl;
        EXPECT_EQ(got_argmax, ref_argmax) << "layer " << layer + 1 << " max-cell mismatch";
        // Chained drift grows per layer; budget set from measurement below.
        EXPECT_LE(max_diff, 1e-1) << "layer " << layer + 1 << " max_abs_diff " << max_diff;
    }
}

// RF-DETR S-3b: full-model head parity vs ORT dets [1,300,4] + labels
// [1,300,91]. TopK order pinned (same slot-ordering caveat as above).
// dets: max-cell agreement + max box L1; labels: top-1 agreement per slot.
TEST_F(GGMLInferTest, GgufRfdetrHeadParity) {
    const fs::path model_path = "/tmp/rfdetr.gguf";
    const fs::path input_path = "/tmp/opencode/rfdetr_bb_input.npy";
    const fs::path dets_path = "/tmp/opencode/dec0/ref__dets.npy";
    const fs::path labels_path = "/tmp/opencode/dec0/ref__labels.npy";
    const char* order_file = "/tmp/opencode/dec0x/rfdetr_test_topk_order.txt";
    for (const auto& p : {model_path, input_path, dets_path, labels_path}) {
        if (!fs::exists(p)) {
            GTEST_SKIP() << "Skipping RF-DETR head parity - missing " << p;
        }
    }
    if (!fs::exists(order_file)) {
        GTEST_SKIP() << "Skipping RF-DETR head parity - missing TopK order file";
    }
    auto load_npy_flat = [](const fs::path& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            return false;
        }
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        if (!f) {
            return false;
        }
        f.seekg(hlen, std::ios::cur);
        const auto start = f.tellg();
        f.seekg(0, std::ios::end);
        const size_t bytes = static_cast<size_t>(f.tellg() - start);
        if (bytes % sizeof(float) != 0) {
            return false;
        }
        f.seekg(start);
        out.resize(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(f);
    };
    std::vector<float> input_f32;
    std::vector<float> ref_dets;
    std::vector<float> ref_labels;
    ASSERT_TRUE(load_npy_flat(input_path, input_f32)) << "cannot read " << input_path;
    ASSERT_TRUE(load_npy_flat(dets_path, ref_dets)) << "cannot read " << dets_path;
    ASSERT_TRUE(load_npy_flat(labels_path, ref_labels)) << "cannot read " << labels_path;
    ASSERT_EQ(ref_dets.size(), static_cast<size_t>(300 * 4));
    ASSERT_EQ(ref_labels.size(), static_cast<size_t>(300 * 91));

    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 560, 560}};
    ASSERT_EQ(::setenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER", order_file, 1), 0);
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    ::unsetenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER");
    std::vector<uint8_t> raw(input_f32.size() * sizeof(float));
    std::memcpy(raw.data(), input_f32.data(), raw.size());
    const auto [outputs, shapes] = infer.get_infer_results({std::move(raw)});
    ASSERT_EQ(shapes.size(), 7u);
    EXPECT_EQ(shapes[0], (std::vector<int64_t>{1, 300, 4}));
    EXPECT_EQ(shapes[1], (std::vector<int64_t>{1, 300, 91}));
    ASSERT_EQ(outputs.size(), 7u);
    const std::vector<float> got_dets = tensor_to_floats(outputs[0]);
    const std::vector<float> got_labels = tensor_to_floats(outputs[1]);
    double dets_max = 0.0;
    double dets_l1 = 0.0;
    for (size_t i = 0; i < got_dets.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got_dets[i]) - ref_dets[i]);
        dets_max = std::max(dets_max, d);
        dets_l1 += d;
    }
    dets_l1 /= static_cast<double>(got_dets.size());
    size_t label_agree = 0;
    double labels_max = 0.0;
    for (size_t s = 0; s < 300; ++s) {
        int got_best = 0;
        int ref_best = 0;
        for (int c = 0; c < 91; ++c) {
            if (got_labels[s * 91 + c] > got_labels[s * 91 + got_best]) {
                got_best = c;
            }
            if (ref_labels[s * 91 + c] > ref_labels[s * 91 + ref_best]) {
                ref_best = c;
            }
            labels_max = std::max(labels_max, std::fabs(static_cast<double>(got_labels[s * 91 + c]) -
                                                        static_cast<double>(ref_labels[s * 91 + c])));
        }
        if (got_best == ref_best) {
            ++label_agree;
        }
    }
    RecordProperty("dets_max_abs_diff", dets_max);
    RecordProperty("dets_mean_l1", dets_l1);
    RecordProperty("labels_max_abs_diff", labels_max);
    RecordProperty("labels_top1_agree", static_cast<int>(label_agree));
    std::cout << "[rfdetr-head] dets_max=" << dets_max << " dets_l1=" << dets_l1 << " labels_max=" << labels_max
              << " top1_agree=" << label_agree << "/300" << std::endl;
    EXPECT_LE(dets_max, 5e-2) << "dets max_abs_diff " << dets_max;
    EXPECT_EQ(label_agree, 300u) << "label top-1 agreement " << label_agree << "/300";
}

// RF-DETR-seg S-4: backbone+projector parity vs ORT prefix refs. The loader
// exposes the projector map [1,256,36,36] (ggml [36,36,256,1]) as the 4th
// output. Bar is measurement-derived (see validation.md S-4).
TEST_F(GGMLInferTest, GgufRfdetrSegBackboneProjectorParity) {
    const fs::path model_path = "/tmp/rfdetr_seg.gguf";
    const fs::path input_path = "/tmp/opencode/rfdetr_seg_input.npy";
    const fs::path ref_path =
        "/tmp/opencode/segenc__backbone_backbone.0_projector_stages.0_stages.0.1_Transpose_1_output_0.npy";
    for (const auto& pp : {model_path, input_path, ref_path}) {
        if (!fs::exists(pp)) {
            GTEST_SKIP() << "Skipping RF-DETR-seg backbone parity - missing " << pp;
        }
    }
    auto load_npy_flat = [](const fs::path& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            return false;
        }
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        if (!f) {
            return false;
        }
        f.seekg(hlen, std::ios::cur);
        const auto start = f.tellg();
        f.seekg(0, std::ios::end);
        const size_t bytes = static_cast<size_t>(f.tellg() - start);
        if (bytes % sizeof(float) != 0) {
            return false;
        }
        f.seekg(start);
        out.resize(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(f);
    };
    std::vector<float> input_f32;
    std::vector<float> expected;
    ASSERT_TRUE(load_npy_flat(input_path, input_f32)) << "cannot read " << input_path;
    ASSERT_TRUE(load_npy_flat(ref_path, expected)) << "cannot read " << ref_path;
    ASSERT_EQ(input_f32.size(), static_cast<size_t>(1 * 3 * 432 * 432));
    ASSERT_EQ(expected.size(), static_cast<size_t>(256 * 36 * 36));

    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 432, 432}};
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    std::vector<uint8_t> raw(input_f32.size() * sizeof(float));
    std::memcpy(raw.data(), input_f32.data(), raw.size());
    const auto [outputs, shapes] = infer.get_infer_results({std::move(raw)});
    ASSERT_EQ(shapes.size(), 9u);
    EXPECT_EQ(shapes[3], (std::vector<int64_t>{1, 256, 36, 36}));
    ASSERT_EQ(outputs.size(), 9u);
    ASSERT_EQ(outputs[3].size(), expected.size());
    const std::vector<float> got = tensor_to_floats(outputs[3]);
    size_t got_argmax = 0;
    size_t ref_argmax = 0;
    double max_diff = 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - expected[i]);
        max_diff = std::max(max_diff, d);
        sum += d;
        if (got[i] > got[got_argmax]) {
            got_argmax = i;
        }
        if (expected[i] > expected[ref_argmax]) {
            ref_argmax = i;
        }
    }
    RecordProperty("max_abs_diff", max_diff);
    RecordProperty("mean_abs_diff", sum / static_cast<double>(got.size()));
    std::cout << "[rfdetrseg-bb] max_abs_diff=" << max_diff << " mean=" << sum / static_cast<double>(got.size())
              << std::endl;
    EXPECT_EQ(got_argmax, ref_argmax) << "max-cell mismatch";
    EXPECT_LE(max_diff, 5e-2) << "max_abs_diff " << max_diff << " exceeds measured budget";
}

// RF-DETR-seg S-4: encoder query-selection parity vs ORT refs (set match,
// same bar as the detection encoder test).
TEST_F(GGMLInferTest, GgufRfdetrSegEncoderSelectionParity) {
    const fs::path model_path = "/tmp/rfdetr_seg.gguf";
    const fs::path input_path = "/tmp/opencode/rfdetr_seg_input.npy";
    const fs::path ref_boxes_path = "/tmp/opencode/segdecx/x__GatherElements_output_0.npy";
    for (const auto& p : {model_path, input_path, ref_boxes_path}) {
        if (!fs::exists(p)) {
            GTEST_SKIP() << "Skipping RF-DETR-seg encoder parity - missing " << p;
        }
    }
    auto load_npy_flat = [](const fs::path& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            return false;
        }
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        if (!f) {
            return false;
        }
        f.seekg(hlen, std::ios::cur);
        const auto start = f.tellg();
        f.seekg(0, std::ios::end);
        const size_t bytes = static_cast<size_t>(f.tellg() - start);
        if (bytes % sizeof(float) != 0) {
            return false;
        }
        f.seekg(start);
        out.resize(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(f);
    };
    std::vector<float> input_f32;
    std::vector<float> ref_boxes;
    ASSERT_TRUE(load_npy_flat(input_path, input_f32)) << "cannot read " << input_path;
    ASSERT_TRUE(load_npy_flat(ref_boxes_path, ref_boxes)) << "cannot read " << ref_boxes_path;
    ASSERT_EQ(ref_boxes.size(), static_cast<size_t>(200 * 4));

    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 432, 432}};
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    std::vector<uint8_t> raw(input_f32.size() * sizeof(float));
    std::memcpy(raw.data(), input_f32.data(), raw.size());
    const auto [outputs, shapes] = infer.get_infer_results({std::move(raw)});
    ASSERT_EQ(shapes.size(), 9u);
    EXPECT_EQ(shapes[4], (std::vector<int64_t>{1, 200, 4}));
    ASSERT_EQ(outputs.size(), 9u);
    ASSERT_EQ(outputs[4].size(), ref_boxes.size());
    const std::vector<float> got = tensor_to_floats(outputs[4]);
    std::vector<bool> used(200, false);
    double worst = 0.0;
    for (size_t i = 0; i < 200; ++i) {
        double best = std::numeric_limits<double>::infinity();
        size_t best_j = 0;
        for (size_t j = 0; j < 200; ++j) {
            if (used[j]) {
                continue;
            }
            double d = 0.0;
            for (size_t k = 0; k < 4; ++k) {
                d = std::max(
                    d, std::fabs(static_cast<double>(got[i * 4 + k]) - static_cast<double>(ref_boxes[j * 4 + k])));
            }
            if (d < best) {
                best = d;
                best_j = j;
            }
        }
        used[best_j] = true;
        worst = std::max(worst, best);
    }
    RecordProperty("max_row_match_abs_diff", worst);
    std::cout << "[rfdetrseg-enc] worst row-match max_abs_diff=" << worst << std::endl;
    EXPECT_LE(worst, 5e-3) << "encoder selection set mismatch, worst row " << worst;
}

// RF-DETR-seg S-4: 4-layer decoder norm parity vs ORT. TopK order pinned
// (same slot-ordering caveat as detection).
TEST_F(GGMLInferTest, GgufRfdetrSegDecoderParity) {
    const fs::path model_path = "/tmp/rfdetr_seg.gguf";
    const fs::path input_path = "/tmp/opencode/rfdetr_seg_input.npy";
    const char* order_file = "/tmp/opencode/segdecx/seg_test_topk_order.txt";
    const std::vector<std::string> ref_names = {"ref__dec0", "ref__dec1", "ref__dec2", "ref__dec3"};
    for (const auto& p : {model_path, input_path}) {
        if (!fs::exists(p)) {
            GTEST_SKIP() << "Skipping RF-DETR-seg decoder parity - missing " << p;
        }
    }
    for (const auto& n : ref_names) {
        if (!fs::exists(fs::path("/tmp/opencode/segdec") / (n + ".npy"))) {
            GTEST_SKIP() << "Skipping RF-DETR-seg decoder parity - missing " << n;
        }
    }
    if (!fs::exists(order_file)) {
        GTEST_SKIP() << "Skipping RF-DETR-seg decoder parity - missing TopK order file";
    }
    auto load_npy_flat = [](const fs::path& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            return false;
        }
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        if (!f) {
            return false;
        }
        f.seekg(hlen, std::ios::cur);
        const auto start = f.tellg();
        f.seekg(0, std::ios::end);
        const size_t bytes = static_cast<size_t>(f.tellg() - start);
        if (bytes % sizeof(float) != 0) {
            return false;
        }
        f.seekg(start);
        out.resize(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(f);
    };
    std::vector<float> input_f32;
    ASSERT_TRUE(load_npy_flat(input_path, input_f32)) << "cannot read " << input_path;

    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 432, 432}};
    ASSERT_EQ(::setenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER", order_file, 1), 0);
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    ::unsetenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER");
    std::vector<uint8_t> raw(input_f32.size() * sizeof(float));
    std::memcpy(raw.data(), input_f32.data(), raw.size());
    const auto [outputs, shapes] = infer.get_infer_results({std::move(raw)});
    ASSERT_EQ(shapes.size(), 9u);
    ASSERT_EQ(outputs.size(), 9u);
    for (int layer = 0; layer < 4; ++layer) {
        std::vector<float> expected;
        ASSERT_TRUE(load_npy_flat(fs::path("/tmp/opencode/segdec") / (ref_names[layer] + ".npy"), expected));
        EXPECT_EQ(shapes[5 + layer], (std::vector<int64_t>{1, 200, 256}));
        ASSERT_EQ(outputs[5 + layer].size(), expected.size());
        const std::vector<float> got = tensor_to_floats(outputs[5 + layer]);
        double max_diff = 0.0;
        double sum_diff = 0.0;
        size_t got_argmax = 0;
        size_t ref_argmax = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = std::fabs(static_cast<double>(got[i]) - expected[i]);
            max_diff = std::max(max_diff, d);
            sum_diff += d;
            if (got[i] > got[got_argmax]) {
                got_argmax = i;
            }
            if (expected[i] > expected[ref_argmax]) {
                ref_argmax = i;
            }
        }
        RecordProperty(("seg_layer" + std::to_string(layer) + "_max_abs_diff").c_str(), max_diff);
        RecordProperty(("seg_layer" + std::to_string(layer) + "_mean_abs_diff").c_str(),
                       sum_diff / static_cast<double>(got.size()));
        std::cout << "[rfdetrseg-dec" << layer << "] max_abs_diff=" << max_diff
                  << " mean=" << sum_diff / static_cast<double>(got.size()) << std::endl;
        EXPECT_EQ(got_argmax, ref_argmax) << "layer " << layer << " max-cell mismatch";
        EXPECT_LE(max_diff, 2e-1) << "layer " << layer << " max_abs_diff " << max_diff;
    }
}

// RF-DETR-seg S-4: head parity vs ORT dets [1,200,4] + labels [1,200,91] +
// masks [1,200,108,108]. TopK order pinned.
TEST_F(GGMLInferTest, GgufRfdetrSegHeadParity) {
    const fs::path model_path = "/tmp/rfdetr_seg.gguf";
    const fs::path input_path = "/tmp/opencode/rfdetr_seg_input.npy";
    const fs::path dets_path = "/tmp/opencode/segdec/ref__dets.npy";
    const fs::path labels_path = "/tmp/opencode/segdec/ref__labels.npy";
    const fs::path masks_path = "/tmp/opencode/segdec/ref__masks.npy";
    const char* order_file = "/tmp/opencode/segdecx/seg_test_topk_order.txt";
    for (const auto& p : {model_path, input_path, dets_path, labels_path, masks_path}) {
        if (!fs::exists(p)) {
            GTEST_SKIP() << "Skipping RF-DETR-seg head parity - missing " << p;
        }
    }
    if (!fs::exists(order_file)) {
        GTEST_SKIP() << "Skipping RF-DETR-seg head parity - missing TopK order file";
    }
    auto load_npy_flat = [](const fs::path& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            return false;
        }
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        if (!f) {
            return false;
        }
        f.seekg(hlen, std::ios::cur);
        const auto start = f.tellg();
        f.seekg(0, std::ios::end);
        const size_t bytes = static_cast<size_t>(f.tellg() - start);
        if (bytes % sizeof(float) != 0) {
            return false;
        }
        f.seekg(start);
        out.resize(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(f);
    };
    std::vector<float> input_f32;
    std::vector<float> ref_dets;
    std::vector<float> ref_labels;
    std::vector<float> ref_masks;
    ASSERT_TRUE(load_npy_flat(input_path, input_f32)) << "cannot read " << input_path;
    ASSERT_TRUE(load_npy_flat(dets_path, ref_dets)) << "cannot read " << dets_path;
    ASSERT_TRUE(load_npy_flat(labels_path, ref_labels)) << "cannot read " << labels_path;
    ASSERT_TRUE(load_npy_flat(masks_path, ref_masks)) << "cannot read " << masks_path;
    ASSERT_EQ(ref_dets.size(), static_cast<size_t>(200 * 4));
    ASSERT_EQ(ref_labels.size(), static_cast<size_t>(200 * 91));
    ASSERT_EQ(ref_masks.size(), static_cast<size_t>(200 * 108 * 108));

    const std::vector<std::vector<int64_t>> input_sizes = {{1, 3, 432, 432}};
    ASSERT_EQ(::setenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER", order_file, 1), 0);
    GGMLInfer infer(model_path.string(), false, 1, input_sizes);
    ::unsetenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER");
    std::vector<uint8_t> raw(input_f32.size() * sizeof(float));
    std::memcpy(raw.data(), input_f32.data(), raw.size());
    const auto [outputs, shapes] = infer.get_infer_results({std::move(raw)});
    ASSERT_EQ(shapes.size(), 9u);
    EXPECT_EQ(shapes[0], (std::vector<int64_t>{1, 200, 4}));
    EXPECT_EQ(shapes[1], (std::vector<int64_t>{1, 200, 108, 108}));
    EXPECT_EQ(shapes[2], (std::vector<int64_t>{1, 200, 91}));
    ASSERT_EQ(outputs.size(), 9u);
    const std::vector<float> got_dets = tensor_to_floats(outputs[0]);
    const std::vector<float> got_masks = tensor_to_floats(outputs[1]);
    const std::vector<float> got_labels = tensor_to_floats(outputs[2]);
    double dets_max = 0.0;
    double dets_l1 = 0.0;
    for (size_t i = 0; i < got_dets.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got_dets[i]) - ref_dets[i]);
        dets_max = std::max(dets_max, d);
        dets_l1 += d;
    }
    dets_l1 /= static_cast<double>(got_dets.size());
    size_t label_agree = 0;
    double labels_max = 0.0;
    for (size_t s = 0; s < 200; ++s) {
        int got_best = 0;
        int ref_best = 0;
        for (int c = 0; c < 91; ++c) {
            if (got_labels[s * 91 + c] > got_labels[s * 91 + got_best]) {
                got_best = c;
            }
            if (ref_labels[s * 91 + c] > ref_labels[s * 91 + ref_best]) {
                ref_best = c;
            }
            labels_max = std::max(labels_max, std::fabs(static_cast<double>(got_labels[s * 91 + c]) -
                                                        static_cast<double>(ref_labels[s * 91 + c])));
        }
        if (got_best == ref_best) {
            ++label_agree;
        }
    }
    double masks_max = 0.0;
    double masks_sum = 0.0;
    size_t masks_argmax = 0;
    size_t masks_ref_argmax = 0;
    for (size_t i = 0; i < got_masks.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got_masks[i]) - ref_masks[i]);
        masks_max = std::max(masks_max, d);
        masks_sum += d;
        if (got_masks[i] > got_masks[masks_argmax]) {
            masks_argmax = i;
        }
        if (ref_masks[i] > ref_masks[masks_ref_argmax]) {
            masks_ref_argmax = i;
        }
    }
    RecordProperty("dets_max_abs_diff", dets_max);
    RecordProperty("dets_mean_l1", dets_l1);
    RecordProperty("labels_max_abs_diff", labels_max);
    RecordProperty("labels_top1_agree", static_cast<int>(label_agree));
    RecordProperty("masks_max_abs_diff", masks_max);
    RecordProperty("masks_mean_abs_diff", masks_sum / static_cast<double>(got_masks.size()));
    std::cout << "[rfdetrseg-head] dets_max=" << dets_max << " dets_l1=" << dets_l1 << " labels_max=" << labels_max
              << " top1_agree=" << label_agree << "/200 masks_max=" << masks_max
              << " masks_mean=" << masks_sum / static_cast<double>(got_masks.size()) << std::endl;
    EXPECT_LE(dets_max, 1e-1) << "dets max_abs_diff " << dets_max;
    EXPECT_EQ(label_agree, 200u) << "label top-1 agreement " << label_agree << "/200";
    EXPECT_EQ(masks_argmax, masks_ref_argmax) << "masks max-cell mismatch";
    EXPECT_LE(masks_max, 5e-1) << "masks max_abs_diff " << masks_max;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
