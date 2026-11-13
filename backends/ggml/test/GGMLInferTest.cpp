#include "GGMLInfer.hpp"

#include "testing/TestBlob.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
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
    // Expect descr '<f4' (or '|f4'/'=f4') and shape (1000,).
    if (header.find("'descr'") == std::string::npos || header.find("f4") == std::string::npos ||
        header.find("1000") == std::string::npos) {
        error = "unexpected npy header (want float32 shape (1000,)) in " + path.string() + ": " + header;
        return false;
    }
    out.resize(1000);
    f.read(reinterpret_cast<char*>(out.data()), 1000 * sizeof(float));
    if (f.gcount() != 1000 * static_cast<std::streamsize>(sizeof(float))) {
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

inline std::vector<std::vector<uint8_t>> gradient_blob_tensors() { return {gradient_blob_bytes()}; }

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

// T-17b (R-3): ResNet-50 ORT parity golden. Same bar as ParityVsOrt, own
// artifact (/tmp/resnet50.ggml) and own references (/tmp/ort50_ref_*.npy);
// SKIPs without either. Proves the bottleneck topology, not just dispatch.
TEST_F(GGMLInferTest, ParityVsOrt50) {
    const fs::path model_path = "/tmp/resnet50.ggml";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping resnet50 parity test - no model artifact (/tmp/resnet50.ggml missing; "
                        "convert: python3 scripts/convert_onnx_to_ggml.py --input <resnet50.onnx>)";
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

// (2) Missing file: constructor throws and the message names the path.
TEST_F(GGMLInferTest, MissingFileThrowsWithPath) {
    const fs::path missing = fs::temp_directory_path() / "ggml_missing_model_test.ggml";
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
    const fs::path corrupt = fs::temp_directory_path() / "ggml_corrupt_model_test.ggml";
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
    const std::string probe_path = has_real_model ? model_path : "nonexistent-probe.ggml";
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
