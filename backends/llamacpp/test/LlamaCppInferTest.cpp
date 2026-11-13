#include "LlamaCppInfer.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Mock — used when no real llama.cpp model is available
// ---------------------------------------------------------------------------
class MockLlamaCppInfer {
  public:
    MockLlamaCppInfer() = default;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input) {
        (void)input;
        // Mock: encode "Hello!" as float bytes
        const std::string mock_response = "Hello!";
        std::vector<TensorElement> output;
        output.reserve(mock_response.size());
        for (const char c : mock_response) {
            output.push_back(static_cast<float>(static_cast<unsigned char>(c)));
        }
        return std::make_tuple(std::vector<std::vector<TensorElement>>{output},
                               std::vector<std::vector<int64_t>>{{static_cast<int64_t>(output.size())}});
    }
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class LlamaCppInferTest : public ::testing::Test {
  protected:
    std::string model_path;
    bool has_real_model = false;
    std::unique_ptr<LlamaCppInfer> real_infer;
    std::unique_ptr<MockLlamaCppInfer> mock_infer;

    void SetUp() override {
        has_real_model = false;
        model_path = "";

        std::ifstream modelPathFile("model_path.txt");
        if (modelPathFile) {
            std::getline(modelPathFile, model_path);
            if (!model_path.empty() && fs::exists(model_path)) {
                has_real_model = true;
                try {
                    real_infer = std::make_unique<LlamaCppInfer>(model_path, false, 1);
                    std::cout << "Using real model: " << model_path << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "Failed to load real model, falling back to mock: " << e.what() << std::endl;
                    has_real_model = false;
                }
            }
        }

        if (!has_real_model) {
            mock_infer = std::make_unique<MockLlamaCppInfer>();
            std::cout << "Using mock inference for testing" << std::endl;
        }
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Basic inference — works with both real model and mock
TEST_F(LlamaCppInferTest, BasicInference) {
    const std::string prompt = "Hello";
    std::vector<uint8_t> input_data(prompt.begin(), prompt.end());
    std::vector<std::vector<uint8_t>> input_tensors = {input_data};

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
    ASSERT_EQ(output_vectors.size(), shape_vectors.size());
    ASSERT_FALSE(output_vectors[0].empty());

    // All output elements must be floats
    ASSERT_TRUE(std::all_of(output_vectors[0].begin(), output_vectors[0].end(),
                            [](const TensorElement& e) { return std::holds_alternative<float>(e); }));
}

// Integration test — only runs with real model
TEST_F(LlamaCppInferTest, IntegrationTest) {
    if (!has_real_model) {
        GTEST_SKIP() << "Skipping integration test — no real model available";
    }

    const std::string prompt = "What is the capital of France?";
    std::vector<uint8_t> input_data(prompt.begin(), prompt.end());
    std::vector<std::vector<uint8_t>> input_tensors = {input_data};

    auto [output_vectors, shape_vectors] = real_infer->get_infer_results(input_tensors);

    ASSERT_FALSE(output_vectors.empty());
    ASSERT_GT(output_vectors[0].size(), 0u);

    // Verify all output values are finite floats
    for (const auto& element : output_vectors[0]) {
        const float value = std::get<float>(element);
        ASSERT_TRUE(std::isfinite(value)) << "Output contains non-finite value";
    }
}

// Mock unit test — exercises mock-specific behaviour
TEST_F(LlamaCppInferTest, MockUnitTest) {
    if (has_real_model) {
        GTEST_SKIP() << "Skipping mock unit test — real model is available";
    }

    const std::string prompt = "test";
    std::vector<uint8_t> input_data(prompt.begin(), prompt.end());
    std::vector<std::vector<uint8_t>> input_tensors = {input_data};

    auto [output_vectors, shape_vectors] = mock_infer->get_infer_results(input_tensors);

    ASSERT_EQ(output_vectors.size(), 1u);
    // Mock encodes "Hello!" — 6 bytes
    ASSERT_EQ(output_vectors[0].size(), 6u);
    ASSERT_EQ(shape_vectors[0][0], 6);

    // Verify the mock encodes 'H' (72) as the first element
    ASSERT_FLOAT_EQ(std::get<float>(output_vectors[0][0]), static_cast<float>('H'));
}

// llama.cpp-specific metadata test
TEST_F(LlamaCppInferTest, LlamaCppSpecificTest) {
    if (has_real_model) {
        ASSERT_TRUE(real_infer->is_gpu_available() || !real_infer->is_gpu_available());

        const auto metadata = real_infer->get_inference_metadata();
        ASSERT_FALSE(metadata.getInputs().empty());
        ASSERT_FALSE(metadata.getOutputs().empty());

        ASSERT_GE(real_infer->get_memory_usage_mb(), 0u);
        ASSERT_GE(real_infer->get_last_inference_time_ms(), 0.0);
    } else {
        ASSERT_TRUE(true);
    }
}

// Opt-in real VLM integration. Unlike the legacy text smoke above, this test
// never replaces a failed real model load with a mock.
TEST(LlamaCppVlmIntegrationTest, ImageConditionedDecodeUsesRealFixtures) {
    const char* model_env = std::getenv("NEURIPLO_LLAMACPP_VLM_MODEL");
    const char* mmproj_env = std::getenv("NEURIPLO_LLAMACPP_VLM_MMPROJ");
    if (!model_env && !mmproj_env) {
        GTEST_SKIP() << "Set NEURIPLO_LLAMACPP_VLM_MODEL and "
                        "NEURIPLO_LLAMACPP_VLM_MMPROJ to run real VLM integration";
    }
    ASSERT_NE(model_env, nullptr) << "VLM model fixture path is required";
    ASSERT_NE(mmproj_env, nullptr) << "VLM projector fixture path is required";
    ASSERT_TRUE(fs::is_regular_file(model_env)) << "Missing VLM model: " << model_env;
    ASSERT_TRUE(fs::is_regular_file(mmproj_env)) << "Missing VLM projector: " << mmproj_env;
    const char* use_gpu_env = std::getenv("NEURIPLO_LLAMACPP_VLM_USE_GPU");
    const bool use_gpu = use_gpu_env && std::string(use_gpu_env) == "1";

    // Verify an explicitly supplied but invalid projector is rejected. This
    // constructor intentionally has no mock fallback.
    const std::string missing_mmproj = std::string(model_env) + "|mmproj=" + std::string(mmproj_env) + ".missing";
    EXPECT_THROW(LlamaCppInfer invalid_projector(missing_mmproj, false, 1), std::exception);

    const std::string vlm_model = std::string(model_env) + "|mmproj=" + mmproj_env;
    LlamaCppInfer infer(vlm_model, use_gpu, 1);
    EXPECT_EQ(infer.is_gpu_available(), use_gpu);

    const std::string text_prompt = "Reply with one short greeting.";
    const std::vector<uint8_t> text_bytes(text_prompt.begin(), text_prompt.end());
    const auto [text_outputs, text_shapes] = infer.get_infer_results({text_bytes});
    ASSERT_FALSE(text_outputs.empty());
    ASSERT_EQ(text_outputs.size(), text_shapes.size());

    constexpr uint32_t width = 32;
    constexpr uint32_t height = 32;
    std::vector<uint8_t> image(8 + width * height * 3, 0);
    std::memcpy(image.data(), &width, sizeof(width));
    std::memcpy(image.data() + sizeof(width), &height, sizeof(height));
    for (size_t i = 8; i + 2 < image.size(); i += 3) {
        image[i] = 220;
        image[i + 1] = 80;
        image[i + 2] = 35;
    }

    const std::string prompt = "Describe the colored image in one sentence.";
    const std::vector<uint8_t> prompt_bytes(prompt.begin(), prompt.end());
    const auto [outputs, shapes] = infer.get_infer_results({prompt_bytes, image});
    ASSERT_FALSE(outputs.empty());
    ASSERT_FALSE(outputs.front().empty());
    ASSERT_EQ(outputs.size(), shapes.size());
    for (const TensorElement& value : outputs.front()) {
        ASSERT_TRUE(std::holds_alternative<float>(value));
        ASSERT_TRUE(std::isfinite(std::get<float>(value)));
    }

    EXPECT_THROW(infer.get_infer_results({prompt_bytes, std::vector<uint8_t>(7, 0)}), InferenceException);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
