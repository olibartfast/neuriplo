#include "ExecuTorchInfer.hpp"

#include "testing/TestBlob.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

namespace fs = std::filesystem;

class MockExecuTorchInfer {
  public:
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input) {
        (void)input;
        std::vector<TensorElement> output_vector(1000);
        for (int i = 0; i < 1000; ++i) {
            output_vector[i] = static_cast<float>(i * 0.001f);
        }

        return {{{output_vector}}, {{1, 1000}}};
    }
};

class ExecuTorchInferTest : public ::testing::Test {
  protected:
    std::string model_path;
    bool has_real_model = false;
    std::unique_ptr<ExecuTorchInfer> real_infer;
    std::unique_ptr<MockExecuTorchInfer> mock_infer;

    void SetUp() override {
        std::ifstream model_path_file("model_path.txt");
        if (model_path_file) {
            std::getline(model_path_file, model_path);
            if (!model_path.empty() && fs::exists(model_path)) {
                try {
                    real_infer = std::make_unique<ExecuTorchInfer>(model_path, false, 1,
                                                                   std::vector<std::vector<int64_t>>{{1, 3, 224, 224}});
                    has_real_model = true;
                } catch (const std::exception& e) {
                    std::cout << "Failed to load ExecuTorch model, falling back to mock: " << e.what() << std::endl;
                }
            }
        }

        if (!has_real_model) {
            mock_infer = std::make_unique<MockExecuTorchInfer>();
        }
    }
};

TEST_F(ExecuTorchInferTest, BasicInference) {
    std::vector<std::vector<uint8_t>> input_tensors = neuriplo::testing::zero_blob_tensors();

    auto [output_vectors, shape_vectors] =
        has_real_model ? real_infer->get_infer_results(input_tensors) : mock_infer->get_infer_results(input_tensors);

    ASSERT_FALSE(output_vectors.empty());
    ASSERT_FALSE(shape_vectors.empty());
    ASSERT_EQ(shape_vectors[0].size(), 2);
    ASSERT_EQ(shape_vectors[0][0], 1);
    ASSERT_EQ(shape_vectors[0][1], 1000);
    ASSERT_EQ(output_vectors[0].size(), 1000);
    ASSERT_TRUE(std::holds_alternative<float>(output_vectors[0][0]));
}

TEST_F(ExecuTorchInferTest, IntegrationTest) {
    if (!has_real_model) {
        GTEST_SKIP() << "Skipping integration test - no real ExecuTorch model available";
    }

    std::vector<std::vector<uint8_t>> input_tensors = neuriplo::testing::zero_blob_tensors();

    auto [output_vectors, shape_vectors] = real_infer->get_infer_results(input_tensors);

    ASSERT_FALSE(output_vectors.empty());
    ASSERT_FALSE(shape_vectors.empty());
    ASSERT_EQ(shape_vectors[0][0], 1);
}

// The RF-DETR export leaves aten::linear.out outside its XNNPACK delegate.
// An explicitly supplied fixture must execute; it must never fall back to mocks.
TEST(ExecuTorchRegistration, RfDetrMatchesOnnxReference) {
    const char* fixture_env = std::getenv("NEURIPLO_EXECUTORCH_PARITY_FIXTURE");
    if (fixture_env == nullptr) {
        GTEST_SKIP() << "NEURIPLO_EXECUTORCH_PARITY_FIXTURE is not set";
    }
    const fs::path fixture(fixture_env);
    ExecuTorchInfer infer((fixture / "model.pte").string(), false);
    std::ifstream input(fixture / "input0.bin", std::ios::binary);
    ASSERT_TRUE(input.is_open());
    std::vector<uint8_t> bytes(std::istreambuf_iterator<char>{input}, {});
    const auto [outputs, shapes] = infer.get_infer_results({bytes});
    ASSERT_EQ(outputs.size(), 2U);
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::ifstream reference(fixture / ("output" + std::to_string(i) + ".bin"), std::ios::binary);
        ASSERT_TRUE(reference.is_open());
        for (size_t j = 0; j < outputs[i].size(); ++j) {
            float expected = 0;
            ASSERT_TRUE(reference.read(reinterpret_cast<char*>(&expected), sizeof(expected)));
            const float actual = std::get<float>(outputs[i][j]);
            ASSERT_TRUE(std::isfinite(actual));
            EXPECT_NEAR(actual, expected, 0.001f) << "output " << i << ", element " << j;
        }
        EXPECT_EQ(reference.peek(), std::ifstream::traits_type::eof());
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
