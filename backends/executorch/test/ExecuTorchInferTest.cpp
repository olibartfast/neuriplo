#include "ExecuTorchInfer.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

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
    cv::Mat input = cv::Mat::zeros(224, 224, CV_32FC3);
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.f / 255.f, cv::Size(224, 224), cv::Scalar(), true, false);

    std::vector<uint8_t> input_data(blob.total() * blob.elemSize());
    std::memcpy(input_data.data(), blob.data, input_data.size());
    std::vector<std::vector<uint8_t>> input_tensors = {input_data};

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

    cv::Mat input = cv::Mat::zeros(224, 224, CV_32FC3);
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.f / 255.f, cv::Size(224, 224), cv::Scalar(), true, false);

    std::vector<uint8_t> input_data(blob.total() * blob.elemSize());
    std::memcpy(input_data.data(), blob.data, input_data.size());
    std::vector<std::vector<uint8_t>> input_tensors = {input_data};

    auto [output_vectors, shape_vectors] = real_infer->get_infer_results(input_tensors);

    ASSERT_FALSE(output_vectors.empty());
    ASSERT_FALSE(shape_vectors.empty());
    ASSERT_EQ(shape_vectors[0][0], 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
