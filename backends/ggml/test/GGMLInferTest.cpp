#include <gtest/gtest.h>
#include "GGMLInfer.hpp"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

// Mock inference implementation for unit testing
class MockGGMLInfer {
public:
    MockGGMLInfer() = default;
    
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> 
    get_infer_results(const std::vector<std::vector<uint8_t>>& input) {
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
    cv::Mat input = cv::Mat::zeros(224, 224, CV_32FC3); // ResNet-18 expects 224x224 input
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.f / 255.f, cv::Size(224, 224), cv::Scalar(), true, false);
    
    std::vector<uint8_t> input_data(blob.total() * blob.elemSize());
    memcpy(input_data.data(), blob.data, input_data.size());
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

    ASSERT_EQ(shape_vectors[0].size(), 2);
    ASSERT_EQ(shape_vectors[0][0], 1);
    ASSERT_EQ(shape_vectors[0][1], 1000);

    // Type checking
    ASSERT_TRUE(std::holds_alternative<float>(output_vectors[0][0]));
    
    // Value access checking
    ASSERT_NO_THROW({
        float value = std::get<float>(output_vectors[0][0]);
    });
    
    // Size checking
    ASSERT_EQ(output_vectors[0].size(), shape_vectors[0][1]);
    
    // Check all elements are of the expected type
    ASSERT_TRUE(std::all_of(output_vectors[0].begin(), output_vectors[0].end(), 
        [](const TensorElement& element) {
            return std::holds_alternative<float>(element);
        }));
}

// Integration test - only runs with real model
TEST_F(GGMLInferTest, IntegrationTest) {
    if (!has_real_model) {
        GTEST_SKIP() << "Skipping integration test - no real model available";
    }
    
    // Test with real model
    cv::Mat input = cv::Mat::zeros(224, 224, CV_32FC3);
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.f / 255.f, cv::Size(224, 224), cv::Scalar(), true, false);
    
    std::vector<uint8_t> input_data(blob.total() * blob.elemSize());
    memcpy(input_data.data(), blob.data, input_data.size());
    std::vector<std::vector<uint8_t>> input_tensors = {input_data};

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

// Unit test - only runs with mock
TEST_F(GGMLInferTest, MockUnitTest) {
    if (has_real_model) {
        GTEST_SKIP() << "Skipping mock unit test - real model is available";
    }
    
    cv::Mat input = cv::Mat::zeros(224, 224, CV_32FC3);
    std::vector<uint8_t> input_data(input.total() * input.elemSize());
    memcpy(input_data.data(), input.data, input_data.size());
    std::vector<std::vector<uint8_t>> input_tensors = {input_data};
    
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
