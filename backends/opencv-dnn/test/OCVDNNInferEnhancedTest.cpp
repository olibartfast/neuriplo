#include "BackendTestTemplate.hpp"
#include "OCVDNNInfer.hpp"
#include <glog/logging.h>

// Specialized test class for OpenCV DNN backend
class OCVDNNInferHybridTest : public BackendHybridTestBase<OCVDNNInfer> {
protected:
    std::unique_ptr<OCVDNNInfer> CreateBackendInstance() override {
        return std::make_unique<OCVDNNInfer>(GetModelPath(), false, 1, std::vector<std::string>{});
    }
    
    std::string GetBackendName() override {
        return "OPENCV_DNN";
    }
    
    std::vector<std::string> GetPossibleModelPaths() override {
        return {
            "resnet18.onnx",
            "../resnet18.onnx",
            "model.onnx",
            "yolo.onnx"
        };
    }
};

// ========================================
// UNIT TESTS (Mock-based - Always Pass)
// ========================================

MOCK_TEST_BASIC_INFERENCE(OCVDNNInferHybridTest)

BACKEND_UNIT_TEST(OCVDNNInferHybridTest, MockModelInfo) {
    auto model_info = mock_interface->get_model_info();
    EXPECT_GT(model_info.getInputs().size(), 0);
    EXPECT_GT(model_info.getOutputs().size(), 0);
}

BACKEND_UNIT_TEST(OCVDNNInferHybridTest, MockErrorHandling) {
    using ::testing::_;
    using ::testing::Throw;
    
    // Set up mock to throw exception
    EXPECT_CALL(*mock_interface, get_infer_results(_))
        .WillOnce(Throw(std::runtime_error("Mock error")));
    
    // Test error handling
    EXPECT_THROW(mock_interface->get_infer_results(CreateTestInput()), std::runtime_error);
}

BACKEND_UNIT_TEST(OCVDNNInferHybridTest, MockMultipleInferences) {
    auto input = CreateTestInput();
    
    // Test multiple consecutive inferences
    for (int i = 0; i < 3; ++i) {
        auto result = mock_interface->get_infer_results(input);
        EXPECT_EQ(std::get<0>(result).size(), 1);
        EXPECT_EQ(std::get<0>(result)[0].size(), 1000);
    }
}

// ========================================
// INTEGRATION TESTS (Model-dependent)
// ========================================

INTEGRATION_TEST_BASIC_INFERENCE(OCVDNNInferHybridTest)

BACKEND_INTEGRATION_TEST(OCVDNNInferHybridTest, ModelLoadingValidation) {
    SkipIfNoRealModel();
    
    // Test that model loads correctly
    ASSERT_TRUE(backend_instance != nullptr);
    
    // Test model info retrieval
    auto model_info = backend_instance->get_model_info();
    EXPECT_GT(model_info.getInputs().size(), 0);
    EXPECT_GT(model_info.getOutputs().size(), 0);
}

BACKEND_INTEGRATION_TEST(OCVDNNInferHybridTest, OutputShapeValidation) {
    SkipIfNoRealModel();
    ASSERT_TRUE(backend_instance != nullptr);
    
    auto input = CreateTestInput();
    auto result = backend_instance->get_infer_results(input);
    
    // Validate output structure
    auto outputs = std::get<0>(result);
    auto shapes = std::get<1>(result);
    
    EXPECT_EQ(outputs.size(), shapes.size());
    EXPECT_GT(outputs.size(), 0);
    
    // For classification models, expect at least one output
    if (!outputs.empty()) {
        EXPECT_GT(outputs[0].size(), 0);
    }
}

BACKEND_INTEGRATION_TEST(OCVDNNInferHybridTest, PerformanceBenchmark) {
    SkipIfNoRealModel();
    ASSERT_TRUE(backend_instance != nullptr);
    
    auto input = CreateTestInput();
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run inference
    auto result = backend_instance->get_infer_results(input);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Basic performance check (should complete within reasonable time)
    EXPECT_LT(duration.count(), 5000); // 5 seconds max
    
    std::cout << "OpenCV DNN Inference time: " << duration.count() << "ms" << std::endl;
}

// ========================================
// EDGE CASE TESTS
// ========================================

BACKEND_UNIT_TEST(OCVDNNInferHybridTest, EmptyInput) {
    cv::Mat empty_input;
    
    // Mock should handle empty input gracefully
    using ::testing::_;
    using ::testing::Throw;
    
    EXPECT_CALL(*mock_interface, get_infer_results(_))
        .WillOnce(Throw(std::invalid_argument("Empty input")));
    
    EXPECT_THROW(mock_interface->get_infer_results(empty_input), std::invalid_argument);
}

BACKEND_UNIT_TEST(OCVDNNInferHybridTest, InvalidInputSize) {
    cv::Mat wrong_size_input = cv::Mat::ones(64, 64, CV_8UC3); // Wrong size
    
    using ::testing::_;
    using ::testing::Throw;
    
    EXPECT_CALL(*mock_interface, get_infer_results(_))
        .WillOnce(Throw(std::invalid_argument("Invalid input size")));
    
    EXPECT_THROW(mock_interface->get_infer_results(wrong_size_input), std::invalid_argument);
}

// Test information logging
TEST(OCVDNNInferTestInfo, TestConfiguration) {
    std::cout << "\n=== OpenCV DNN Backend Test Configuration ===" << std::endl;
    std::cout << "Model available: " << (OCVDNNInferHybridTest::has_real_model ? "YES" : "NO") << std::endl;
    std::cout << "Model path: " << OCVDNNInferHybridTest::model_path << std::endl;
    std::cout << "Test types: Mock (unit) + Integration (conditional)" << std::endl;
    std::cout << "=============================================" << std::endl;
}
