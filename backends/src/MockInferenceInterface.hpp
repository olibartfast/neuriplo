#pragma once

#include "InferenceInterface.hpp"
#include <opencv2/opencv.hpp>
#include <gmock/gmock.h>
#include <chrono>
#include <random>

/**
 * Mock implementation of InferenceInterface for unit testing.
 * This enables atomic and isolated testing of individual components.
 */
class MockInferenceInterface : public InferenceInterface {
public:
    MockInferenceInterface() : InferenceInterface("mock_model", false, 1, {}) {
        SetupDefaultExpectations();
        // Initialize performance tracking
        last_inference_time_ms_ = 0.0;
        total_inferences_ = 0;
        memory_usage_mb_ = 50; // Mock memory usage
    }
    
    // Mock the main inference method
    MOCK_METHOD(
        (std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>),
        get_infer_results,
        (const cv::Mat& input_blob),
        (override)
    );
    
    // Mock model info retrieval
    MOCK_METHOD(ModelInfo, get_model_info, (), (override));
    
    // Mock performance methods
    MOCK_METHOD(double, get_last_inference_time_ms, (), (const, override));
    MOCK_METHOD(size_t, get_total_inferences, (), (const, override));
    MOCK_METHOD(size_t, get_memory_usage_mb, (), (const, override));
    MOCK_METHOD(void, clear_cache, (), (override));
    
    // Helper method to set up common mock expectations
    void SetupDefaultExpectations() {
        // Default behavior for get_infer_results
        ON_CALL(*this, get_infer_results(testing::_))
            .WillByDefault(testing::Invoke([this](const cv::Mat& input_blob) {
                start_timer();
                auto result = CreateMockInferenceResult();
                end_timer();
                total_inferences_++;
                return result;
            }));
        
        // Default behavior for get_model_info
        ON_CALL(*this, get_model_info())
            .WillByDefault(testing::Return(CreateMockModelInfo()));
            
        // Default performance behavior
        ON_CALL(*this, get_last_inference_time_ms())
            .WillByDefault(testing::Return(5.0)); // Mock 5ms inference time
            
        ON_CALL(*this, get_total_inferences())
            .WillByDefault(testing::Return(total_inferences_));
            
        ON_CALL(*this, get_memory_usage_mb())
            .WillByDefault(testing::Return(memory_usage_mb_));
            
        ON_CALL(*this, clear_cache())
            .WillByDefault(testing::Invoke([this]() {
                memory_usage_mb_ = 10; // Reduced memory after cache clear
            }));
    }
    
    // Helper to create mock inference results
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> 
    CreateMockInferenceResult() {
        // Create mock output similar to ResNet-18 classification
        std::vector<std::vector<TensorElement>> outputs;
        std::vector<std::vector<int64_t>> shapes;
        
        // Mock classification output: 1000 classes
        std::vector<TensorElement> classification_output;
        for (int i = 0; i < 1000; ++i) {
            // Generate mock probabilities (small random values)
            classification_output.push_back(static_cast<float>(0.001f + (i % 10) * 0.0001f));
        }
        
        outputs.push_back(classification_output);
        shapes.push_back({1, 1000}); // Batch size 1, 1000 classes
        
        return std::make_tuple(outputs, shapes);
    }
    
    // Helper to create mock model info
    ModelInfo CreateMockModelInfo() {
        ModelInfo mock_info;
        
        // Add mock input
        std::vector<int64_t> input_shape = {3, 224, 224}; // CHW format
        mock_info.addInput("input", input_shape, 1);
        
        // Add mock output
        std::vector<int64_t> output_shape = {1000}; // Classification output
        mock_info.addOutput("output", output_shape, 1);
        
        return mock_info;
    }
    
    // Enhanced mock scenarios for comprehensive testing
    void SetupPerformanceTestExpectations() {
        using ::testing::_;
        using ::testing::Return;
        using ::testing::Invoke;
        
        // Simulate variable performance
        ON_CALL(*this, get_infer_results(_))
            .WillByDefault(Invoke([this](const cv::Mat& input_blob) {
                start_timer();
                
                // Simulate processing time based on input size
                size_t total_pixels = input_blob.total();
                std::this_thread::sleep_for(std::chrono::microseconds(total_pixels / 100));
                
                auto result = CreateMockInferenceResult();
                end_timer();
                total_inferences_++;
                return result;
            }));
    }
    
    void SetupMemoryLeakTestExpectations() {
        using ::testing::_;
        using ::testing::Invoke;
        
        // Simulate memory usage growth
        ON_CALL(*this, get_infer_results(_))
            .WillByDefault(Invoke([this](const cv::Mat& input_blob) {
                memory_usage_mb_ += 1; // Simulate memory growth
                return CreateMockInferenceResult();
            }));
    }
    
    void SetupErrorScenarios() {
        using ::testing::_;
        using ::testing::Return;
        using ::testing::Throw;
        
        // Normal inference
        EXPECT_CALL(*this, get_infer_results(_))
            .WillOnce(Return(CreateMockInferenceResult()));
        
        // Error scenario
        EXPECT_CALL(*this, get_infer_results(_))
            .WillOnce(Throw(InferenceExecutionException("Mock inference error")));
            
        // Memory error
        EXPECT_CALL(*this, get_infer_results(_))
            .WillOnce(Throw(std::bad_alloc()));
    }
    
    // Utility methods for testing
    void ResetPerformanceCounters() {
        last_inference_time_ms_ = 0.0;
        total_inferences_ = 0;
        memory_usage_mb_ = 50;
    }
    
    void SimulateMemoryLeak() {
        memory_usage_mb_ += 100; // Simulate significant memory leak
    }
};

/**
 * Test fixture for atomic backend testing.
 * Provides common setup and utilities for all backend tests.
 */
class AtomicBackendTest : public ::testing::Test {
protected:
    std::unique_ptr<MockInferenceInterface> mock_interface;
    cv::Mat test_input;
    
    void SetUp() override {
        // Create mock interface
        mock_interface = std::make_unique<MockInferenceInterface>();
        
        // Create standard test input
        test_input_ = cv::Mat::zeros(224, 224, CV_32FC3);
        cv::dnn::blobFromImage(test_input_, test_blob_, 1.f / 255.f, 
                              cv::Size(224, 224), cv::Scalar(), true, false);
    }
    
    void TearDown() override {
        // Cleanup if needed
        mock_interface.reset();
    }
    
    // Common test utilities
    cv::Mat test_input_;
    cv::Mat test_blob_;
    
    // Validate basic inference result structure
    void ValidateInferenceResult(
        const std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>& result) {
        
        auto [output_vectors, shape_vectors] = result;
        
        // Basic structure validation
        ASSERT_FALSE(output_vectors.empty()) << "Output vectors should not be empty";
        ASSERT_FALSE(shape_vectors.empty()) << "Shape vectors should not be empty";
        ASSERT_EQ(output_vectors.size(), shape_vectors.size()) << "Output and shape vector counts should match";
        
        // Validate each output
        for (size_t i = 0; i < output_vectors.size(); ++i) {
            ASSERT_FALSE(output_vectors[i].empty()) << "Output " << i << " should not be empty";
            
            // Calculate expected size from shape
            size_t expected_size = 1;
            for (auto dim : shape_vectors[i]) {
                ASSERT_GT(dim, 0) << "Shape dimensions should be positive";
                expected_size *= static_cast<size_t>(dim);
            }
            
            ASSERT_EQ(output_vectors[i].size(), expected_size) 
                << "Output " << i << " size should match shape";
        }
    }
    
    // Validate model info structure
    void ValidateModelInfo(const ModelInfo& model_info) {
        auto inputs = model_info.getInputs();
        auto outputs = model_info.getOutputs();
        
        ASSERT_FALSE(inputs.empty()) << "Model should have at least one input";
        ASSERT_FALSE(outputs.empty()) << "Model should have at least one output";
    }
    
    // Test tensor element types
    void ValidateTensorElementTypes(const std::vector<TensorElement>& tensor) {
        for (const auto& element : tensor) {
            // Ensure each element is one of the expected types
            ASSERT_TRUE(
                std::holds_alternative<float>(element) ||
                std::holds_alternative<int32_t>(element) ||
                std::holds_alternative<int64_t>(element)
            ) << "Tensor element should be of supported type";
        }
    }
    
    // Performance testing utilities
    void ValidatePerformanceMetrics(double max_time_ms = 100.0) {
        ASSERT_GT(mock_interface->get_total_inferences(), 0) << "Should have executed at least one inference";
        ASSERT_GT(mock_interface->get_last_inference_time_ms(), 0.0) << "Inference time should be positive";
        ASSERT_LT(mock_interface->get_last_inference_time_ms(), max_time_ms) << "Inference time should be reasonable";
    }
    
    void ValidateMemoryUsage(size_t max_memory_mb = 1000) {
        size_t memory_usage = mock_interface->get_memory_usage_mb();
        ASSERT_GT(memory_usage, 0) << "Memory usage should be positive";
        ASSERT_LT(memory_usage, max_memory_mb) << "Memory usage should be reasonable";
    }
    
    // Create various test inputs
    cv::Mat CreateSmallTestInput() {
        return cv::Mat::ones(64, 64, CV_8UC3) * 128;
    }
    
    cv::Mat CreateLargeTestInput() {
        return cv::Mat::ones(512, 512, CV_8UC3) * 128;
    }
    
    cv::Mat CreateInvalidTestInput() {
        return cv::Mat(); // Empty matrix
    }
};
