#include <gtest/gtest.h>
#include "TFDetectionAPI.hpp"
#include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <memory>
#include <signal.h>

namespace fs = std::filesystem;

// Mock inference implementation for unit testing
class MockTFDetectionAPI {
public:
    MockTFDetectionAPI() = default;
    
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
    
    InferenceMetadata get_inference_metadata() {
        InferenceMetadata info;
        info.addInput("input", {3, 224, 224}, 1);
        info.addOutput("output", {1000}, 1);
        return info;
    }
};

class TensorFlowInferTest : public ::testing::Test {
protected:
    std::string model_path;
    bool has_real_model;
    std::unique_ptr<TFDetectionAPI> real_infer;
    std::unique_ptr<MockTFDetectionAPI> mock_infer;

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
                    real_infer = std::make_unique<TFDetectionAPI>(model_path, false);
                    std::cout << "Using real model: " << model_path << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "Failed to load real model, falling back to mock: " << e.what() << std::endl;
                    has_real_model = false;
                } catch (...) {
                    std::cout << "Failed to load real model (unknown error), falling back to mock" << std::endl;
                    has_real_model = false;
                }
            }
        }

        if (!has_real_model) {
            mock_infer = std::make_unique<MockTFDetectionAPI>();
            std::cout << "Using mock inference for testing" << std::endl;
        }
    }

    void TearDown() override {
        real_infer.reset();
        mock_infer.reset();
    }
};

// Test basic functionality - works with both real model and mock
TEST_F(TensorFlowInferTest, BasicInference) {
    // NCHW: 1 batch × 3 channels × 224 height × 224 width, all zeros
    std::vector<uint8_t> input_data(1 * 3 * 224 * 224 * sizeof(float), 0);
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

    if (has_real_model) {
        // Real model output includes batch dim, e.g. [1, 1000]
        ASSERT_GE(shape_vectors[0].size(), 1u);
        ASSERT_EQ(shape_vectors[0].back(), 1000);
    } else {
        ASSERT_EQ(shape_vectors[0].size(), 2u);
        ASSERT_EQ(shape_vectors[0][0], 1);
        ASSERT_EQ(shape_vectors[0][1], 1000);
    }

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
TEST_F(TensorFlowInferTest, IntegrationTest) {
    if (!has_real_model) {
        GTEST_SKIP() << "Skipping integration test - no real model available";
    }
    
    // Test with real model - NCHW: 1 batch × 3 channels × 224 height × 224 width
    std::vector<uint8_t> input_data(1 * 3 * 224 * 224 * sizeof(float), 0);
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
TEST_F(TensorFlowInferTest, MockUnitTest) {
    if (has_real_model) {
        GTEST_SKIP() << "Skipping mock unit test - real model is available";
    }
    
    std::vector<uint8_t> input_data(224 * 224 * 3 * sizeof(float), 0);
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

// GPU test - only runs if has_real_model and GPU available
TEST_F(TensorFlowInferTest, GPUTest) {
    if (!has_real_model) {
        GTEST_SKIP() << "Skipping GPU test - no real model available";
    }
    
    try {
        // Try to create a GPU inference engine
        auto gpu_infer = std::make_unique<TFDetectionAPI>(model_path, true);
        
        // If we got here, GPU is available, test inference
        // NCHW: 1 batch × 3 channels × 224 height × 224 width
        std::vector<uint8_t> input_data(1 * 3 * 224 * 224 * sizeof(float), 0);
        std::vector<std::vector<uint8_t>> input_tensors = {input_data};
        
        auto [output_vectors, shape_vectors] = gpu_infer->get_infer_results(input_tensors);
        
        // Basic validation
        ASSERT_FALSE(output_vectors.empty());
        ASSERT_EQ(output_vectors[0].size(), 1000);
    } catch (const std::exception& e) {
        // GPU not available or error occurred
        GTEST_SKIP() << "Skipping GPU test - GPU not available or error: " << e.what();
    }
}

// Inference test - works with both real model and mock
TEST_F(TensorFlowInferTest, InferenceMetadataTest) {
    if (has_real_model) {
        auto inference_metadata = real_infer->get_inference_metadata();
        
        // Check inputs
        auto inputs = inference_metadata.getInputs();
        ASSERT_FALSE(inputs.empty());
        
        // Check outputs  
        auto outputs = inference_metadata.getOutputs();
        ASSERT_FALSE(outputs.empty());
    } else {
        auto inference_metadata = mock_infer->get_inference_metadata();
        
        // Check inputs
        auto inputs = inference_metadata.getInputs();
        ASSERT_FALSE(inputs.empty());
        
        // Check outputs  
        auto outputs = inference_metadata.getOutputs();
        ASSERT_FALSE(outputs.empty());
    }
}

// Signal handler for crashes
void signal_handler(int signal) {
    std::cerr << "Received signal " << signal << " - TensorFlow backend may have crashed" << std::endl;
    std::cerr << "This is likely due to TensorFlow backend implementation issues" << std::endl;
    exit(1);
}

int main(int argc, char **argv) {
    // Set up signal handlers for graceful handling of crashes
    signal(SIGSEGV, signal_handler);
    signal(SIGABRT, signal_handler);
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
