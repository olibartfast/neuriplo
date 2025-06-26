#include <gtest/gtest.h>
#include "TFDetectionAPI.hpp"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// Mock logger for atomic testing
class MockLogger {
public:
    void info(const std::string& message) {
        std::cout << "INFO: " << message << std::endl;
    }
};

// Test fixture for TensorFlow backend
class TensorFlowInferTest : public ::testing::Test {
protected:
    std::shared_ptr<MockLogger> logger;
    static std::string model_path;

    void SetUp() override {
        logger = std::make_shared<MockLogger>();
        if (model_path.empty()) {
            model_path = GenerateModelPath();
        }
    }

    static std::string GenerateModelPath() {
        // For TensorFlow, we need a SavedModel format
        // First try to find if there's already a model available
        fs::path current_path = fs::current_path();
        
        // Look for existing SavedModel
        std::vector<std::string> possible_paths = {
            "test_model/saved_model",
            "../test_model/saved_model",
            "saved_model"
        };
        
        for (const auto& path : possible_paths) {
            if (fs::exists(path) && fs::is_directory(path)) {
                return path;
            }
        }
        
        // If no model found, try to generate one
        fs::path script_path = current_path / "generate_tf_model.py";
        if (fs::exists(script_path)) {
            std::string script = "python3 " + script_path.string();
            if (system(script.c_str()) == 0) {
                return "saved_model";
            }
        }
        
        // As a fallback, create a dummy path for testing
        throw std::runtime_error("TensorFlow SavedModel not found. Please create a test model first.");
    }
};

// Initialize static member
std::string TensorFlowInferTest::model_path;

// Test CPU initialization
TEST_F(TensorFlowInferTest, InitializationCPU) {
    ASSERT_NO_THROW({
        TFDetectionAPI infer(model_path, false); // CPU only
        auto model_info = infer.get_model_info();
        ASSERT_FALSE(model_info.getInputs().empty());
        ASSERT_FALSE(model_info.getOutputs().empty());
    });
}

// Test GPU initialization (if available) 
TEST_F(TensorFlowInferTest, InitializationGPU) {
    // TensorFlow GPU support is optional
    ASSERT_NO_THROW({
        TFDetectionAPI infer(model_path, true); // Try GPU
        auto model_info = infer.get_model_info();
        ASSERT_FALSE(model_info.getInputs().empty());
        ASSERT_FALSE(model_info.getOutputs().empty());
    });
}

// Test inference results
TEST_F(TensorFlowInferTest, InferenceResults) {
    bool use_gpu = false;
    TFDetectionAPI infer(model_path, use_gpu);

    // Create test input (typical image classification input)
    cv::Mat input = cv::Mat::zeros(224, 224, CV_32FC3);
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.f / 255.f, cv::Size(224, 224), cv::Scalar(), true, false);
    
    auto [output_vectors, shape_vectors] = infer.get_infer_results(blob);

    // Basic validation
    ASSERT_FALSE(output_vectors.empty());
    ASSERT_FALSE(shape_vectors.empty());

    // Type checking - ensure we have appropriate outputs
    ASSERT_TRUE(std::holds_alternative<float>(output_vectors[0][0]) || 
                std::holds_alternative<int32_t>(output_vectors[0][0]) ||
                std::holds_alternative<int64_t>(output_vectors[0][0]));
    
    // Size consistency check
    size_t expected_size = 1;
    for (auto dim : shape_vectors[0]) {
        expected_size *= dim;
    }
    ASSERT_EQ(output_vectors[0].size(), expected_size);
}

// Test model info retrieval
TEST_F(TensorFlowInferTest, ModelInfoRetrieval) {
    TFDetectionAPI infer(model_path, false);
    auto model_info = infer.get_model_info();
    
    // Check inputs
    auto inputs = model_info.getInputs();
    ASSERT_FALSE(inputs.empty());
    
    // Check outputs  
    auto outputs = model_info.getOutputs();
    ASSERT_FALSE(outputs.empty());
}

// Test with different batch sizes
TEST_F(TensorFlowInferTest, BatchSizeHandling) {
    size_t batch_size = 2;
    std::vector<std::vector<int64_t>> input_sizes = {{3, 224, 224}};
    
    ASSERT_NO_THROW({
        TFDetectionAPI infer(model_path, false, batch_size, input_sizes);
        auto model_info = infer.get_model_info();
        ASSERT_FALSE(model_info.getInputs().empty());
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
