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
        
        // TEMPORARILY DISABLED: Look for existing SavedModel
        // std::vector<std::string> possible_paths = {
        //     "test_model/saved_model",
        //     "../test_model/saved_model",
        //     "saved_model",
        //     "backends/libtensorflow/test/saved_model"
        // };
        
        // for (const auto& path : possible_paths) {
        //     if (fs::exists(path) && fs::is_directory(path)) {
        //         std::cout << "Found existing model at: " << path << std::endl;
        //         return path;
        //     }
        // }
        
        // Try to use model downloader to get TensorFlow model
        fs::path model_downloader = current_path.parent_path().parent_path().parent_path() / "scripts" / "model_downloader.py";
        if (fs::exists(model_downloader)) {
            std::cout << "Using model downloader to get TensorFlow model: " << model_downloader << std::endl;
            // Source the TensorFlow environment and run the model downloader
            std::string script = "bash -c 'source /home/oli/dependencies/setup_env.sh && python3 " + model_downloader.string() + " LIBTENSORFLOW --output-dir .'";
            if (system(script.c_str()) == 0) {
                // Check if saved_model was created
                if (fs::exists("saved_model") && fs::is_directory("saved_model")) {
                    std::cout << "Model downloader generated SavedModel successfully at: saved_model" << std::endl;
                    return "saved_model";
                }
            }
            std::cout << "Model downloader failed" << std::endl;
        }
        
        // Try to use model downloader script
        fs::path model_downloader = current_path.parent_path().parent_path().parent_path() / "scripts" / "model_downloader.py";
        if (fs::exists(model_downloader)) {
            std::cout << "Using model downloader to generate TensorFlow model..." << std::endl;
            std::string script = "bash -c 'source /home/oli/dependencies/setup_env.sh && python3 " + 
                                model_downloader.string() + " LIBTENSORFLOW --output-dir " + 
                                current_path.string() + "'";
            if (system(script.c_str()) == 0) {
                // Check if model was created
                if (fs::exists("saved_model") && fs::is_directory("saved_model")) {
                    std::cout << "Model generated successfully using model downloader at: saved_model" << std::endl;
                    return "saved_model";
                }
            }
            std::cout << "Failed to generate model using model downloader" << std::endl;
        }
        
        // If no model found, try to generate one using model downloader
        fs::path downloader_script = current_path / "../../../../scripts/model_downloader.py";
        if (fs::exists(downloader_script)) {
            std::cout << "Generating TensorFlow model using model downloader: " << downloader_script << std::endl;
            // Source the TensorFlow environment and run the model downloader
            std::string script = "bash -c 'source /home/oli/dependencies/setup_env.sh && python3 " + downloader_script.string() + " LIBTENSORFLOW --output-dir .'";
            if (system(script.c_str()) == 0) {
                // Check if model was created
                if (fs::exists("saved_model") && fs::is_directory("saved_model")) {
                    std::cout << "Model generated successfully at: saved_model" << std::endl;
                    return "saved_model";
                }
            }
            std::cout << "Failed to generate model using model downloader" << std::endl;
        }
        
        // As a fallback, create a dummy path for testing
        std::cout << "No TensorFlow model found and could not generate one" << std::endl;
        std::cout << "WARNING: Using mock test mode - this only tests C++ integration, not actual inference" << std::endl;
        return "mock_model"; // Special string to indicate mock mode
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
