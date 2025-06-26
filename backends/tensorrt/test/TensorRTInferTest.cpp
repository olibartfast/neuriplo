#include <gtest/gtest.h>
#include "TRTInfer.hpp"
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

// Test fixture for TensorRT backend
class TensorRTInferTest : public ::testing::Test {
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
        // Get the current working directory
        fs::path current_path = fs::current_path();
        
        // Look for existing TensorRT engine file
        std::vector<std::string> possible_paths = {
            "resnet18.engine",
            "../resnet18.engine",
            "test_model.engine"
        };
        
        for (const auto& path : possible_paths) {
            if (fs::exists(path)) {
                return path;
            }
        }
        
        // Try to generate engine from ONNX model
        fs::path script_path = current_path / "generate_trt_engine.sh";
        if (fs::exists(script_path)) {
            std::string script = script_path.string();
            if (system(script.c_str()) == 0) {
                return "resnet18.engine";
            }
        }
        
        // As a fallback for testing without actual engine
        throw std::runtime_error("TensorRT engine file not found. Please create a test engine first.");
    }
};

// Initialize static member
std::string TensorRTInferTest::model_path;

// Test GPU initialization (TensorRT requires GPU)
TEST_F(TensorRTInferTest, InitializationGPU) {
    ASSERT_NO_THROW({
        TRTInfer infer(model_path, true); // TensorRT always uses GPU
        auto model_info = infer.get_model_info();
        ASSERT_FALSE(model_info.getInputs().empty());
        ASSERT_FALSE(model_info.getOutputs().empty());
    });
}

// Test inference results
TEST_F(TensorRTInferTest, InferenceResults) {
    TRTInfer infer(model_path, true);

    // Create test input (ResNet-18 expects 224x224)
    cv::Mat input = cv::Mat::zeros(224, 224, CV_32FC3);
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.f / 255.f, cv::Size(224, 224), cv::Scalar(), true, false);
    
    auto [output_vectors, shape_vectors] = infer.get_infer_results(blob);

    // Basic validation
    ASSERT_FALSE(output_vectors.empty());
    ASSERT_FALSE(shape_vectors.empty());

    // Check shape (ResNet-18 classification output should be [1, 1000])
    ASSERT_EQ(shape_vectors[0].size(), 2);
    ASSERT_EQ(shape_vectors[0][0], 1);
    ASSERT_EQ(shape_vectors[0][1], 1000);

    // Type checking - ensure we have float outputs
    ASSERT_TRUE(std::holds_alternative<float>(output_vectors[0][0]));
    
    // Value access checking
    ASSERT_NO_THROW({
        float value = std::get<float>(output_vectors[0][0]);
    });
    
    // Size consistency check
    ASSERT_EQ(output_vectors[0].size(), static_cast<size_t>(shape_vectors[0][1]));
    
    // Check all elements are floats
    ASSERT_TRUE(std::all_of(output_vectors[0].begin(), output_vectors[0].end(), 
        [](const TensorElement& element) {
            return std::holds_alternative<float>(element);
        }));
}

// Test model info retrieval
TEST_F(TensorRTInferTest, ModelInfoRetrieval) {
    TRTInfer infer(model_path, true);
    auto model_info = infer.get_model_info();
    
    // Check inputs
    auto inputs = model_info.getInputs();
    ASSERT_FALSE(inputs.empty());
    
    // Check outputs
    auto outputs = model_info.getOutputs();
    ASSERT_FALSE(outputs.empty());
}

// Test with different batch sizes
TEST_F(TensorRTInferTest, BatchSizeHandling) {
    size_t batch_size = 2;
    std::vector<std::vector<int64_t>> input_sizes = {{3, 224, 224}};
    
    ASSERT_NO_THROW({
        TRTInfer infer(model_path, true, batch_size, input_sizes);
        auto model_info = infer.get_model_info();
        ASSERT_FALSE(model_info.getInputs().empty());
    });
}

// Test CUDA memory management
TEST_F(TensorRTInferTest, CudaMemoryManagement) {
    // This test ensures proper CUDA memory handling
    {
        TRTInfer infer(model_path, true);
        
        // Multiple inference calls to test memory management
        cv::Mat input = cv::Mat::zeros(224, 224, CV_32FC3);
        cv::Mat blob;
        cv::dnn::blobFromImage(input, blob, 1.f / 255.f, cv::Size(224, 224), cv::Scalar(), true, false);
        
        for (int i = 0; i < 3; ++i) {
            auto [output_vectors, shape_vectors] = infer.get_infer_results(blob);
            ASSERT_FALSE(output_vectors.empty());
        }
    }
    // Destructor should properly clean up CUDA resources
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
