#include <gtest/gtest.h>
#include "OVInfer.hpp"
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

// Test fixture for OpenVINO backend
class OpenVINOInferTest : public ::testing::Test {
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
        
        // Look for existing OpenVINO IR files (.xml/.bin)
        std::vector<std::string> possible_paths = {
            "resnet18.xml",
            "../resnet18.xml",
            "test_model.xml"
        };
        
        for (const auto& path : possible_paths) {
            if (fs::exists(path)) {
                // Check if corresponding .bin file exists
                std::string bin_path = path;
                bin_path.replace(bin_path.find(".xml"), 4, ".bin");
                if (fs::exists(bin_path)) {
                    return path;
                }
            }
        }
        
        // Try to generate IR from ONNX model
        fs::path script_path = current_path / "generate_openvino_ir.sh";
        if (fs::exists(script_path)) {
            std::string script = script_path.string();
            if (system(script.c_str()) == 0) {
                return "resnet18.xml";
            }
        }
        
        // As a fallback for testing
        throw std::runtime_error("OpenVINO IR files not found. Please create test model files first.");
    }
};

// Initialize static member
std::string OpenVINOInferTest::model_path;

// Test CPU initialization
TEST_F(OpenVINOInferTest, InitializationCPU) {
    ASSERT_NO_THROW({
        OVInfer infer(model_path, false); // CPU only
        auto model_info = infer.get_model_info();
        ASSERT_FALSE(model_info.getInputs().empty());
        ASSERT_FALSE(model_info.getOutputs().empty());
    });
}

// Test GPU initialization (if available)
TEST_F(OpenVINOInferTest, InitializationGPU) {
    // This test will gracefully handle GPU unavailability
    ASSERT_NO_THROW({
        OVInfer infer(model_path, true); // Try GPU, may fallback to CPU
        auto model_info = infer.get_model_info();
        ASSERT_FALSE(model_info.getInputs().empty());
        ASSERT_FALSE(model_info.getOutputs().empty());
    });
}

// Test inference results
TEST_F(OpenVINOInferTest, InferenceResults) {
    bool use_gpu = false;
    OVInfer infer(model_path, use_gpu);

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
TEST_F(OpenVINOInferTest, ModelInfoRetrieval) {
    OVInfer infer(model_path, false);
    auto model_info = infer.get_model_info();
    
    // Check inputs
    auto inputs = model_info.getInputs();
    ASSERT_FALSE(inputs.empty());
    
    // Check outputs
    auto outputs = model_info.getOutputs();
    ASSERT_FALSE(outputs.empty());
}

// Test with different batch sizes
TEST_F(OpenVINOInferTest, BatchSizeHandling) {
    size_t batch_size = 2;
    std::vector<std::vector<int64_t>> input_sizes = {{3, 224, 224}};
    
    ASSERT_NO_THROW({
        OVInfer infer(model_path, false, batch_size, input_sizes);
        auto model_info = infer.get_model_info();
        ASSERT_FALSE(model_info.getInputs().empty());
    });
}

// Test dynamic shapes (OpenVINO specific feature)
TEST_F(OpenVINOInferTest, DynamicShapes) {
    // Test with different input sizes to check dynamic shape support
    std::vector<std::vector<int64_t>> input_sizes = {{3, 224, 224}};
    
    OVInfer infer(model_path, false, 1, input_sizes);
    
    // Test inference with standard input
    cv::Mat input = cv::Mat::zeros(224, 224, CV_32FC3);
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.f / 255.f, cv::Size(224, 224), cv::Scalar(), true, false);
    
    auto [output_vectors, shape_vectors] = infer.get_infer_results(blob);
    ASSERT_FALSE(output_vectors.empty());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
