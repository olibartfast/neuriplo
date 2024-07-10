#include <gtest/gtest.h>
#include "LibtorchInfer.hpp"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <cstdlib> // For system function
#include <fstream> // For file I/O
#include <iostream> // For std::cout
#include <filesystem> // For filesystem operations

namespace fs = std::filesystem;

// Mock or a simple logger for the test environment
class MockLogger {
public:
    void info(const std::string& message) {
        std::cout << "INFO: " << message << std::endl;
    }
};

class LibtorchInferTest : public ::testing::Test {
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
        fs::path script_path = current_path / "generate_model.sh";

        // Call the bash script to generate the model and return the path
        std::string script = script_path.string();
        if (system(script.c_str()) != 0) {
            throw std::runtime_error("Failed to execute script: " + script);
        }
        
        // Assuming the script outputs the model path to a file
        std::ifstream modelPathFile("model_path.txt");
        if (!modelPathFile) {
            throw std::runtime_error("Failed to open model path file");
        }

        std::string model_path;
        std::getline(modelPathFile, model_path);
        return model_path;
    }
};

// Initialize the static member variable
std::string LibtorchInferTest::model_path;

TEST_F(LibtorchInferTest, InitializationCPU) {
    // Arrange
    bool use_gpu = false;

    // Act
    LibtorchInfer infer(model_path, use_gpu);
    
    // Assert
    // Add checks to verify the correct device is set and module is loaded
    // This requires access to private members or public getters in actual implementation
}

TEST_F(LibtorchInferTest, InitializationGPU) {
    // Arrange
    bool use_gpu = true;

    // Act
    LibtorchInfer infer(model_path, use_gpu);
    
    // Assert
    // Add checks to verify the correct device is set and module is loaded
    // This requires access to private members or public getters in actual implementation
}

TEST_F(LibtorchInferTest, InferenceResults) {
    // Arrange
    bool use_gpu = false; // Change to true if GPU is available and desired

    LibtorchInfer infer(model_path, use_gpu);

    cv::Mat input = cv::Mat::zeros(224, 224, CV_32FC3); // ResNet-18 expects 224x224 input
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.f / 255.f, cv::Size(224, 224), cv::Scalar(), true, false);
    auto [output_vectors, shape_vectors] = infer.get_infer_results(blob);

    ASSERT_FALSE(output_vectors.empty());
    ASSERT_FALSE(shape_vectors.empty());

    ASSERT_EQ(shape_vectors[0].size(), 2);
    ASSERT_EQ(shape_vectors[0][0], 1);
    ASSERT_EQ(shape_vectors[0][1], 1000);

    ASSERT_TRUE(std::any_cast<float>(&output_vectors[0][0]));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
