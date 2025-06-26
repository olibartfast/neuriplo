#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "MockInferenceInterface.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <memory>

namespace fs = std::filesystem;

/**
 * Template base class for all backend hybrid tests.
 * Provides common functionality for model discovery, fallback logic, and test execution.
 */
template<typename BackendClass>
class BackendHybridTestBase : public AtomicBackendTest {
protected:
    static std::string model_path;
    static bool has_real_model;
    static bool model_discovery_done;
    std::unique_ptr<BackendClass> backend_instance;
    
    void SetUp() override {
        AtomicBackendTest::SetUp();
        
        if (!model_discovery_done) {
            DiscoverModel();
            model_discovery_done = true;
        }
        
        // Try to create backend instance if model is available
        if (has_real_model) {
            try {
                backend_instance = CreateBackendInstance();
            } catch (const std::exception& e) {
                // If real model fails, mark as unavailable and use mocks
                has_real_model = false;
                GTEST_LOG_(WARNING) << "Failed to create backend with real model: " << e.what();
            }
        }
    }
    
    void TearDown() override {
        backend_instance.reset();
        AtomicBackendTest::TearDown();
    }
    
    // Virtual method to be implemented by each backend
    virtual std::unique_ptr<BackendClass> CreateBackendInstance() = 0;
    virtual std::string GetBackendName() = 0;
    
private:
    void DiscoverModel() {
        // 1. Check for model path file (from downloader)
        std::ifstream modelPathFile("model_path.txt");
        if (modelPathFile) {
            std::string path;
            std::getline(modelPathFile, path);
            if (!path.empty() && fs::exists(path) && IsValidModel(path)) {
                model_path = path;
                has_real_model = true;
                return;
            }
        }
        
        // 2. Look for common model files
        std::vector<std::string> possible_models = GetPossibleModelPaths();
        for (const auto& path : possible_models) {
            if (fs::exists(path) && IsValidModel(path)) {
                model_path = path;
                has_real_model = true;
                return;
            }
        }
        
        // 3. Try to generate model
        if (system("bash generate_model.sh") == 0) {
            modelPathFile.close();
            modelPathFile.open("model_path.txt");
            if (modelPathFile) {
                std::string path;
                std::getline(modelPathFile, path);
                if (!path.empty() && fs::exists(path) && IsValidModel(path)) {
                    model_path = path;
                    has_real_model = true;
                    return;
                }
            }
        }
        
        // 4. Fallback to dummy model path (will be handled by mock tests)
        model_path = GetDefaultModelPath();
        has_real_model = false;
    }
    
    virtual std::vector<std::string> GetPossibleModelPaths() {
        return {
            "resnet18.onnx",
            "../resnet18.onnx",
            "model.onnx"
        };
    }
    
    virtual std::string GetDefaultModelPath() {
        return "resnet18.onnx";
    }
    
    bool IsValidModel(const std::string& path) {
        return fs::file_size(path) > 1000; // Basic size check
    }

protected:
    // Test utilities
    bool HasRealModel() const { return has_real_model; }
    const std::string& GetModelPath() const { return model_path; }
    
    // Skip integration test if no real model available
    void SkipIfNoRealModel() {
        if (!has_real_model) {
            GTEST_SKIP() << "Skipping integration test - no real model available";
        }
    }
    
    // Common test input creation
    cv::Mat CreateTestInput() {
        return cv::Mat::ones(224, 224, CV_8UC3) * 128; // Gray image
    }
};

// Static member definitions (to be specialized by each backend)
template<typename BackendClass>
std::string BackendHybridTestBase<BackendClass>::model_path = "";

template<typename BackendClass>
bool BackendHybridTestBase<BackendClass>::has_real_model = false;

template<typename BackendClass>
bool BackendHybridTestBase<BackendClass>::model_discovery_done = false;

// Utility macros for consistent test naming
#define BACKEND_UNIT_TEST(TestClass, TestName) \
    TEST_F(TestClass, Unit_##TestName)

#define BACKEND_INTEGRATION_TEST(TestClass, TestName) \
    TEST_F(TestClass, Integration_##TestName)

// Mock test that always passes
#define MOCK_TEST_BASIC_INFERENCE(TestClass) \
    BACKEND_UNIT_TEST(TestClass, MockBasicInference) { \
        auto result = mock_interface->get_infer_results(CreateTestInput()); \
        EXPECT_EQ(std::get<0>(result).size(), 1); \
        EXPECT_EQ(std::get<0>(result)[0].size(), 1000); \
    }

// Integration test with real model (conditional)
#define INTEGRATION_TEST_BASIC_INFERENCE(TestClass) \
    BACKEND_INTEGRATION_TEST(TestClass, RealModelInference) { \
        SkipIfNoRealModel(); \
        ASSERT_TRUE(backend_instance != nullptr); \
        auto result = backend_instance->get_infer_results(CreateTestInput()); \
        EXPECT_EQ(std::get<0>(result).size(), 1); \
        EXPECT_GT(std::get<0>(result)[0].size(), 0); \
    }
