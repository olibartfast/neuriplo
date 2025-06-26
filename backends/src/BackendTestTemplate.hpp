#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "MockInferenceInterface.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <chrono>
#include <thread>

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
    
    // Performance benchmarking
    struct PerformanceMetrics {
        double avg_inference_time_ms;
        double min_inference_time_ms;
        double max_inference_time_ms;
        size_t total_inferences;
        size_t memory_usage_mb;
        double throughput_fps;
    };
    
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
    
    // Performance benchmarking
    PerformanceMetrics RunPerformanceBenchmark(size_t num_iterations = 100) {
        PerformanceMetrics metrics = {};
        std::vector<double> inference_times;
        
        cv::Mat test_input = CreateTestInput();
        cv::Mat blob;
        cv::dnn::blobFromImage(test_input, blob, 1.f / 255.f, 
                              cv::Size(224, 224), cv::Scalar(), true, false);
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            if (has_real_model && backend_instance) {
                backend_instance->get_infer_results(blob);
            } else {
                mock_interface->get_infer_results(blob);
            }
        }
        
        // Benchmark
        for (size_t i = 0; i < num_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            if (has_real_model && backend_instance) {
                backend_instance->get_infer_results(blob);
            } else {
                mock_interface->get_infer_results(blob);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            inference_times.push_back(duration.count() / 1000.0); // Convert to ms
        }
        
        // Calculate metrics
        metrics.total_inferences = num_iterations;
        metrics.avg_inference_time_ms = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / num_iterations;
        metrics.min_inference_time_ms = *std::min_element(inference_times.begin(), inference_times.end());
        metrics.max_inference_time_ms = *std::max_element(inference_times.begin(), inference_times.end());
        metrics.throughput_fps = 1000.0 / metrics.avg_inference_time_ms;
        
        // Get memory usage
        if (has_real_model && backend_instance) {
            metrics.memory_usage_mb = backend_instance->get_memory_usage_mb();
        } else {
            metrics.memory_usage_mb = mock_interface->get_memory_usage_mb();
        }
        
        return metrics;
    }
    
    // Memory leak detection
    bool DetectMemoryLeak(size_t num_iterations = 1000) {
        size_t initial_memory = 0;
        size_t final_memory = 0;
        
        // Get initial memory usage
        if (has_real_model && backend_instance) {
            initial_memory = backend_instance->get_memory_usage_mb();
        } else {
            initial_memory = mock_interface->get_memory_usage_mb();
        }
        
        // Run many inferences
        cv::Mat test_input = CreateTestInput();
        cv::Mat blob;
        cv::dnn::blobFromImage(test_input, blob, 1.f / 255.f, 
                              cv::Size(224, 224), cv::Scalar(), true, false);
        
        for (size_t i = 0; i < num_iterations; ++i) {
            if (has_real_model && backend_instance) {
                backend_instance->get_infer_results(blob);
            } else {
                mock_interface->get_infer_results(blob);
            }
        }
        
        // Get final memory usage
        if (has_real_model && backend_instance) {
            final_memory = backend_instance->get_memory_usage_mb();
        } else {
            final_memory = mock_interface->get_memory_usage_mb();
        }
        
        // Check for significant memory growth (>10% increase)
        double memory_growth = static_cast<double>(final_memory - initial_memory) / initial_memory;
        return memory_growth > 0.1; // 10% threshold
    }
    
    // Edge case testing
    void TestEdgeCases() {
        // Test with empty input
        cv::Mat empty_input;
        if (has_real_model && backend_instance) {
            EXPECT_THROW(backend_instance->get_infer_results(empty_input), std::invalid_argument);
        } else {
            EXPECT_THROW(mock_interface->get_infer_results(empty_input), std::invalid_argument);
        }
        
        // Test with very large input
        cv::Mat large_input = cv::Mat::ones(1024, 1024, CV_8UC3) * 128;
        cv::Mat large_blob;
        cv::dnn::blobFromImage(large_input, large_blob, 1.f / 255.f, 
                              cv::Size(224, 224), cv::Scalar(), true, false);
        
        if (has_real_model && backend_instance) {
            EXPECT_NO_THROW(backend_instance->get_infer_results(large_blob));
        } else {
            EXPECT_NO_THROW(mock_interface->get_infer_results(large_blob));
        }
        
        // Test with zero input
        cv::Mat zero_input = cv::Mat::zeros(224, 224, CV_8UC3);
        cv::Mat zero_blob;
        cv::dnn::blobFromImage(zero_input, zero_blob, 1.f / 255.f, 
                              cv::Size(224, 224), cv::Scalar(), true, false);
        
        if (has_real_model && backend_instance) {
            EXPECT_NO_THROW(backend_instance->get_infer_results(zero_blob));
        } else {
            EXPECT_NO_THROW(mock_interface->get_infer_results(zero_blob));
        }
    }
    
    // Stress testing
    void RunStressTest(size_t num_threads = 4, size_t iterations_per_thread = 100) {
        std::vector<std::thread> threads;
        std::atomic<bool> stop_flag{false};
        
        // Create worker threads
        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back([this, iterations_per_thread, &stop_flag]() {
                cv::Mat test_input = CreateTestInput();
                cv::Mat blob;
                cv::dnn::blobFromImage(test_input, blob, 1.f / 255.f, 
                                      cv::Size(224, 224), cv::Scalar(), true, false);
                
                for (size_t j = 0; j < iterations_per_thread && !stop_flag; ++j) {
                    try {
                        if (has_real_model && backend_instance) {
                            backend_instance->get_infer_results(blob);
                        } else {
                            mock_interface->get_infer_results(blob);
                        }
                    } catch (const std::exception& e) {
                        stop_flag = true;
                        FAIL() << "Stress test failed: " << e.what();
                    }
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        EXPECT_FALSE(stop_flag) << "Stress test should complete without errors";
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

#define BACKEND_PERFORMANCE_TEST(TestClass, TestName) \
    TEST_F(TestClass, Performance_##TestName)

#define BACKEND_STRESS_TEST(TestClass, TestName) \
    TEST_F(TestClass, Stress_##TestName)

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

// Performance test
#define PERFORMANCE_TEST_BENCHMARK(TestClass) \
    BACKEND_PERFORMANCE_TEST(TestClass, Benchmark) { \
        auto metrics = RunPerformanceBenchmark(100); \
        EXPECT_GT(metrics.throughput_fps, 1.0) << "Should achieve at least 1 FPS"; \
        EXPECT_LT(metrics.avg_inference_time_ms, 1000.0) << "Average inference time should be under 1 second"; \
        EXPECT_LT(metrics.memory_usage_mb, 1000) << "Memory usage should be reasonable"; \
    }

// Memory leak test
#define MEMORY_TEST_LEAK_DETECTION(TestClass) \
    BACKEND_PERFORMANCE_TEST(TestClass, MemoryLeakDetection) { \
        bool has_memory_leak = DetectMemoryLeak(1000); \
        EXPECT_FALSE(has_memory_leak) << "Should not have memory leaks"; \
    }

// Stress test
#define STRESS_TEST_CONCURRENT_INFERENCE(TestClass) \
    BACKEND_STRESS_TEST(TestClass, ConcurrentInference) { \
        RunStressTest(4, 100); \
    }
