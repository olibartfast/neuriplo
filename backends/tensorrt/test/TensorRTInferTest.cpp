#include "TRTInfer.hpp"
#include "testing/TestBlob.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

namespace fs = std::filesystem;

// Directory CMake builds this test into; also where generate_trt_engine.sh and
// the engine it produces live.
#ifndef TRT_TEST_BINARY_DIR
#define TRT_TEST_BINARY_DIR "."
#endif
constexpr const char* kTestBinaryDir = TRT_TEST_BINARY_DIR;

// Mock logger for atomic testing
class MockLogger {
  public:
    void info(const std::string& message) { std::cout << "INFO: " << message << std::endl; }
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
        if (model_path.empty()) {
            // A skipped test still exits 0, so a suite that skips everything reports
            // success having never touched the GPU. An environment that is meant to
            // exercise TensorRT sets NEURIPLO_REQUIRE_TENSORRT_TESTS=1 and gets a
            // failure instead of that false green.
            const char* required = std::getenv("NEURIPLO_REQUIRE_TENSORRT_TESTS");
            if (required != nullptr && std::string(required) != "0") {
                FAIL() << "NEURIPLO_REQUIRE_TENSORRT_TESTS is set, but no TensorRT engine could be found "
                          "or generated. Engine generation needs trtexec from the TensorRT installation "
                          "this build was configured against (-DTENSORRT_DIR).";
            }
            GTEST_SKIP() << "TensorRT engine file not found and scripted generation is unavailable on this platform";
        }
    }

    static std::string GenerateModelPath() {
        // Anchor on the directory CMake placed this binary and its generator script
        // in, so the engine is found no matter what the caller's working directory
        // is, and fall back to the working directory for hand-placed engines.
        const std::vector<fs::path> search_roots = {fs::path(kTestBinaryDir), fs::current_path(),
                                                    fs::current_path().parent_path()};
        const std::vector<std::string> names = {"resnet18.engine", "resnet18.plan", "test_model.engine",
                                                "test_model.plan"};
        for (const auto& name : names) {
            for (const auto& root : search_roots) {
                const fs::path candidate = root / name;
                if (fs::exists(candidate)) {
                    return candidate.string();
                }
            }
        }

        // Try to generate engine from ONNX model
#ifndef _WIN32
        const fs::path script_path = fs::path(kTestBinaryDir) / "generate_trt_engine.sh";
        if (fs::exists(script_path)) {
            // Run from the test directory so the engine lands beside the script.
            const std::string command =
                "cd \"" + std::string(kTestBinaryDir) + "\" && \"" + script_path.string() + "\"";
            if (std::system(command.c_str()) == 0) {
                const fs::path generated = fs::path(kTestBinaryDir) / "resnet18.engine";
                if (fs::exists(generated)) {
                    return generated.string();
                }
            }
        }
#endif

        return {};
    }
};

// Initialize static member
std::string TensorRTInferTest::model_path;

// Test GPU initialization (TensorRT requires GPU)
TEST_F(TensorRTInferTest, InitializationGPU) {
    ASSERT_NO_THROW({
        TRTInfer infer(model_path, true); // TensorRT always uses GPU
        std::cout << "TRTInfer object created successfully!" << std::endl;
    });
}

// Test inference results
TEST_F(TensorRTInferTest, InferenceResults) {
    TRTInfer infer(model_path, true);

    // Create test input (ResNet-18 expects 224x224)
    std::vector<std::vector<uint8_t>> input_tensors = neuriplo::testing::zero_blob_tensors();

    auto [output_vectors, shape_vectors] = infer.get_infer_results(input_tensors);

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
    ASSERT_NO_THROW({ (void)std::get<float>(output_vectors[0][0]); });

    // Size consistency check
    ASSERT_EQ(output_vectors[0].size(), static_cast<size_t>(shape_vectors[0][1]));

    // Check all elements are floats
    ASSERT_TRUE(std::all_of(output_vectors[0].begin(), output_vectors[0].end(),
                            [](const TensorElement& element) { return std::holds_alternative<float>(element); }));
}

// The crash this test was once disabled for was a dangling ILogger: the runtime
// was created with a stack-local Logger that died when initializeBuffers()
// returned, so the next TensorRT diagnostic called through a freed vtable.
// TRTInfer now uses a logger that outlives the runtime.
TEST_F(TensorRTInferTest, InferenceMetadataRetrieval) {
    TRTInfer infer(model_path, true);
    const auto inference_metadata = infer.get_inference_metadata();

    ASSERT_FALSE(inference_metadata.getInputs().empty());
    ASSERT_FALSE(inference_metadata.getOutputs().empty());
}

// Reported shapes must include the batch dimension. Without it the same model
// advertised [3,H,W] through TensorRT and [1,3,H,W] through ONNX Runtime, so a
// client that worked against one backend was rejected by the other.
TEST_F(TensorRTInferTest, MetadataShapesIncludeBatchDimension) {
    const size_t batch_size = 1;
    const std::vector<std::vector<int64_t>> input_sizes = {{3, 224, 224}};
    TRTInfer infer(model_path, true, batch_size, input_sizes);

    const auto inference_metadata = infer.get_inference_metadata();
    ASSERT_FALSE(inference_metadata.getInputs().empty());

    const auto& input = inference_metadata.getInputs().front();
    // 3-dimensional input sizes plus the restored batch dimension.
    EXPECT_EQ(input.shape.size(), input_sizes[0].size() + 1);
    EXPECT_EQ(input.shape.front(), static_cast<int64_t>(batch_size));

    for (const auto& output : inference_metadata.getOutputs()) {
        ASSERT_FALSE(output.shape.empty());
        EXPECT_EQ(output.shape.front(), static_cast<int64_t>(batch_size));
    }
}

// Test with different batch sizes
// A static-batch engine cannot honour a larger batch size. TensorRT rejects the
// requested shape, TRTInfer falls back to the engine's own shape, and the
// metadata must then advertise the batch the engine will actually run rather
// than the one that was asked for -- a client told "2" would send twice the
// data the engine consumes.
TEST_F(TensorRTInferTest, BatchSizeHandling) {
    const size_t requested_batch_size = 2;
    const std::vector<std::vector<int64_t>> input_sizes = {{3, 224, 224}};

    std::unique_ptr<TRTInfer> infer;
    ASSERT_NO_THROW({ infer = std::make_unique<TRTInfer>(model_path, true, requested_batch_size, input_sizes); });
    ASSERT_NE(infer, nullptr);

    const auto inference_metadata = infer->get_inference_metadata();
    ASSERT_FALSE(inference_metadata.getInputs().empty());

    const auto& input = inference_metadata.getInputs().front();
    ASSERT_FALSE(input.shape.empty());
    EXPECT_EQ(input.shape.front(), 1) << "the resnet18 test engine is built with a static batch of 1, so the "
                                         "unsupported request for "
                                      << requested_batch_size << " must not be reported back as satisfied";
}

// Test CUDA memory management
TEST_F(TensorRTInferTest, CudaMemoryManagement) {
    // This test ensures proper CUDA memory handling
    {
        TRTInfer infer(model_path, true);

        // Multiple inference calls to test memory management
        std::vector<std::vector<uint8_t>> input_tensors = neuriplo::testing::zero_blob_tensors();

        for (int i = 0; i < 3; ++i) {
            auto [output_vectors, shape_vectors] = infer.get_infer_results(input_tensors);
            ASSERT_FALSE(output_vectors.empty());
        }
    }
    // Destructor should properly clean up CUDA resources
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
