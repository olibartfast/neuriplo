#include "TRTInfer.hpp"
#include "testing/TestBlob.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
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

TEST_F(TensorRTInferTest, ReinitializationReplacesOwnedBuffers) {
    TRTInfer infer(model_path, true);
    ASSERT_NO_THROW(infer.initializeBuffers(model_path, {}));
    // Metadata must be rebuilt for the replacement engine, not left from the
    // constructor's load (and not wiped without being repopulated).
    const auto metadata = infer.get_inference_metadata();
    ASSERT_FALSE(metadata.getInputs().empty());
    ASSERT_FALSE(metadata.getOutputs().empty());
    const auto [outputs, shapes] = infer.get_infer_results(neuriplo::testing::zero_blob_tensors());
    ASSERT_FALSE(outputs.empty());
    ASSERT_FALSE(shapes.empty());
}

TEST_F(TensorRTInferTest, FailedReinitializationPreservesLiveEngine) {
    TRTInfer infer(model_path, true);
    ASSERT_EQ(infer.state(), BackendState::Ready);

    const auto metadata_before = infer.get_inference_metadata();
    ASSERT_FALSE(metadata_before.getInputs().empty());
    ASSERT_FALSE(metadata_before.getOutputs().empty());

    const fs::path missing_engine = fs::path(kTestBinaryDir) / "missing-reinitialization.engine";
    ASSERT_FALSE(fs::exists(missing_engine));
    EXPECT_THROW(infer.initializeBuffers(missing_engine.string(), {}), std::runtime_error);
    EXPECT_EQ(infer.state(), BackendState::Ready);

    // The rollback must restore the live engine's metadata, not leave the
    // staged-and-wiped InferenceMetadata behind.
    const auto metadata_after = infer.get_inference_metadata();
    ASSERT_EQ(metadata_after.getInputs().size(), metadata_before.getInputs().size());
    ASSERT_EQ(metadata_after.getOutputs().size(), metadata_before.getOutputs().size());
    for (size_t i = 0; i < metadata_before.getInputs().size(); ++i) {
        EXPECT_EQ(metadata_after.getInputs()[i].name, metadata_before.getInputs()[i].name);
        EXPECT_EQ(metadata_after.getInputs()[i].shape, metadata_before.getInputs()[i].shape);
    }
    for (size_t i = 0; i < metadata_before.getOutputs().size(); ++i) {
        EXPECT_EQ(metadata_after.getOutputs()[i].name, metadata_before.getOutputs()[i].name);
        EXPECT_EQ(metadata_after.getOutputs()[i].shape, metadata_before.getOutputs()[i].shape);
    }

    const auto [outputs, shapes] = infer.get_infer_results(neuriplo::testing::zero_blob_tensors());
    ASSERT_FALSE(outputs.empty());
    ASSERT_FALSE(shapes.empty());
}

TEST(TensorRTDynamicOutputs, EmptyAndChangingCountsThroughTypedAndRawApis) {
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    ASSERT_NE(builder, nullptr);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    ASSERT_NE(network, nullptr);
    ASSERT_NE(config, nullptr);
    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});
    ASSERT_NE(input, nullptr);
#if NV_TENSORRT_MAJOR > 10 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 14)
    auto* nonzero = network->addNonZero(*input, nvinfer1::DataType::kINT32);
#else
    auto* nonzero = network->addNonZero(*input);
#endif
    ASSERT_NE(nonzero, nullptr);
    auto* transpose = network->addShuffle(*nonzero->getOutput(0));
    ASSERT_NE(transpose, nullptr);
    nvinfer1::Permutation order{};
    order.order[0] = 1;
    order.order[1] = 0;
    transpose->setFirstTranspose(order);
    transpose->getOutput(0)->setName("indices");
    network->markOutput(*transpose->getOutput(0));
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    ASSERT_NE(plan, nullptr);
    const auto path = fs::path(kTestBinaryDir) / "nonzero-test.engine";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
        ASSERT_TRUE(file.good());
    }
    TRTInfer infer(path.string(), true);
    EXPECT_EQ(infer.get_inference_metadata().getOutputs().at(0).shape, (std::vector<int64_t>{-1, 1}));
    for (const int count : {0, 2, 4, 1, 0}) {
        std::array<float, 4> values{};
        for (int i = 0; i < count; ++i)
            values[static_cast<size_t>(i)] = 1.0f;
        std::vector<uint8_t> bytes(sizeof(values));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        const auto [outputs, shapes] = infer.get_infer_results({bytes});
        ASSERT_EQ(outputs.size(), 1U);
        ASSERT_EQ(shapes[0], (std::vector<int64_t>{count, 1}));
        ASSERT_EQ(outputs[0].size(), static_cast<size_t>(count));
        for (int i = 0; i < count; ++i)
            EXPECT_EQ(std::get<int32_t>(outputs[0][static_cast<size_t>(i)]), i);
        const auto raw = infer.get_infer_results_raw({bytes});
        ASSERT_EQ(raw.size(), 1U);
        EXPECT_EQ(raw[0].shape, shapes[0]);
        ASSERT_EQ(raw[0].bytes.size(), static_cast<size_t>(count) * sizeof(int32_t));
        for (int i = 0; i < count; ++i) {
            int32_t value;
            std::memcpy(&value, raw[0].bytes.data() + static_cast<size_t>(i) * sizeof(value), sizeof(value));
            EXPECT_EQ(value, i);
        }
    }
    EXPECT_NO_THROW(infer.createContextAndAllocateBuffers({}));
    EXPECT_NO_THROW(infer.get_infer_results_raw({std::vector<uint8_t>(4 * sizeof(float))}));
    EXPECT_THROW(infer.initializeBuffers("missing-dynamic-test.engine", {}), std::runtime_error);
    EXPECT_NO_THROW(infer.get_infer_results({std::vector<uint8_t>(4 * sizeof(float))}));
    EXPECT_NO_THROW(infer.initializeBuffers(path.string(), {}));
    EXPECT_NO_THROW(infer.get_infer_results_raw({std::vector<uint8_t>(4 * sizeof(float))}));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
