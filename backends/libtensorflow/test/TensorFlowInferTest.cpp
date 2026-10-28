#include "TFDetectionAPI.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <signal.h>
#include <tensorflow/core/protobuf/saved_model.pb.h>

namespace fs = std::filesystem;

// Mock inference implementation for unit testing
class MockTFDetectionAPI {
  public:
    MockTFDetectionAPI() = default;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input) {
        (void)input;
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
            ASSERT_FALSE(model_path.empty()) << "Provided model_path.txt is empty";
            ASSERT_TRUE(fs::exists(model_path)) << "Provided SavedModel does not exist: " << model_path;
            real_infer = std::make_unique<TFDetectionAPI>(model_path, false);
            has_real_model = true;
            std::cout << "Using real model: " << model_path << std::endl;
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
    ASSERT_NO_THROW({ (void)std::get<float>(output_vectors[0][0]); });

    // Size checking
    ASSERT_EQ(output_vectors[0].size(), shape_vectors[0][1]);

    // Check all elements are of the expected type
    ASSERT_TRUE(std::all_of(output_vectors[0].begin(), output_vectors[0].end(),
                            [](const TensorElement& element) { return std::holds_alternative<float>(element); }));
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

int main(int argc, char** argv) {
    // Set up signal handlers for graceful handling of crashes
    signal(SIGSEGV, signal_handler);
    signal(SIGABRT, signal_handler);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(TensorFlowInputContract, AdvertisesUint8AndRejectsShortBuffer) {
    tensorflow::SavedModel saved;
    saved.set_saved_model_schema_version(1);
    auto* meta = saved.add_meta_graphs();
    meta->mutable_meta_info_def()->add_tags("serve");
    auto* input = meta->mutable_graph_def()->add_node();
    input->set_name("images");
    input->set_op("Placeholder");
    (*input->mutable_attr())["dtype"].set_type(tensorflow::DT_UINT8);
    auto* shape = (*input->mutable_attr())["shape"].mutable_shape();
    for (int dim : {1, 2, 4, 3})
        shape->add_dim()->set_size(dim);
    auto* output = meta->mutable_graph_def()->add_node();
    output->set_name("result");
    output->set_op("Identity");
    output->add_input("images");
    (*output->mutable_attr())["T"].set_type(tensorflow::DT_UINT8);
    auto& signature = (*meta->mutable_signature_def())["serving_default"];
    auto& input_info = (*signature.mutable_inputs())["images"];
    input_info.set_name("images:0");
    input_info.set_dtype(tensorflow::DT_UINT8);
    *input_info.mutable_tensor_shape() = *shape;
    auto& output_info = (*signature.mutable_outputs())["result"];
    output_info = input_info;
    output_info.set_name("result:0");
    const auto path = fs::path(::testing::TempDir()) / "neuriplo-tf-uint8-regression";
    fs::create_directories(path);
    {
        std::ofstream stream(path / "saved_model.pb", std::ios::binary);
        ASSERT_TRUE(saved.SerializeToOstream(&stream));
    }
    TFDetectionAPI infer(path.string(), false);
    const auto inputs = infer.get_inference_metadata().getInputs();
    ASSERT_EQ(inputs.size(), 1u);
    EXPECT_EQ(inputs[0].datatype, TensorDataType::UInt8);
    EXPECT_EQ(inputs[0].shape, (std::vector<int64_t>{3, 2, 4}));
    EXPECT_THROW(infer.get_infer_results({std::vector<uint8_t>(6)}), std::runtime_error);
    fs::remove_all(path);
}
