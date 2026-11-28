#include "LiteRTInfer.hpp"

#include "LiteRTInputLayout.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

namespace fs = std::filesystem;

TEST(LiteRTInferTest, SkipsWhenNoModelIsAvailable) {
    std::ifstream model_path_file("model_path.txt");
    std::string model_path;
    if (model_path_file) {
        std::getline(model_path_file, model_path);
    }

    if (model_path.empty() || !fs::exists(model_path)) {
        GTEST_SKIP() << "No LiteRT model available";
    }

    LiteRTInfer infer(model_path, false);
    const auto metadata = infer.get_inference_metadata();
    ASSERT_FALSE(metadata.getInputs().empty());
    ASSERT_FALSE(metadata.getOutputs().empty());
}

TEST(LiteRTInputLayoutTest, UInt8TransposesWithoutFloatReinterpretation) {
    const std::vector<uint8_t> input{10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61};
    std::vector<uint8_t> output(input.size());
    neuriplo::detail::transpose_nchw_to_nhwc(input, output, 2, 3, 1, 2);
    EXPECT_EQ(output, (std::vector<uint8_t>{10, 20, 30, 11, 21, 31, 40, 50, 60, 41, 51, 61}));
}

TEST(LiteRTInputLayoutTest, Float32RetainsAllBytes) {
    const std::vector<float> input{10, 11, 20, 21, 30, 31};
    std::vector<uint8_t> bytes(input.size() * sizeof(float));
    std::memcpy(bytes.data(), input.data(), bytes.size());
    std::vector<uint8_t> output(bytes.size());
    neuriplo::detail::transpose_nchw_to_nhwc(bytes, output, 1, 3, 1, 2);
    std::vector<float> values(input.size());
    std::memcpy(values.data(), output.data(), output.size());
    EXPECT_EQ(values, (std::vector<float>{10, 20, 30, 11, 21, 31}));
}

TEST(LiteRTInputLayoutTest, RejectsMismatchedBuffers) {
    std::vector<uint8_t> output(24);
    EXPECT_THROW(neuriplo::detail::transpose_nchw_to_nhwc({1, 2, 3, 4, 5, 6}, output, 1, 3, 1, 2),
                 InferenceExecutionException);
}

namespace {
class LiteRTMetadataTest : public ::testing::Test {
  protected:
    fs::path model_path;
    void SetUp() override {
        model_path = fs::temp_directory_path() / "neuriplo-litert-metadata-identity.tflite";
        tflite::ModelT model;
        model.version = TFLITE_SCHEMA_VERSION;
        model.buffers.push_back(std::make_unique<tflite::BufferT>());
        auto graph = std::make_unique<tflite::SubGraphT>();
        auto tensor = std::make_unique<tflite::TensorT>();
        tensor->name = "nhwc_identity";
        tensor->shape = {1, 4, 2, 3};
        tensor->type = tflite::TensorType_FLOAT32;
        graph->tensors.push_back(std::move(tensor));
        graph->inputs = {0};
        graph->outputs = {0};
        model.subgraphs.push_back(std::move(graph));
        flatbuffers::FlatBufferBuilder builder;
        tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
        std::ofstream file(model_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                   static_cast<std::streamsize>(builder.GetSize()));
    }
    void TearDown() override { fs::remove(model_path); }
};
} // namespace

TEST_F(LiteRTMetadataTest, ReportsTheCallerLayoutAndPreservesNativeOutput) {
    LiteRTInfer infer(model_path.string(), false);
    const auto metadata = infer.get_inference_metadata();
    ASSERT_EQ(metadata.getInputs().size(), 1U);
    EXPECT_EQ(metadata.getInputs()[0].shape, (std::vector<int64_t>{1, 3, 4, 2}));
    EXPECT_EQ(metadata.getOutputs()[0].shape, (std::vector<int64_t>{1, 4, 2, 3}));
    std::vector<float> nchw(24);
    for (size_t i = 0; i < nchw.size(); ++i) {
        nchw[i] = static_cast<float>(i);
    }
    std::vector<uint8_t> bytes(nchw.size() * sizeof(float));
    std::memcpy(bytes.data(), nchw.data(), bytes.size());
    const auto [outputs, shapes] = infer.get_infer_results({bytes});
    ASSERT_EQ(outputs.size(), 1U);
    ASSERT_EQ(outputs[0].size(), 24U);
    for (size_t pixel = 0; pixel < 8; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            EXPECT_FLOAT_EQ(std::get<float>(outputs[0][pixel * 3 + channel]), nchw[channel * 8 + pixel]);
        }
    }
}

TEST_F(LiteRTMetadataTest, ExplicitInputSizeUsesTheCallerLayout) {
    LiteRTInfer infer(model_path.string(), false, 1, {{1, 3, 4, 2}});
    EXPECT_EQ(infer.get_inference_metadata().getInputs()[0].shape, (std::vector<int64_t>{1, 3, 4, 2}));
    EXPECT_EQ(infer.get_inference_metadata().getOutputs()[0].shape, (std::vector<int64_t>{1, 4, 2, 3}));
}

TEST_F(LiteRTMetadataTest, BatchOmittedInputSizeUsesTheCallerLayout) {
    LiteRTInfer infer(model_path.string(), false, 1, {{3, 4, 2}});
    EXPECT_EQ(infer.get_inference_metadata().getInputs()[0].shape, (std::vector<int64_t>{1, 3, 4, 2}));
    EXPECT_EQ(infer.get_inference_metadata().getOutputs()[0].shape, (std::vector<int64_t>{1, 4, 2, 3}));
}

TEST_F(LiteRTMetadataTest, PreservesExplicitNativeInputSize) {
    LiteRTInfer infer(model_path.string(), false, 1, {{1, 4, 2, 3}});
    EXPECT_EQ(infer.get_inference_metadata().getInputs()[0].shape, (std::vector<int64_t>{1, 3, 4, 2}));
    EXPECT_EQ(infer.get_inference_metadata().getOutputs()[0].shape, (std::vector<int64_t>{1, 4, 2, 3}));
}

// Optional real-model regression: generate the fixture from identical ORT
// inputs/outputs using the downstream backend coverage validation packet.
TEST(LiteRTInferTest, DynamicEdgeCrafterMatchesOnnxReference) {
    const char* fixture_env = std::getenv("NEURIPLO_LITERT_PARITY_FIXTURE");
    if (fixture_env == nullptr) {
        GTEST_SKIP() << "NEURIPLO_LITERT_PARITY_FIXTURE is not set";
    }
    const fs::path fixture(fixture_env);
    LiteRTInfer infer((fixture / "model.tflite").string(), false);
    std::vector<std::vector<uint8_t>> inputs;
    for (size_t i = 0; i < infer.get_inference_metadata().getInputs().size(); ++i) {
        std::ifstream input(fixture / ("input" + std::to_string(i) + ".bin"), std::ios::binary);
        ASSERT_TRUE(input.is_open());
        inputs.emplace_back(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    const auto [outputs, shapes] = infer.get_infer_results(inputs);
    ASSERT_EQ(outputs.size(), 3U);
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::ifstream reference(fixture / ("output" + std::to_string(i) + ".bin"), std::ios::binary);
        ASSERT_TRUE(reference.is_open());
        for (size_t j = 0; j < outputs[i].size(); ++j) {
            float expected = 0;
            ASSERT_TRUE(reference.read(reinterpret_cast<char*>(&expected), sizeof(expected)));
            const float actual = std::visit([](auto value) { return static_cast<float>(value); }, outputs[i][j]);
            // Query ordering is score-sorted; compare the high-confidence set
            // separately from low-score numerical ties between queries.
            if (i == 2 || j < (i == 1 ? 24U : 6U)) {
                EXPECT_NEAR(actual, expected, i == 0 ? 0.0f : (i == 1 ? 0.01f : 0.0001f))
                    << "output " << i << ", element " << j;
            }
        }
        EXPECT_EQ(reference.peek(), std::ifstream::traits_type::eof());
    }
}
