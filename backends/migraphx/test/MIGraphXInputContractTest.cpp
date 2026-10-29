#include "MIGraphXInfer.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>

namespace {
template <class T> std::vector<uint8_t> bytes(const std::vector<T>& values) {
    std::vector<uint8_t> result(values.size() * sizeof(T));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

class MIGraphXInputContractTest : public ::testing::TestWithParam<bool> {
  protected:
    std::string fixture(const char* name) {
        const char* root = std::getenv("NEURIPLO_MIGRAPHX_CONTRACT_FIXTURES");
        if (!root)
            throw std::runtime_error("NEURIPLO_MIGRAPHX_CONTRACT_FIXTURES is required");
        return (std::filesystem::path(root) / name).string();
    }
    void SetUp() override {
        if (GetParam() && !std::getenv("NEURIPLO_TEST_AMD_GPU"))
            GTEST_SKIP() << "GPU contract suite requires NEURIPLO_TEST_AMD_GPU=1";
    }
};

TEST_P(MIGraphXInputContractTest, MixedInt64MetadataAndValues) {
    MIGraphXInfer infer(fixture("mixed-int64.onnx"), GetParam());
    const auto metadata = infer.get_inference_metadata();
    ASSERT_EQ(metadata.getInputs().size(), 2u);
    std::vector<std::vector<uint8_t>> inputs;
    for (const auto& input : metadata.getInputs()) {
        if (input.name == "image") {
            EXPECT_EQ(input.datatype, TensorDataType::Float32);
            inputs.push_back(bytes<float>({0.25f, 0.5f}));
        } else {
            EXPECT_EQ(input.name, "sizes");
            EXPECT_EQ(input.datatype, TensorDataType::Int64);
            inputs.push_back(bytes<int64_t>({3, 5}));
        }
    }
    auto [outputs, shapes] = infer.get_infer_results(inputs);
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].size(), 2u);
    EXPECT_EQ(shapes[0], (std::vector<int64_t>{1, 2}));
    ASSERT_TRUE(std::holds_alternative<float>(outputs[0][0]));
    EXPECT_FLOAT_EQ(std::get<float>(outputs[0][0]), 3.25f);
    EXPECT_FLOAT_EQ(std::get<float>(outputs[0][1]), 5.5f);
    inputs[0].push_back(0);
    EXPECT_THROW(infer.get_infer_results(inputs), std::exception);
}

TEST_P(MIGraphXInputContractTest, MixedInt32AndBufferBoundaries) {
    MIGraphXInfer infer(fixture("mixed-int32.onnx"), GetParam());
    std::vector<std::vector<uint8_t>> inputs;
    const auto metadata = infer.get_inference_metadata();
    for (const auto& input : metadata.getInputs()) {
        if (input.name == "image") {
            EXPECT_EQ(input.datatype, TensorDataType::Float32);
            inputs.push_back(bytes<float>({0.25f, 0.5f}));
        } else {
            EXPECT_EQ(input.datatype, TensorDataType::Int32);
            inputs.push_back(bytes<int32_t>({3, 5}));
        }
    }
    auto [outputs, shapes] = infer.get_infer_results(inputs);
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].size(), 2u);
    EXPECT_FLOAT_EQ(std::get<float>(outputs[0][0]), 3.25f);
    EXPECT_FLOAT_EQ(std::get<float>(outputs[0][1]), 5.5f);
    auto malformed = inputs;
    malformed[0].pop_back();
    EXPECT_THROW(infer.get_infer_results(malformed), std::exception);
    malformed[0].clear();
    EXPECT_THROW(infer.get_infer_results(malformed), std::exception);
    malformed = inputs;
    malformed.pop_back();
    EXPECT_THROW(infer.get_infer_results(malformed), std::exception);
    malformed = inputs;
    malformed.push_back(inputs[0]);
    EXPECT_THROW(infer.get_infer_results(malformed), std::exception);
}

TEST_P(MIGraphXInputContractTest, IntegerOutputPreservesValues) {
    MIGraphXInfer infer(fixture("integer-output.onnx"), GetParam());
    const auto metadata = infer.get_inference_metadata();
    ASSERT_EQ(metadata.getOutputs().size(), 1u);
    EXPECT_EQ(metadata.getOutputs()[0].datatype, TensorDataType::Int64);
    auto [outputs, shapes] = infer.get_infer_results({bytes<int64_t>({3, 5})});
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].size(), 2u);
    ASSERT_TRUE(std::holds_alternative<int64_t>(outputs[0][0]));
    EXPECT_EQ(std::get<int64_t>(outputs[0][0]), 6);
    EXPECT_EQ(std::get<int64_t>(outputs[0][1]), 10);
}

INSTANTIATE_TEST_SUITE_P(Targets, MIGraphXInputContractTest, ::testing::Values(false, true));
} // namespace
