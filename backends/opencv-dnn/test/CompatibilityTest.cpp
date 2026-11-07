#include "OCVDNNInfer.hpp"

#include <cstring>
#include <gtest/gtest.h>
#include <opencv2/core/version.hpp>
#include <stdexcept>
#include <string>
#include <vector>

TEST(OpenCVCompatibility, RealOnnxValuesShapesAndRawBuffers) {
    OCVDNNInfer infer(OPENCV_COMPAT_FIXTURE, false, 1, {{1, 2, 3}});
    const std::vector<float> input{-3, -2, -1, 0, 1, 2};
    const std::vector<float> expected{0, 0, 0, 0, 1, 2};
    std::vector<uint8_t> bytes(input.size() * sizeof(float));
    std::memcpy(bytes.data(), input.data(), bytes.size());
    const auto [values, shapes] = infer.get_infer_results({bytes});
    ASSERT_EQ(shapes, (std::vector<std::vector<int64_t>>{{1, 1, 2, 3}}));
    ASSERT_EQ(values.size(), 1u);
    ASSERT_EQ(values[0].size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(std::get<float>(values[0][i]), expected[i]);
    }
    const auto raw = infer.get_infer_results_raw({bytes});
    ASSERT_EQ(raw.size(), 1u);
    EXPECT_EQ(raw[0].shape, shapes[0]);
    EXPECT_EQ(raw[0].dtype, TensorDtype::FP32);
    ASSERT_EQ(raw[0].bytes.size(), bytes.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        float value;
        std::memcpy(&value, raw[0].bytes.data() + i * sizeof(float), sizeof(float));
        EXPECT_FLOAT_EQ(value, expected[i]);
    }
}

#if CV_VERSION_MAJOR >= 5
TEST(OpenCVCompatibility, RejectsDarknetWithMigrationAdvice) {
    try {
        OCVDNNInfer infer("missing.weights", false, 1, {{1, 2, 3}});
        FAIL() << "Darknet must be rejected";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("Convert the model to ONNX"), std::string::npos);
    }
}

TEST(OpenCVCompatibility, RejectsGpuBeforeModelLoadingEvenWithoutCuda) {
    try {
        OCVDNNInfer infer("missing.onnx", true, 1, {{1, 2, 3}});
        FAIL() << "GPU request must be rejected";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("use_gpu=false"), std::string::npos);
    }
}
#endif

TEST(OpenCVCompatibility, RejectsMismatchedInputBuffersBeforeForward) {
    OCVDNNInfer infer(OPENCV_COMPAT_FIXTURE, false, 1, {{1, 2, 3}});
    for (const size_t bytes : {size_t{0}, size_t{6}, size_t{23}, size_t{25}}) {
        EXPECT_THROW(infer.get_infer_results({std::vector<uint8_t>(bytes)}), InferenceExecutionException);
        EXPECT_THROW(infer.get_infer_results_raw({std::vector<uint8_t>(bytes)}), InferenceExecutionException);
    }
}

TEST(OpenCVCompatibility, CountsBatchInInputByteSize) {
#if CV_VERSION_MAJOR >= 5
    EXPECT_THROW((OCVDNNInfer{OPENCV_COMPAT_FIXTURE, false, 2, {{1, 2, 3}}}), ModelLoadException);
#else
    OCVDNNInfer infer(OPENCV_COMPAT_FIXTURE, false, 2, {{1, 2, 3}});
    EXPECT_THROW(infer.get_infer_results({std::vector<uint8_t>(6 * sizeof(float))}), InferenceExecutionException);
#endif
}

TEST(OpenCVCompatibility, RejectsInvalidInputDimensions) {
#if CV_VERSION_MAJOR >= 5
    EXPECT_THROW((OCVDNNInfer{OPENCV_COMPAT_FIXTURE, false, 1, {{1, -2, 3}}}), ModelLoadException);
#else
    OCVDNNInfer infer(OPENCV_COMPAT_FIXTURE, false, 1, {{1, -2, 3}});
    EXPECT_THROW(infer.get_infer_results({std::vector<uint8_t>(24)}), InferenceExecutionException);
#endif
}

#if CV_VERSION_MAJOR >= 5
TEST(OpenCVCompatibility, NamedMixedInputsPreserveInt64ValuesAndRawOutputs) {
    OCVDNNInfer infer(OPENCV_MIXED_FIXTURE, false);
    const auto metadata = infer.get_inference_metadata().getInputs();
    ASSERT_EQ(metadata.size(), 2U);
    EXPECT_EQ(metadata[0].name, "image");
    EXPECT_EQ(metadata[1].name, "tokens");
    EXPECT_EQ(metadata[1].datatype, TensorDataType::Int64);
    const std::vector<float> image{1, 2};
    const std::vector<int64_t> tokens{3, 4};
    std::vector<std::vector<uint8_t>> input{std::vector<uint8_t>(sizeof(float) * image.size()),
                                            std::vector<uint8_t>(sizeof(int64_t) * tokens.size())};
    std::memcpy(input[0].data(), image.data(), input[0].size());
    std::memcpy(input[1].data(), tokens.data(), input[1].size());
    const auto [values, shapes] = infer.get_infer_results(input);
    const auto raw = infer.get_infer_results_raw(input);
    ASSERT_EQ(values.size(), 2U);
    ASSERT_EQ(raw.size(), 2U);
    for (size_t i = 0; i < values.size(); ++i) {
        ASSERT_EQ(values[i].size(), 2U);
        EXPECT_EQ(raw[i].shape, shapes[i]);
        if (std::holds_alternative<int64_t>(values[i][0])) {
            EXPECT_EQ(raw[i].dtype, TensorDtype::INT64);
            ASSERT_EQ(raw[i].bytes.size(), tokens.size() * sizeof(int64_t));
            for (size_t j = 0; j < tokens.size(); ++j) {
                const int64_t expected = (int64_t{1} << 35) + tokens[j];
                EXPECT_EQ(std::get<int64_t>(values[i][j]), expected);
                int64_t actual;
                std::memcpy(&actual, raw[i].bytes.data() + j * sizeof(actual), sizeof(actual));
                EXPECT_EQ(actual, expected);
            }
        } else {
            EXPECT_EQ(raw[i].dtype, TensorDtype::FP32);
            EXPECT_FLOAT_EQ(std::get<float>(values[i][0]), 4);
            EXPECT_FLOAT_EQ(std::get<float>(values[i][1]), 6);
        }
    }
    EXPECT_THROW(infer.get_infer_results({input[0]}), InferenceExecutionException);
    input[1].resize(8);
    EXPECT_THROW(infer.get_infer_results(input), InferenceExecutionException);
    EXPECT_THROW((OCVDNNInfer{OPENCV_MIXED_FIXTURE, false, 1, {{1, 2}}}), ModelLoadException);
    EXPECT_THROW((OCVDNNInfer{OPENCV_MIXED_FIXTURE, false, 1, {{1, 3}, {1, 2}}}), ModelLoadException);
}
#endif
