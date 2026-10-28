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
    OCVDNNInfer infer(OPENCV_COMPAT_FIXTURE, false, 2, {{1, 2, 3}});
    EXPECT_THROW(infer.get_infer_results({std::vector<uint8_t>(6 * sizeof(float))}), InferenceExecutionException);
}

TEST(OpenCVCompatibility, RejectsInvalidInputDimensions) {
    OCVDNNInfer infer(OPENCV_COMPAT_FIXTURE, false, 1, {{1, -2, 3}});
    EXPECT_THROW(infer.get_infer_results({std::vector<uint8_t>(24)}), InferenceExecutionException);
}
