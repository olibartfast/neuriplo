#include "LiteRTInfer.hpp"

#include "LiteRTInputLayout.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

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
