#include "LiteRTInfer.hpp"

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
