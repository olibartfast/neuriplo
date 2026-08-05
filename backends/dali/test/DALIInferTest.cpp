#include "DALIInfer.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Path to a serialized pipeline, written by the test harness. Generate one with
// export/dali/generate_yolo_pipeline.py. Without it the runtime tests skip:
// DALI needs a GPU, which CI runners do not have.
std::string pipelinePath() {
    std::ifstream file("dali_pipeline_path.txt");
    std::string path;
    if (file) {
        std::getline(file, path);
    }
    return path;
}

std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

} // namespace

// Loading is where every contract violation is caught, so the failure paths are
// worth pinning even on a machine with no GPU.
TEST(DALIInferTest, RejectsAMissingPipelineFile) {
    EXPECT_THROW(DALIInfer("definitely-not-a-pipeline.dali", true, 1, {{3, 640, 640}}), ModelLoadException);
}

TEST(DALIInferTest, RejectsAFileThatIsNotASerializedPipeline) {
    const std::string path = "not_a_pipeline.dali";
    {
        std::ofstream out(path, std::ios::binary);
        out << "this is plainly not a serialized DALI pipeline";
    }
    EXPECT_THROW(DALIInfer(path, true, 1, {{3, 640, 640}}), ModelLoadException);
    std::remove(path.c_str());
}

// The contract fixes one image per request; a batched pipeline would silently
// process only the first.
TEST(DALIInferTest, RejectsBatchSizesOtherThanOne) {
    const std::string path = pipelinePath();
    if (path.empty() || !fs::exists(path)) {
        GTEST_SKIP() << "No serialized DALI pipeline available";
    }
    EXPECT_THROW(DALIInfer(path, true, 4, {{3, 640, 640}}), ModelLoadException);
}

TEST(DALIInferTest, ReportsEncodedImageInputAndPreprocessedOutputs) {
    const std::string path = pipelinePath();
    if (path.empty() || !fs::exists(path)) {
        GTEST_SKIP() << "No serialized DALI pipeline available";
    }

    DALIInfer infer(path, true, 1, {{3, 640, 640}});
    const auto metadata = infer.get_inference_metadata();

    ASSERT_EQ(metadata.getInputs().size(), 1u);
    EXPECT_EQ(metadata.getInputs()[0].name, DALIInfer::kEncodedInputName);
    ASSERT_GE(metadata.getOutputs().size(), 1u);
    EXPECT_EQ(metadata.getOutputs()[0].name, DALIInfer::kPreprocessedOutputName);
}

TEST(DALIInferTest, DecodesAnEncodedImageIntoThePreprocessedTensor) {
    const std::string path = pipelinePath();
    std::ifstream image_path_file("dali_image_path.txt");
    std::string image_path;
    if (image_path_file) {
        std::getline(image_path_file, image_path);
    }
    if (path.empty() || !fs::exists(path) || image_path.empty() || !fs::exists(image_path)) {
        GTEST_SKIP() << "No serialized DALI pipeline and encoded image available";
    }

    DALIInfer infer(path, true, 1, {{3, 640, 640}});
    const auto outputs = infer.get_infer_results_raw({readFile(image_path)});

    // Output 0 is the model tensor; output 1 carries the source dimensions.
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0].dtype, TensorDtype::FP32);
    EXPECT_EQ(outputs[0].element_count(), 3u * 640u * 640u);
    EXPECT_EQ(outputs[1].dtype, TensorDtype::INT32);
    EXPECT_EQ(outputs[1].element_count(), 3u);
}

TEST(DALIInferTest, RejectsAnEmptyEncodedImage) {
    const std::string path = pipelinePath();
    if (path.empty() || !fs::exists(path)) {
        GTEST_SKIP() << "No serialized DALI pipeline available";
    }
    DALIInfer infer(path, true, 1, {{3, 640, 640}});
    EXPECT_THROW(infer.get_infer_results_raw({{}}), InferenceExecutionException);
}
