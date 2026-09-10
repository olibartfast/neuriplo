#include "DALIInfer.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
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

namespace {
std::string fixture(const char* name) {
    const char* root = std::getenv("NEURIPLO_DALI_TEST_FIXTURES");
    if (root == nullptr || *root == '\0') {
        return {};
    }
    return (fs::path(root) / (std::string(name) + ".dali")).string();
}

// Resolved path of the most recent require_fixture call.
std::string fixture_path;

void require_fixture(const char* name) {
    fixture_path = fixture(name);
    if (fixture_path.empty() || !fs::exists(fixture_path)) {
        GTEST_SKIP() << "DALI fixture unavailable: " << name;
    }
}

DALIInfer load_fixture(const char* name, const std::vector<std::vector<int64_t>>& shapes = {{3}}) {
    require_fixture(name);
    return DALIInfer(fixture_path.c_str(), false, 1, shapes);
}

template <typename T> std::vector<uint8_t> bytes(std::initializer_list<T> values) {
    std::vector<uint8_t> result(values.size() * sizeof(T));
    size_t offset = 0;
    for (const T value : values) {
        std::memcpy(result.data() + offset, &value, sizeof(T));
        offset += sizeof(T);
    }
    return result;
}

template <typename T> T value_at(const RawOutputTensor& output, size_t index = 0) {
    T value{};
    std::memcpy(&value, output.bytes.data() + index * sizeof(T), sizeof(T));
    return value;
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
    EXPECT_EQ(metadata.getOutputs()[0].name, "output0");
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
    EXPECT_EQ(outputs[1].dtype, TensorDtype::INT64);
    EXPECT_EQ(outputs[1].element_count(), 2u);
}

TEST(DALIInferTest, RejectsAnEmptyEncodedImage) {
    const std::string path = pipelinePath();
    if (path.empty() || !fs::exists(path)) {
        GTEST_SKIP() << "No serialized DALI pipeline available";
    }
    DALIInfer infer(path, true, 1, {{3, 640, 640}});
    EXPECT_THROW(infer.get_infer_results_raw({{}}), InferenceExecutionException);
}

TEST(DALIInferFixtureTest, PreservesUint8RawOutput) {
    auto infer = load_fixture("single_uint8");
    const auto output = infer.get_infer_results_raw({bytes<uint8_t>({1, 2, 255})});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].dtype, TensorDtype::UINT8);
    EXPECT_EQ(output[0].shape, (std::vector<int64_t>{1, 3}));
    EXPECT_EQ(output[0].bytes, bytes<uint8_t>({1, 2, 255}));
}
TEST(DALIInferFixtureTest, PreservesInt32RawOutput) {
    auto infer = load_fixture("single_int32");
    const auto output = infer.get_infer_results_raw({bytes<int32_t>({-7, 42, 100000})});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].dtype, TensorDtype::INT32);
    EXPECT_EQ(value_at<int32_t>(output[0], 1), 42);
}
TEST(DALIInferFixtureTest, RejectsPartiallyAlignedTypedInputWithMatchingShape) {
    auto infer = load_fixture("single_int32", {{1}});
    EXPECT_THROW(infer.get_infer_results_raw({std::vector<uint8_t>(sizeof(int32_t) + 1)}), InferenceExecutionException);
}
TEST(DALIInferFixtureTest, RejectsShapeWhoseTypedByteCountOverflows) {
    auto infer = load_fixture("single_int32", {{int64_t{1} << 62}});
    EXPECT_THROW(infer.get_infer_results_raw({bytes<int32_t>({1})}), InferenceExecutionException);
}
TEST(DALIInferFixtureTest, RejectsElementCountOverflowAsExecutionFailure) {
    auto infer = load_fixture("single_int32_2d", {{int64_t{1} << 62, 4}});
    EXPECT_THROW(infer.get_infer_results_raw({bytes<int32_t>({1})}), InferenceExecutionException);
}
TEST(DALIInferFixtureTest, PreservesInt64RawOutput) {
    auto infer = load_fixture("single_int64");
    const auto output =
        infer.get_infer_results_raw({bytes<int64_t>({std::numeric_limits<int64_t>::min(), 9007199254740993LL, 3})});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].dtype, TensorDtype::INT64);
    EXPECT_EQ(value_at<int64_t>(output[0], 1), 9007199254740993LL);
}
TEST(DALIInferFixtureTest, PreservesFloatRawOutput) {
    auto infer = load_fixture("single_float");
    const auto output = infer.get_infer_results_raw({bytes<float>({-1.25F, 3.5F, 9.0F})});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].dtype, TensorDtype::FP32);
    EXPECT_FLOAT_EQ(value_at<float>(output[0], 1), 3.5F);
}
TEST(DALIInferFixtureTest, ReportsExactOutputCounts) {
    for (const auto* name : {"two_outputs", "three_outputs", "four_outputs"}) {
        auto infer = load_fixture(name);
        const auto output = infer.get_infer_results_raw({bytes<uint8_t>({3, 4, 5})});
        EXPECT_EQ(output.size(), std::string(name) == "two_outputs"     ? 2u
                                 : std::string(name) == "three_outputs" ? 3u
                                                                        : 4u);
    }
}
TEST(DALIInferFixtureTest, MetadataIsReadyBeforeInferenceAndLearnsRuntimeTypes) {
    auto infer = load_fixture("single_int64");
    const auto before = infer.get_inference_metadata();
    ASSERT_EQ(before.getOutputs().size(), 1u);
    EXPECT_EQ(before.getOutputs()[0].datatype, TensorDataType::Int64);
    infer.get_infer_results_raw({bytes<int64_t>({1, 2, 3})});
    const auto after = infer.get_inference_metadata();
    EXPECT_EQ(after.getOutputs()[0].shape, (std::vector<int64_t>{1, 3}));
    EXPECT_EQ(after.getOutputs()[0].datatype, TensorDataType::Int64);
}
TEST(DALIInferFixtureTest, RuntimeShapeReplacesOutHint) {
    const auto path = fixture("changing_shape");
    if (path.empty() || !fs::exists(path))
        GTEST_SKIP() << "DALI fixture unavailable";
    DALIInfer infer(path + "|out=99", false, 1, {{}});
    EXPECT_EQ(infer.get_inference_metadata().getOutputs()[0].shape, (std::vector<int64_t>{1, 99}));
    infer.get_infer_results_raw({bytes<int32_t>({1, 2, 3, 4})});
    EXPECT_EQ(infer.get_inference_metadata().getOutputs()[0].shape, (std::vector<int64_t>{1, 4}));
}
TEST(DALIInferFixtureTest, RepeatedChangingShapesUpdateMetadata) {
    auto infer = load_fixture("changing_shape", {{}});
    infer.get_infer_results_raw({bytes<int32_t>({1, 2})});
    EXPECT_EQ(infer.get_inference_metadata().getOutputs()[0].shape, (std::vector<int64_t>{1, 2}));
    infer.get_infer_results_raw({bytes<int32_t>({1, 2, 3, 4, 5})});
    EXPECT_EQ(infer.get_inference_metadata().getOutputs()[0].shape, (std::vector<int64_t>{1, 5}));
}
TEST(DALIInferFixtureTest, InvalidInputPreservesLastMetadata) {
    auto infer = load_fixture("changing_shape");
    infer.get_infer_results_raw({bytes<int32_t>({1, 2, 3})});
    const auto shape = infer.get_inference_metadata().getOutputs()[0].shape;
    EXPECT_THROW(infer.get_infer_results_raw({}), InferenceExecutionException);
    EXPECT_EQ(infer.get_inference_metadata().getOutputs()[0].shape, shape);
}
TEST(DALIInferFixtureTest, RejectsBatchTwoWithDiagnostic) {
    const auto path = fixture("batch2");
    if (path.empty() || !fs::exists(path))
        GTEST_SKIP() << "DALI fixture unavailable";
    try {
        DALIInfer infer(path, false, 2, {{3}});
        FAIL();
    } catch (const ModelLoadException& error) {
        EXPECT_NE(std::string(error.what()).find("batch"), std::string::npos);
    }
}
TEST(DALIInferFixtureTest, RejectsDeclaredFloat16) {
    require_fixture("unsupported_declared_float16");
    // An unsupported declared type is rejected at load, before any inference.
    EXPECT_THROW(DALIInfer infer(fixture_path.c_str(), false, 1, {{3}}), ModelLoadException);
}
TEST(DALIInferFixtureTest, RejectsRuntimeFloat16) {
    auto infer = load_fixture("unsupported_runtime_float16");
    EXPECT_THROW(infer.get_infer_results_raw({bytes<uint8_t>({1, 2, 3, 4, 5, 6})}), InferenceExecutionException);
}
TEST(DALIInferFixtureTest, KeepsLargeInt64ExactAcrossTwoInputs) {
    auto infer = load_fixture("multi_input_identity", {{3}, {3}});
    const auto output = infer.get_infer_results_raw(
        {bytes<int64_t>({9007199254740993LL, -4, 7}), bytes<int64_t>({-9007199254740993LL, 8, 9})});
    ASSERT_EQ(output.size(), 2u);
    EXPECT_EQ(value_at<int64_t>(output[0]), 9007199254740993LL);
    EXPECT_EQ(value_at<int64_t>(output[1]), -9007199254740993LL);
}
TEST(DALIInferFixtureTest, RejectsOutputNameCountMismatch) {
    const auto path = fixture("two_outputs");
    if (path.empty() || !fs::exists(path))
        GTEST_SKIP() << "DALI fixture unavailable";
    EXPECT_THROW(DALIInfer(path + "|outnames=only_one", false, 1, {{3}}), ModelLoadException);
}
TEST(DALIInferFixtureTest, HandlesUndeclaredOutputType) {
    auto infer = load_fixture("undeclared_type");
    const auto output = infer.get_infer_results_raw({bytes<uint8_t>({1, 2, 3})});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].dtype, TensorDtype::UINT8);
}
TEST(DALIInferFixtureTest, HandlesUndeclaredOutputRank) {
    auto infer = load_fixture("undeclared_rank");
    const auto output = infer.get_infer_results_raw({bytes<uint8_t>({1, 2, 3})});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].shape, (std::vector<int64_t>{1, 3}));
}
TEST(DALIInferFixtureTest, HandlesMixedDeclarations) {
    auto infer = load_fixture("mixed_declared", {{3}, {3}});
    const auto output = infer.get_infer_results_raw({bytes<int32_t>({1, 2, 3}), bytes<uint8_t>({4, 5, 6})});
    ASSERT_EQ(output.size(), 2u);
    EXPECT_EQ(output[0].dtype, TensorDtype::INT32);
    EXPECT_EQ(output[1].dtype, TensorDtype::UINT8);
}
