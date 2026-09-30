#include "NativeInfer.hpp"

#include "InferenceBackendSetup.hpp"
#include "NativeRuntimeFactory.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

constexpr const char* kFixtureEnv = "NEURIPLO_NATIVE_FIXTURE";
constexpr const char* kFixtureFallback = "/tmp/opencode/resnet18_export/resnet18.onnx";
constexpr size_t kInputElements = 1u * 3u * 224u * 224u;

// The fixture is resolved from the environment first (ctest passes the cache
// variable through), then the well-known local export. A missing fixture is a
// test failure, never a skip.
std::string resolve_fixture() {
    const char* from_env = std::getenv(kFixtureEnv);
    if (from_env != nullptr && from_env[0] != '\0') {
        return from_env;
    }
    if (std::filesystem::exists(kFixtureFallback)) {
        return kFixtureFallback;
    }
    return "";
}

std::vector<std::vector<uint8_t>> make_input() {
    std::vector<uint8_t> bytes(kInputElements * sizeof(float));
    for (size_t i = 0; i < kInputElements; ++i) {
        const float value = static_cast<float>(i % 17) * 0.01F;
        std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(float));
    }
    return {std::move(bytes)};
}

} // namespace

class NativeInferTest : public ::testing::Test {
  protected:
    void SetUp() override {
        fixture_ = resolve_fixture();
        ASSERT_FALSE(fixture_.empty()) << "ResNet-18 fixture not found; set " << kFixtureEnv << " or place it at "
                                       << kFixtureFallback;
        ASSERT_TRUE(std::filesystem::exists(fixture_)) << "fixture does not exist: " << fixture_;
    }

    std::string fixture_;
};

TEST_F(NativeInferTest, MetadataHasSingleInputAndOutput) {
    NativeInfer infer(fixture_);

    const InferenceMetadata metadata = infer.get_inference_metadata();
    ASSERT_EQ(metadata.getInputs().size(), 1u);
    ASSERT_EQ(metadata.getOutputs().size(), 1u);
    EXPECT_EQ(metadata.getOutputs()[0].shape, (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(metadata.getOutputs()[0].datatype, TensorDataType::Float32);
}

TEST_F(NativeInferTest, TypedInferenceReturnsOneThousandFloats) {
    NativeInfer infer(fixture_);

    auto [outputs, shapes] = infer.get_infer_results(make_input());

    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].size(), 1000u);
    EXPECT_TRUE(std::holds_alternative<float>(outputs[0][0]));
    ASSERT_EQ(shapes.size(), 1u);
    EXPECT_EQ(shapes[0], (std::vector<int64_t>{1, 1000}));
}

TEST_F(NativeInferTest, RawInferenceReturnsFp32Bytes) {
    NativeInfer infer(fixture_);

    const std::vector<RawOutputTensor> raw = infer.get_infer_results_raw(make_input());

    ASSERT_EQ(raw.size(), 1u);
    EXPECT_EQ(raw[0].dtype, TensorDtype::FP32);
    EXPECT_EQ(raw[0].shape, (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(raw[0].bytes.size(), 4000u);
    EXPECT_EQ(raw[0].element_count(), 1000u);
}

TEST_F(NativeInferTest, RawAndTypedValuesAgreeElementwise) {
    NativeInfer infer(fixture_);
    const std::vector<std::vector<uint8_t>> input = make_input();

    auto [typed, shapes] = infer.get_infer_results(input);
    const std::vector<RawOutputTensor> raw = infer.get_infer_results_raw(input);

    ASSERT_EQ(typed.size(), raw.size());
    ASSERT_EQ(typed[0].size(), 1000u);
    ASSERT_EQ(raw[0].bytes.size(), typed[0].size() * sizeof(float));

    for (size_t i = 0; i < typed[0].size(); ++i) {
        float raw_value = 0.0F;
        std::memcpy(&raw_value, raw[0].bytes.data() + i * sizeof(float), sizeof(float));
        EXPECT_FLOAT_EQ(std::get<float>(typed[0][i]), raw_value);
    }
}

TEST(NativeRuntimeFactoryTest, RejectsGpuRequests) {
    NativeRuntimeFactory factory;

    try {
        (void)factory.create_backend("unused.onnx", true, 1, {});
        FAIL() << "expected a GPU request to be rejected";
    } catch (const ::InferenceException& ex) {
        EXPECT_NE(std::string(ex.what()).find("CPU only"), std::string::npos) << ex.what();
    }

    EXPECT_STREQ(factory.name(), "NativeRuntimeFactory");
    EXPECT_NE(factory.create_allocator(), nullptr);
    EXPECT_NE(factory.create_converter(), nullptr);
}

// [V-8]b: the NATIVE factory rejects GPU requests by throwing the global
// InferenceException; the public boundary translates that throw into the
// nullptr contract. No engine is constructed and no inference runs.
TEST_F(NativeInferTest, EngineOptionsGpuRequestReturnsNull) {
    EngineOptions options;
    options.model_path = fixture_;
    options.backend_id = "NATIVE";
    options.use_gpu = true;

    EXPECT_EQ(setup_inference_engine(options), nullptr);
}

TEST_F(NativeInferTest, LegacyOverloadGpuRequestReturnsNull) {
    EXPECT_EQ(setup_inference_engine(fixture_, /*use_gpu=*/true, 1, {}), nullptr);
}

// Control: the same NATIVE backend with the GPU request removed must construct
// a ready backend. This proves the nulls above come from the GPU rejection and
// not from a general failure (e.g. an unresolvable backend id or model).
TEST_F(NativeInferTest, CpuRequestReturnsReadyBackend) {
    EngineOptions options;
    options.model_path = fixture_;
    options.backend_id = "NATIVE";
    options.use_gpu = false;

    auto backend = setup_inference_engine(options);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->state(), BackendState::Ready);
}

TEST(NativeInferErrorTest, MissingModelThrowsModelLoadException) {
    try {
        NativeInfer infer("/nonexistent/neuriplo-native-missing.onnx");
        (void)infer;
        FAIL() << "expected a missing model path to throw";
    } catch (const ::ModelLoadException& ex) {
        EXPECT_NE(std::string(ex.what()).find("Model loading failed"), std::string::npos) << ex.what();
    }
}
