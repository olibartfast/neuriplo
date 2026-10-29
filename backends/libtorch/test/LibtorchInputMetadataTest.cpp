#include "LibtorchInfer.hpp"
#include "LibtorchInputContract.hpp"
#include "LibtorchTensorUtils.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

// The metadata published for runtime input packing must carry each graph
// input's real scalar type. Regression: the constructor used to advertise
// every input as FP32 while only the internal packing list kept the real
// types, so an INT64 size tensor next to an FP32 image was packed as float.
TEST(LibtorchScalarToDatatype, CoversBackendSupportedKinds) {
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Float), TensorDataType::Float32);
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Int), TensorDataType::Int32);
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Long), TensorDataType::Int64);
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Byte), TensorDataType::UInt8);
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Char), TensorDataType::Int8);
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Bool), TensorDataType::Bool);
}

TEST(LibtorchScalarToDatatype, RejectsKindsWithoutMetadataEquivalent) {
    EXPECT_THROW(libtorch_scalar_to_datatype(c10::ScalarType::Double), std::runtime_error);
    EXPECT_THROW(libtorch_scalar_to_datatype(c10::ScalarType::Half), std::runtime_error);
}

namespace {

class SerializedInputModel {
  public:
    explicit SerializedInputModel(const std::string& metadata = "") {
        static std::atomic<unsigned> next{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("neuriplo-input-types-" + std::to_string(stamp) + "-" + std::to_string(next++) + ".pt");
        torch::jit::script::Module module("InputContract");
        module.define(R"JIT(
            def forward(self, image: Tensor, sizes: Tensor) -> Tensor:
                return image + sizes.to(image).sum()
        )JIT");
        torch::jit::ExtraFilesMap files;
        if (!metadata.empty()) {
            files["neuriplo_input_dtypes.txt"] = metadata;
        }
        module.save(path.string(), files);
    }
    ~SerializedInputModel() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    SerializedInputModel(const SerializedInputModel&) = delete;
    SerializedInputModel& operator=(const SerializedInputModel&) = delete;
    std::filesystem::path path;
};

const std::vector<std::vector<int64_t>> kInputShapes{{1, 3, 2, 2}, {1, 2}};

template <typename T> std::vector<uint8_t> input_bytes(const std::vector<T>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

} // namespace

// Serialized TorchScript graphs lose scalar types. An absent artifact contract
// must never become an invented FP32 signature, especially for auxiliary inputs.
TEST(LibtorchInputMetadata, RejectsMissingSerializedTypes) {
    SerializedInputModel model;
    EXPECT_THROW(LibtorchInfer(model.path.string(), false, 1, kInputShapes), ModelLoadException);
}

TEST(LibtorchInputMetadata, LoadsMixedInt64ArtifactTypesAndValues) {
    SerializedInputModel model("NEURIPLO_INPUT_DTYPES_V1 2 FP32 INT64");
    LibtorchInfer infer(model.path.string(), false, 1, kInputShapes);
    const auto inputs = infer.get_inference_metadata().getInputs();
    ASSERT_EQ(inputs.size(), 2U);
    ASSERT_EQ(inputs[0].datatype, TensorDataType::Float32);
    ASSERT_EQ(inputs[1].datatype, TensorDataType::Int64);
    const auto values = input_bytes<float>(std::vector<float>(12, 0.25F));
    const auto sizes = input_bytes<int64_t>({3, 5});
    auto [outputs, shapes] = infer.get_infer_results({values, sizes});
    ASSERT_EQ(shapes[0], kInputShapes[0]);
    for (const auto& value : outputs[0]) {
        EXPECT_FLOAT_EQ(std::get<float>(value), 8.25F);
    }
}

TEST(LibtorchInputMetadata, LoadsMixedInt32ArtifactTypes) {
    SerializedInputModel model("NEURIPLO_INPUT_DTYPES_V1 2 FP32 INT32");
    LibtorchInfer infer(model.path.string(), false, 1, kInputShapes);
    ASSERT_EQ(infer.get_inference_metadata().getInputs()[1].datatype, TensorDataType::Int32);
}

TEST(LibtorchInputMetadata, RejectsExtraBufferElements) {
    SerializedInputModel model("NEURIPLO_INPUT_DTYPES_V1 2 FP32 FP32");
    LibtorchInfer infer(model.path.string(), false, 1, kInputShapes);
    const auto values = input_bytes<float>(std::vector<float>(13, 0.25F));
    const auto sizes = input_bytes<float>({3.0F, 5.0F});
    EXPECT_THROW(infer.get_infer_results({values, sizes}), std::runtime_error);
}

TEST(LibtorchInputMetadata, RejectsMalformedAndUnsupportedArtifactTypes) {
    for (const auto* metadata :
         {"wrong 2 FP32 INT64", "NEURIPLO_INPUT_DTYPES_V1 1 FP32", "NEURIPLO_INPUT_DTYPES_V1 2 FP32",
          "NEURIPLO_INPUT_DTYPES_V1 2 FP32 FP16", "NEURIPLO_INPUT_DTYPES_V1 2 FP32 INT64 trailing"}) {
        SCOPED_TRACE(metadata);
        SerializedInputModel model(metadata);
        EXPECT_THROW(LibtorchInfer(model.path.string(), false, 1, kInputShapes), ModelLoadException);
    }
}

TEST(LibtorchInputMetadata, RejectsShortAndEmptyBuffers) {
    SerializedInputModel model("NEURIPLO_INPUT_DTYPES_V1 2 FP32 INT64");
    LibtorchInfer infer(model.path.string(), false, 1, kInputShapes);
    const auto sizes = input_bytes<int64_t>({3, 5});
    EXPECT_THROW(infer.get_infer_results({input_bytes<float>(std::vector<float>(11, 0.25F)), sizes}),
                 std::runtime_error);
    EXPECT_THROW(infer.get_infer_results({{}, sizes}), std::runtime_error);
}

TEST(LibtorchInputMetadata, PreservesByteAndBoolInputs) {
    struct Case {
        const char* tag;
        TensorDataType datatype;
        std::vector<uint8_t> buffer;
        float expected;
    };
    const std::vector<Case> cases{{"UINT8", TensorDataType::UInt8, input_bytes<uint8_t>({3, 5}), 8.25F},
                                  {"INT8", TensorDataType::Int8, input_bytes<int8_t>({-3, 5}), 2.25F},
                                  {"BOOL", TensorDataType::Bool, input_bytes<uint8_t>({1, 0}), 1.25F}};
    for (const auto& example : cases) {
        SCOPED_TRACE(example.tag);
        SerializedInputModel model(std::string("NEURIPLO_INPUT_DTYPES_V1 2 FP32 ") + example.tag);
        LibtorchInfer infer(model.path.string(), false, 1, kInputShapes);
        ASSERT_EQ(infer.get_inference_metadata().getInputs()[1].datatype, example.datatype);
        auto [outputs, shapes] =
            infer.get_infer_results({input_bytes<float>(std::vector<float>(12, 0.25F)), example.buffer});
        ASSERT_EQ(shapes[0], kInputShapes[0]);
        for (const auto& value : outputs[0]) {
            EXPECT_FLOAT_EQ(std::get<float>(value), example.expected);
        }
    }
}

TEST(LibtorchInputByteCount, ChecksDimensionsAndOverflow) {
    EXPECT_EQ(libtorch_input_bytes({1, 2}, c10::ScalarType::Long), 16U);
    EXPECT_EQ(libtorch_input_bytes({}, c10::ScalarType::Int), 4U);
    EXPECT_THROW(libtorch_input_bytes({0, 2}, c10::ScalarType::Float), std::runtime_error);
    EXPECT_THROW(libtorch_input_bytes({-1, 2}, c10::ScalarType::Float), std::runtime_error);
    EXPECT_THROW(libtorch_input_bytes({std::numeric_limits<int64_t>::max(), 2}, c10::ScalarType::Float),
                 std::runtime_error);
}

TEST(AotiInputContract, RejectsMissingAndMismatchedCompiledShapes) {
    EXPECT_NO_THROW(validate_aoti_input_shapes("NEURIPLO_INPUT_SHAPES_V1 2 4 1 3 2 2 2 1 2", kInputShapes));
    for (const auto* metadata :
         {"", "wrong 2", "NEURIPLO_INPUT_SHAPES_V1 1 4 1 3 2 2", "NEURIPLO_INPUT_SHAPES_V1 2 4 1 3 1 1 2 1 2",
          "NEURIPLO_INPUT_SHAPES_V1 2 4 1 3 2 2 2 1 2 trailing"}) {
        SCOPED_TRACE(metadata);
        EXPECT_THROW(validate_aoti_input_shapes(metadata, kInputShapes), ModelLoadException);
    }
}
