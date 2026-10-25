#include "NertInfer.hpp"

#include "InferenceBackendSetup.hpp"
#include "NertRuntimeFactory.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <variant>
#include <vector>

namespace {

constexpr const char* kFixtureEnv = "NERT_FIXTURE";
constexpr size_t kInputElements = 1u * 3u * 224u * 224u;

// The fixture is resolved from the environment (ctest passes the cache
// variable through). A missing fixture is a test failure, never a skip; the
// CTest entry itself is registered only when the fixture is configured.
std::string resolve_fixture() {
    const char* from_env = std::getenv(kFixtureEnv);
    if (from_env != nullptr && from_env[0] != '\0') {
        return from_env;
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

// ---------------------------------------------------------------------------
// A hermetic typed-I/O model, hand-encoded the way the engine loader tests do
// (field numbers mirror engine/src/ModelLoader.cpp). Graph input `ids` is
// int64 [1,2]; outputs are `sum` = ids + c (int64), `flag` = ids > c (bool),
// and `as_float` = Cast(ids, FLOAT) (float32), with c = [2, 2] int64.
// ---------------------------------------------------------------------------

void put_varint(std::string& out, uint64_t value) {
    while (value >= 0x80U) {
        out.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

void put_varint_field(std::string& out, int field_number, uint64_t value) {
    put_varint(out, static_cast<uint64_t>(field_number) << 3);
    put_varint(out, value);
}

void put_bytes_field(std::string& out, int field_number, const std::string& payload) {
    put_varint(out, (static_cast<uint64_t>(field_number) << 3) | 2U);
    put_varint(out, payload.size());
    out.append(payload);
}

std::string encode_value_info(const std::string& name, int64_t elem_type, const std::vector<int64_t>& dims) {
    std::string shape;
    for (const int64_t dim : dims) {
        std::string dim_msg;
        put_varint_field(dim_msg, 1, static_cast<uint64_t>(dim));
        put_bytes_field(shape, 1, dim_msg);
    }
    std::string tensor_type;
    put_varint_field(tensor_type, 1, static_cast<uint64_t>(elem_type));
    put_bytes_field(tensor_type, 2, shape);
    std::string type_proto;
    put_bytes_field(type_proto, 1, tensor_type);
    std::string vi;
    put_bytes_field(vi, 1, name);
    put_bytes_field(vi, 2, type_proto);
    return vi;
}

std::string encode_node(const std::string& name, const std::string& op_type, const std::vector<std::string>& inputs,
                        const std::vector<std::string>& outputs, const std::vector<std::string>& attributes = {}) {
    std::string n;
    for (const std::string& input : inputs) {
        put_bytes_field(n, 1, input);
    }
    for (const std::string& output : outputs) {
        put_bytes_field(n, 2, output);
    }
    put_bytes_field(n, 3, name);
    put_bytes_field(n, 4, op_type);
    for (const std::string& attribute : attributes) {
        put_bytes_field(n, 5, attribute);
    }
    return n;
}

std::string encode_attribute_int(const std::string& name, int64_t value) {
    std::string a;
    put_bytes_field(a, 1, name);
    put_varint_field(a, 3, static_cast<uint64_t>(value));
    put_varint_field(a, 20, 2); // INT
    return a;
}

std::string encode_int64_initializer(const std::string& name, const std::vector<int64_t>& dims,
                                     const std::vector<int64_t>& values) {
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 7); // INT64
    put_bytes_field(t, 8, name);
    put_bytes_field(t, 9, std::string(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(int64_t)));
    return t;
}

std::string write_typed_io_model() {
    std::string body;
    put_bytes_field(body, 5, encode_int64_initializer("c", {1, 2}, {2, 2}));
    put_bytes_field(body, 1, encode_node("add0", "Add", {"ids", "c"}, {"sum"}));
    put_bytes_field(body, 1, encode_node("greater0", "Greater", {"ids", "c"}, {"flag"}));
    put_bytes_field(body, 1, encode_node("cast0", "Cast", {"ids"}, {"as_float"}, {encode_attribute_int("to", 1)}));
    put_bytes_field(body, 11, encode_value_info("ids", 7 /*INT64*/, {1, 2}));
    put_bytes_field(body, 12, encode_value_info("sum", 7 /*INT64*/, {1, 2}));
    put_bytes_field(body, 12, encode_value_info("flag", 9 /*BOOL*/, {1, 2}));
    put_bytes_field(body, 12, encode_value_info("as_float", 1 /*FLOAT*/, {1, 2}));

    std::string opset;
    put_varint_field(opset, 2, 18);
    std::string model;
    put_bytes_field(model, 8, opset);
    put_bytes_field(model, 7, body);

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "nert_infer_tests";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "typed_io.onnx").string();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(model.data(), static_cast<std::streamsize>(model.size()));
    return path;
}

std::vector<uint8_t> int64_bytes(const std::vector<int64_t>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

} // namespace

class NertInferTest : public ::testing::Test {
  protected:
    void SetUp() override {
        fixture_ = resolve_fixture();
        ASSERT_FALSE(fixture_.empty()) << "ResNet-18 fixture not found; set " << kFixtureEnv
                                       << " (the CTest entry is registered only when it is configured)";
        ASSERT_TRUE(std::filesystem::exists(fixture_)) << "fixture does not exist: " << fixture_;
    }

    std::string fixture_;
};

TEST_F(NertInferTest, MetadataHasSingleInputAndOutput) {
    NertInfer infer(fixture_);

    const InferenceMetadata metadata = infer.get_inference_metadata();
    ASSERT_EQ(metadata.getInputs().size(), 1u);
    ASSERT_EQ(metadata.getOutputs().size(), 1u);
    EXPECT_EQ(metadata.getOutputs()[0].shape, (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(metadata.getOutputs()[0].datatype, TensorDataType::Float32);
}

TEST_F(NertInferTest, TypedInferenceReturnsOneThousandFloats) {
    NertInfer infer(fixture_);

    auto [outputs, shapes] = infer.get_infer_results(make_input());

    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].size(), 1000u);
    EXPECT_TRUE(std::holds_alternative<float>(outputs[0][0]));
    ASSERT_EQ(shapes.size(), 1u);
    EXPECT_EQ(shapes[0], (std::vector<int64_t>{1, 1000}));
}

TEST_F(NertInferTest, RawInferenceReturnsFp32Bytes) {
    NertInfer infer(fixture_);

    const std::vector<RawOutputTensor> raw = infer.get_infer_results_raw(make_input());

    ASSERT_EQ(raw.size(), 1u);
    EXPECT_EQ(raw[0].dtype, TensorDtype::FP32);
    EXPECT_EQ(raw[0].shape, (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(raw[0].bytes.size(), 4000u);
    EXPECT_EQ(raw[0].element_count(), 1000u);
}

TEST_F(NertInferTest, RawAndTypedValuesAgreeElementwise) {
    NertInfer infer(fixture_);
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

TEST(NertRuntimeFactoryTest, RejectsGpuRequests) {
    NertRuntimeFactory factory;

#ifdef NEURIPLO_NERT_CUDA
    auto backend = factory.create_backend(write_typed_io_model(), true, 1, {});
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->state(), BackendState::Ready);
#else
    try {
        (void)factory.create_backend("unused.onnx", true, 1, {});
        FAIL() << "expected a GPU request to be rejected";
    } catch (const ::InferenceException& ex) {
        EXPECT_NE(std::string(ex.what()).find("CPU only"), std::string::npos) << ex.what();
    }
#endif

    EXPECT_STREQ(factory.name(), "NertRuntimeFactory");
    EXPECT_NE(factory.create_allocator(), nullptr);
    EXPECT_NE(factory.create_converter(), nullptr);
}

// [V-8]b: without CUDA support the NERT factory rejects GPU requests by
// throwing the global InferenceException; the public boundary translates
// that throw into the nullptr contract. With NEURIPLO_NERT_CUDA the same
// requests construct ready GPU backends (covered below).
TEST_F(NertInferTest, EngineOptionsGpuRequestReturnsNull) {
#ifdef NEURIPLO_NERT_CUDA
    // The ResNet fixture needs cuDNN-gated Conv rows, so the GPU-ready
    // check runs on the hermetic typed_io model (cuDNN-free CUDA rows).
    EngineOptions gpu_options;
    gpu_options.model_path = write_typed_io_model();
    gpu_options.backend_id = "NERT";
    gpu_options.use_gpu = true;
    auto backend = setup_inference_engine(gpu_options);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->state(), BackendState::Ready);
#else
    EngineOptions options;
    options.model_path = fixture_;
    options.backend_id = "NERT";
    options.use_gpu = true;
    EXPECT_EQ(setup_inference_engine(options), nullptr);
#endif
}

TEST_F(NertInferTest, LegacyOverloadGpuRequestReturnsNull) {
#ifdef NEURIPLO_NERT_CUDA
    auto backend = setup_inference_engine(write_typed_io_model(), /*use_gpu=*/true, 1, {});
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->state(), BackendState::Ready);
#else
    EXPECT_EQ(setup_inference_engine(fixture_, /*use_gpu=*/true, 1, {}), nullptr);
#endif
}

#ifdef NEURIPLO_NERT_CUDA
// Typed-I/O GPU-vs-CPU parity: Add/Greater/Cast agree exactly on both
// devices (integer and bool paths are bitwise; Cast is exact).
TEST_F(NertInferTest, TypedIoGpuMatchesCpu) {
    NertInfer cpu(write_typed_io_model());
    NertInfer gpu(write_typed_io_model(), /*use_gpu=*/true);
    const std::vector<std::vector<uint8_t>> input = {int64_bytes({1, 5})};

    auto [cpu_typed, cpu_shapes] = cpu.get_infer_results(input);
    auto [gpu_typed, gpu_shapes] = gpu.get_infer_results(input);
    ASSERT_EQ(gpu_typed.size(), cpu_typed.size());
    ASSERT_EQ(gpu_shapes, cpu_shapes);
    for (size_t o = 0; o < cpu_typed.size(); ++o) {
        ASSERT_EQ(gpu_typed[o].size(), cpu_typed[o].size());
        for (size_t i = 0; i < cpu_typed[o].size(); ++i) {
            EXPECT_EQ(gpu_typed[o][i], cpu_typed[o][i]) << "output " << o << " index " << i;
        }
    }

    const std::vector<RawOutputTensor> cpu_raw = cpu.get_infer_results_raw(input);
    const std::vector<RawOutputTensor> gpu_raw = gpu.get_infer_results_raw(input);
    ASSERT_EQ(gpu_raw.size(), cpu_raw.size());
    for (size_t o = 0; o < cpu_raw.size(); ++o) {
        EXPECT_EQ(gpu_raw[o].dtype, cpu_raw[o].dtype);
        EXPECT_EQ(gpu_raw[o].shape, cpu_raw[o].shape);
        EXPECT_EQ(gpu_raw[o].bytes, cpu_raw[o].bytes);
    }
}
#endif

// Control: the same NERT backend with the GPU request removed must construct
// a ready backend. This proves the nulls above come from the GPU rejection and
// not from a general failure (e.g. an unresolvable backend id or model).
TEST_F(NertInferTest, CpuRequestReturnsReadyBackend) {
    EngineOptions options;
    options.model_path = fixture_;
    options.backend_id = "NERT";
    options.use_gpu = false;

    auto backend = setup_inference_engine(options);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->state(), BackendState::Ready);
}

TEST(NertInferErrorTest, MissingModelThrowsModelLoadException) {
    try {
        NertInfer infer("/nonexistent/nert-missing.onnx");
        (void)infer;
        FAIL() << "expected a missing model path to throw";
    } catch (const ::ModelLoadException& ex) {
        EXPECT_NE(std::string(ex.what()).find("Model loading failed"), std::string::npos) << ex.what();
    }
}

TEST(NertInferTypedIoTest, MetadataReportsInt64InputAndTypedOutputs) {
    NertInfer infer(write_typed_io_model());

    const InferenceMetadata metadata = infer.get_inference_metadata();
    ASSERT_EQ(metadata.getInputs().size(), 1u);
    EXPECT_EQ(metadata.getInputs()[0].name, "ids");
    EXPECT_EQ(metadata.getInputs()[0].datatype, TensorDataType::Int64);
    ASSERT_EQ(metadata.getOutputs().size(), 3u);
    EXPECT_EQ(metadata.getOutputs()[0].datatype, TensorDataType::Int64);
    EXPECT_EQ(metadata.getOutputs()[1].datatype, TensorDataType::Bool);
    EXPECT_EQ(metadata.getOutputs()[2].datatype, TensorDataType::Float32);
}

TEST(NertInferTypedIoTest, Int64InputYieldsTypedElements) {
    NertInfer infer(write_typed_io_model());

    auto [outputs, shapes] = infer.get_infer_results({int64_bytes({1, 5})});

    ASSERT_EQ(outputs.size(), 3u);
    ASSERT_EQ(outputs[0].size(), 2u);
    EXPECT_EQ(std::get<int64_t>(outputs[0][0]), 3);
    EXPECT_EQ(std::get<int64_t>(outputs[0][1]), 7);
    ASSERT_EQ(outputs[1].size(), 2u);
    EXPECT_EQ(std::get<uint8_t>(outputs[1][0]), 0u);
    EXPECT_EQ(std::get<uint8_t>(outputs[1][1]), 1u);
    ASSERT_EQ(outputs[2].size(), 2u);
    EXPECT_FLOAT_EQ(std::get<float>(outputs[2][0]), 1.0F);
    EXPECT_FLOAT_EQ(std::get<float>(outputs[2][1]), 5.0F);
    ASSERT_EQ(shapes.size(), 3u);
    EXPECT_EQ(shapes[0], (std::vector<int64_t>{1, 2}));
}

TEST(NertInferTypedIoTest, Int64InputYieldsTypedRawBuffers) {
    NertInfer infer(write_typed_io_model());

    const std::vector<RawOutputTensor> raw = infer.get_infer_results_raw({int64_bytes({1, 5})});

    ASSERT_EQ(raw.size(), 3u);
    EXPECT_EQ(raw[0].dtype, TensorDtype::INT64);
    ASSERT_EQ(raw[0].bytes.size(), 2u * sizeof(int64_t));
    int64_t sums[2] = {0, 0};
    std::memcpy(sums, raw[0].bytes.data(), sizeof(sums));
    EXPECT_EQ(sums[0], 3);
    EXPECT_EQ(sums[1], 7);
    EXPECT_EQ(raw[1].dtype, TensorDtype::UINT8);
    EXPECT_EQ(raw[1].bytes, (std::vector<uint8_t>{0, 1}));
    EXPECT_EQ(raw[2].dtype, TensorDtype::FP32);
    EXPECT_EQ(raw[2].element_count(), 2u);
}

TEST(NertInferTypedIoTest, Int64InputWithFloatSizedBytesIsRejected) {
    NertInfer infer(write_typed_io_model());

    // Two float32 values (8 bytes) are not the 16 bytes two int64 values need.
    const std::vector<uint8_t> float_sized(2 * sizeof(float), 0);
    try {
        (void)infer.get_infer_results({float_sized});
        FAIL() << "expected a mis-sized int64 input to be rejected";
    } catch (const ::InferenceExecutionException& ex) {
        EXPECT_NE(std::string(ex.what()).find("ids"), std::string::npos) << ex.what();
        EXPECT_NE(std::string(ex.what()).find("16 bytes"), std::string::npos) << ex.what();
    }
}
