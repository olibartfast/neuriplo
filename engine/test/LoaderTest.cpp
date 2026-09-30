// Model-loader tests.
//
// Two suites, split for ctest filters (suite names must not carry the ctest
// NAME colons, so the filters live in the add_test COMMAND):
//   EngineLoader.*           — positives, fixture + hermetic wire-bytes cases
//   EngineLoaderNegative.*   — rejection classes with message-content asserts
//
// Positives run on real fixture bytes (the reference ResNet-18 export, with
// external weight data) when the fixture is resolvable via the
// NEURIPLO_NATIVE_FIXTURE environment variable or CMake cache variable. A
// missing fixture FAILS the test — never skipped or mocked — so acceptance
// cannot silently pass without the real bytes. The hermetic cases are
// hand-encoded model bytes and keep CI green without torch or ONNX. The field
// numbers the test encoders use mirror those in ModelLoader.cpp; no public
// header carries any.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "engine/Graph.hpp"
#include "engine/ModelLoader.hpp"

namespace {

// ---------------------------------------------------------------------------
// Hand-rolled protobuf wire encoders (test-side twin of the reader).
//
// Field numbers: ModelProto graph=7 opset_import=8; GraphProto node=1
// initializer=5 input=11 output=12; NodeProto input=1 output=2 name=3
// op_type=4 attribute=5; AttributeProto name=1 f=2 i=3 s=4 type=20;
// TensorProto dims=1 data_type=2 name=8 raw_data=9; OpsetIdList
// domain=1 version=2; ValueInfoProto name=1 type=2; TypeProto tensor_type=1;
// TensorType elem_type=1 shape=2; TensorShapeProto dim=1; Dimension
// dim_value=1 dim_param=2.
// ---------------------------------------------------------------------------

void put_varint(std::string& out, uint64_t value)
{
    while (value >= 0x80U) {
        out.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

void put_tag(std::string& out, int field_number, uint8_t wire_type)
{
    put_varint(out, (static_cast<uint64_t>(field_number) << 3) | wire_type);
}

void put_length_delimited(std::string& out, int field_number,
    const std::string& payload)
{
    put_tag(out, field_number, 2);
    put_varint(out, payload.size());
    out.append(payload);
}

void put_sub(std::string& out, int field_number, const std::string& sub)
{
    put_length_delimited(out, field_number, sub);
}

void put_varint_field(std::string& out, int field_number, uint64_t value)
{
    put_tag(out, field_number, 0);
    put_varint(out, value);
}

void put_fixed32_field(std::string& out, int field_number, uint32_t bits)
{
    put_tag(out, field_number, 5);
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((bits >> (8 * i)) & 0xFFU));
    }
}

std::string encode_opset18()
{
    // Empty domain string decodes as the default/"ai.onnx" domain.
    std::string leaving; // domain (field 1) omitted in the import entry
    put_varint_field(leaving, 2, 18);
    return leaving;
}

std::string encode_value_info(const std::string& name, int64_t elem_type,
    const std::vector<int64_t>& dims, bool dynamic = false)
{
    std::string vi;
    put_length_delimited(vi, 1, name);
    std::string tensor_type;
    put_varint_field(tensor_type, 1, static_cast<uint64_t>(elem_type));
    std::string shape;
    for (const int64_t dim : dims) {
        std::string dim_msg;
        if (dynamic) {
            put_length_delimited(dim_msg, 2, std::string("N"));
        } else {
            put_varint_field(dim_msg, 1, static_cast<uint64_t>(dim));
        }
        put_sub(shape, 1, dim_msg);
    }
    put_sub(tensor_type, 2, shape);
    std::string type_proto;
    put_sub(type_proto, 1, tensor_type);
    put_sub(vi, 2, type_proto);
    return vi;
}

std::string encode_node(const std::string& name, const std::string& op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& attr_bytes)
{
    std::string n;
    for (const std::string& input : inputs) {
        put_length_delimited(n, 1, input);
    }
    for (const std::string& output : outputs) {
        put_length_delimited(n, 2, output);
    }
    if (!name.empty()) {
        put_length_delimited(n, 3, name);
    }
    put_length_delimited(n, 4, op_type);
    for (const std::string& attr : attr_bytes) {
        put_sub(n, 5, attr);
    }
    return n;
}

std::string encode_attribute_int(const std::string& name, int64_t value)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_varint_field(a, 3, static_cast<uint64_t>(value));
    put_varint_field(a, 20, 2); // INT
    return a;
}

std::string encode_attribute_float(const std::string& name, float value)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_fixed32_field(a, 2,
        *reinterpret_cast<const uint32_t*>(&value));
    put_varint_field(a, 20, 1); // FLOAT
    return a;
}

std::string encode_attribute_name_only(const std::string& name)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_varint_field(a, 20, 2); // INT payload kind, but no i field
    return a;
}

std::string encode_initializer(const std::string& name,
    const std::vector<int64_t>& dims, const std::vector<float>& values)
{
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 1); // FLOAT
    put_length_delimited(t, 8, name);
    const std::string raw(reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(float));
    put_length_delimited(t, 9, raw);
    return t;
}

// An integer shape-constant initializer. `via_raw_data` mirrors real exports
// (little-endian raw bytes, TensorProto.raw_data = 9); the alternative writes
// the packed int64_data field (7). Both decode paths are exercised.
std::string encode_initializer_int64(const std::string& name,
    const std::vector<int64_t>& dims, const std::vector<int64_t>& values,
    bool via_raw_data = true)
{
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 7); // INT64
    put_length_delimited(t, 8, name);
    if (via_raw_data) {
        const std::string raw(reinterpret_cast<const char*>(values.data()),
            values.size() * sizeof(int64_t));
        put_length_delimited(t, 9, raw);
    } else {
        for (const int64_t value : values) {
            put_varint_field(t, 7, static_cast<uint64_t>(value));
        }
    }
    return t;
}

// A non-FLOAT, non-INT64 initializer used to assert the rejection message.
std::string encode_initializer_float16(const std::string& name,
    const std::vector<int64_t>& dims, const std::string& raw)
{
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 10); // FLOAT16
    put_length_delimited(t, 8, name);
    put_length_delimited(t, 9, raw);
    return t;
}

std::string encode_model(const std::string& graph_body)
{
    std::string model;
    put_sub(model, 8, encode_opset18());
    put_sub(model, 7, graph_body);
    return model;
}

// Holds hermetic models in temporary files; returns the written path.
std::string write_model_bytes(const std::string& name, const std::string& bytes)
{
    const std::string dir =
        (std::filesystem::temp_directory_path() / "engine_loader_tests")
            .string();
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/" + name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) {
        ADD_FAILURE() << "cannot write test model file: " << path;
    }
    return path;
}

std::string load_error_message(const std::string& bytes,
    const std::string& file_name)
{
    try {
        engine::LoadGraphFromFile(write_model_bytes(file_name, bytes));
    } catch (const engine::ModelLoadException& e) {
        return e.what();
    }
    return "<no ModelLoadException thrown>";
}

const char* fixture_env_name() { return "NEURIPLO_NATIVE_FIXTURE"; }

} // namespace

// ===========================================================================
// EngineLoader.* — positive cases
// ===========================================================================

class EngineLoader : public ::testing::Test {
protected:
    // Env var first; never hard-codes a path inside the test source.
    static std::string fixture_path()
    {
        const char* from_env = std::getenv(fixture_env_name());
        return from_env != nullptr ? std::string(from_env) : std::string();
    }
};

TEST_F(EngineLoader, RealFixtureLoads)
{
    const std::string fixture = fixture_path();
    ASSERT_FALSE(fixture.empty())
        << "missing fixture: set " << fixture_env_name()
        << " to the opset-18 ResNet-18 ONNX model; a missing fixture must "
           "fail, never skip";
    ASSERT_TRUE(std::filesystem::exists(fixture))
        << "fixture path from " << fixture_env_name()
        << " does not exist: " << fixture;
    EXPECT_NO_THROW(engine::LoadGraphFromFile(fixture));
}

TEST_F(EngineLoader, RealFixtureNodeCountAndOpsHistogram)
{
    const std::string fixture = fixture_path();
    ASSERT_TRUE(std::filesystem::exists(fixture));
    const engine::Graph graph = engine::LoadGraphFromFile(fixture);
    ASSERT_EQ(graph.nodes.size(), 49U); // reference model node count

    std::map<std::string, size_t> histogram;
    for (const engine::Node& node : graph.nodes) {
        histogram[node.op_type]++;
    }
    EXPECT_EQ(histogram.at("Conv"), 20U);
    EXPECT_EQ(histogram.at("Relu"), 17U);
    EXPECT_EQ(histogram.at("Add"), 8U);
    EXPECT_EQ(histogram.at("MaxPool"), 1U);
    EXPECT_EQ(histogram.at("ReduceMean"), 1U);
    EXPECT_EQ(histogram.at("Reshape"), 1U);
    EXPECT_EQ(histogram.at("Gemm"), 1U);
    EXPECT_EQ(histogram.size(), 7U); // no operator outside the supported set
}

TEST_F(EngineLoader, RealFixtureInitializerCount)
{
    const std::string fixture = fixture_path();
    ASSERT_TRUE(std::filesystem::exists(fixture));
    const engine::Graph graph = engine::LoadGraphFromFile(fixture);
    EXPECT_EQ(graph.initializers.size(), 44U); // reference model initializers
    for (const auto& init : graph.initializers) {
        EXPECT_FALSE(init.first.empty());
        if (init.second.dtype == engine::DataType::Int64) {
            EXPECT_FALSE(
                std::get<std::vector<std::int64_t>>(init.second.values).empty());
        } else {
            EXPECT_EQ(init.second.dtype, engine::DataType::Float32);
            EXPECT_FALSE(std::get<std::vector<float>>(init.second.values)
                    .empty()); // resolved, real float payloads
        }
    }
}

// The fixture carries int64 Reshape shape constants alongside FLOAT weights.
// Both decode, and the value table records their dtype.

TEST_F(EngineLoader, RealFixtureInt64ShapeConstants)
{
    const std::string fixture = fixture_path();
    ASSERT_TRUE(std::filesystem::exists(fixture));
    const engine::Graph graph = engine::LoadGraphFromFile(fixture);

    ASSERT_EQ(graph.initializers.count("val_230"), 1U);
    const engine::Initializer& shape = graph.initializers.at("val_230");
    EXPECT_EQ(shape.dtype, engine::DataType::Int64);
    EXPECT_EQ(shape.dims, (std::vector<std::int64_t>{2}));
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(shape.values),
        (std::vector<std::int64_t>{1, 512}));

    ASSERT_EQ(graph.tensors.count("val_230"), 1U);
    EXPECT_EQ(graph.tensors.at("val_230").dtype, engine::DataType::Int64);
}

TEST_F(EngineLoader, RealFixtureIODtypesAndShapes)
{
    const std::string fixture = fixture_path();
    ASSERT_TRUE(std::filesystem::exists(fixture));
    const engine::Graph graph = engine::LoadGraphFromFile(fixture);
    ASSERT_EQ(graph.inputs.size(), 1U);
    EXPECT_EQ(graph.inputs.front().second.dtype, engine::DataType::Float32);
    EXPECT_EQ(graph.inputs.front().second.dims,
        (std::vector<std::int64_t>{1, 3, 224, 224})); // FLOAT [1,3,224,224]
    ASSERT_EQ(graph.outputs.size(), 1U);
    EXPECT_EQ(graph.outputs.front().second.dtype, engine::DataType::Float32);
    EXPECT_EQ(graph.outputs.front().second.dims,
        (std::vector<std::int64_t>{1, 1000})); // -> FLOAT [1,1000]
}

// Hermetic positive: single Relu node, no initializers, static IO.

TEST_F(EngineLoader, HermeticReluNodeAndGraphIo)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3, 4}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("relu_model.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().name, "relu0");
    EXPECT_EQ(graph.nodes.front().op_type, "Relu");
    EXPECT_EQ(graph.nodes.front().inputs, (std::vector<std::string>{"x"}));
    EXPECT_EQ(graph.nodes.front().outputs, (std::vector<std::string>{"y"}));
    EXPECT_TRUE(graph.nodes.front().attributes.empty());
    EXPECT_EQ(graph.initializers.size(), 0U);
    ASSERT_EQ(graph.inputs.size(), 1U);
    EXPECT_EQ(graph.inputs.front().first, "x");
    EXPECT_EQ(graph.inputs.front().second.dtype, engine::DataType::Float32);
    EXPECT_EQ(graph.inputs.front().second.dims,
        (std::vector<int64_t>{1, 2, 3, 4}));
    ASSERT_EQ(graph.outputs.size(), 1U);
    EXPECT_EQ(graph.outputs.front().first, "y");
    EXPECT_EQ(graph.outputs.front().second.dims,
        (std::vector<int64_t>{1, 2, 3, 4}));
}

// Hermetic positive: Gemm typed attributes + FLOAT initializer via raw_data.

TEST_F(EngineLoader, HermeticGemmAttributesAndInitializer)
{
    std::string body;
    put_sub(body, 5, encode_initializer("w", {2, 3},
        {1.0F, -2.5F, 3.0F, 4.25F, 0.5F, -1.0F}));
    put_sub(body, 1, encode_node("gemm0", "Gemm", {"y", "w"}, {"z"},
        {encode_attribute_float("alpha", 1.0F),
            encode_attribute_float("beta", 1.0F),
            encode_attribute_int("transA", 0),
            encode_attribute_int("transB", 1)}));
    put_sub(body, 11, encode_value_info("y", 1, {2, 2}));
    put_sub(body, 12, encode_value_info("z", 1, {2, 3}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("gemm_model.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    std::map<std::string, engine::Attribute> attrs;
    for (const engine::Attribute& a : graph.nodes.front().attributes) {
        attrs[a.name] = a;
    }
    ASSERT_EQ(attrs.size(), 4U);
    EXPECT_EQ(std::holds_alternative<float>(attrs.at("alpha").value), true);
    EXPECT_EQ(std::get<float>(attrs.at("alpha").value), 1.0F);
    EXPECT_EQ(std::get<float>(attrs.at("beta").value), 1.0F);
    EXPECT_EQ(std::get<std::int64_t>(attrs.at("transA").value), 0);
    EXPECT_EQ(std::get<std::int64_t>(attrs.at("transB").value), 1);
    ASSERT_EQ(graph.initializers.size(), 1U);
    const engine::Initializer& w = graph.initializers.at("w");
    EXPECT_EQ(w.dtype, engine::DataType::Float32);
    EXPECT_EQ(std::get<std::vector<float>>(w.values),
        (std::vector<float>{1.0F, -2.5F, 3.0F, 4.25F, 0.5F, -1.0F}));
}

// Hermetic positive: an int64 shape constant feeds Reshape, so a float32 graph
// with an integer constant loads. Both storage paths decode.

TEST_F(EngineLoader, HermeticInt64ReshapeConstant)
{
    for (const bool via_raw_data : {true, false}) {
        std::string body;
        put_sub(body, 5, encode_initializer_int64("shape", {2}, {1, 512},
                          via_raw_data));
        put_sub(body, 1, encode_node("reshape0", "Reshape", {"x", "shape"},
                          {"y"}, {encode_attribute_int("allowzero", 0)}));
        put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
        put_sub(body, 12, encode_value_info("y", 1, {1, 48}));
        put_sub(body, 13, encode_value_info("shape", 7 /*INT64*/, {2}));

        const engine::Graph graph = engine::LoadGraphFromFile(write_model_bytes(
            via_raw_data ? "reshape_raw.onnx" : "reshape_varint.onnx",
            encode_model(body)));
        ASSERT_EQ(graph.initializers.size(), 1U);
        const engine::Initializer& shape = graph.initializers.at("shape");
        EXPECT_EQ(shape.dtype, engine::DataType::Int64);
        EXPECT_EQ(shape.dims, (std::vector<std::int64_t>{2}));
        EXPECT_EQ(std::get<std::vector<std::int64_t>>(shape.values),
            (std::vector<std::int64_t>{1, 512}));
        ASSERT_EQ(graph.tensors.count("shape"), 1U);
        EXPECT_EQ(graph.tensors.at("shape").dtype, engine::DataType::Int64);
    }
}

// ===========================================================================
// EngineLoaderNegative.* — rejections whose messages must name the node and
// op type.
// ===========================================================================

class EngineLoaderNegative : public ::testing::Test {
};

// Unknown op: message must name the node AND the op type.

TEST_F(EngineLoaderNegative, UnknownOpNamesNodeAndOpType)
{
    std::string body;
    put_sub(body, 1, encode_node("bogus_node", "DepthWiseConv",
        {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3, 4}));

    const std::string message =
        load_error_message(encode_model(body), "unknown_op.onnx");
    EXPECT_NE(message.find("bogus_node"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("DepthWiseConv"), std::string::npos)
        << "message: " << message;
}

// Unsupported attribute: message must name the node AND the op type.

TEST_F(EngineLoaderNegative, UnsupportedAttributeNamesNodeAndOpType)
{
    // Relu accepts no attributes: any attribute is rejected with full
    // node/op context.
    std::string body;
    put_sub(body, 1, encode_node("relu_foo", "Relu", {"x"}, {"y"},
        {encode_attribute_name_only("leaky")}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3, 4}));

    const std::string message =
        load_error_message(encode_model(body), "bad_attr.onnx");
    EXPECT_NE(message.find("relu_foo"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Relu"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("leaky"), std::string::npos) << "message: " << message;
}

// Non-FP32 I/O: INT64 input tensor is rejected by name and element type.

TEST_F(EngineLoaderNegative, NonFloatGraphInputRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("probe_node", "Relu",
        {"x_int"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x_int", 7 /*INT64*/, {2, 2}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 2}));

    const std::string message =
        load_error_message(encode_model(body), "int_in.onnx");
    EXPECT_NE(message.find("x_int"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("INT64"), std::string::npos) << "message: " << message;
}

// An initializer dtype outside FLOAT/INT64 is rejected by name and type.

TEST_F(EngineLoaderNegative, NonFloatInitializerRejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer_float16("half_w", {2}, "\x00\x3c\x00\x40"));
    put_sub(body, 1, encode_node("relu0", "Relu", {"x", "half_w"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body), "fp16_init.onnx");
    EXPECT_NE(message.find("half_w"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("FLOAT16"), std::string::npos)
        << "message: " << message;
}

// Dynamic dimension (dim_param instead of dim_value) is rejected by name.

TEST_F(EngineLoaderNegative, DynamicDimensionRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("dynamic_node", "Relu",
        {"x_dyn"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x_dyn", 1, {2, 2}, true));
    put_sub(body, 12, encode_value_info("y", 1, {2, 2}));

    const std::string message =
        load_error_message(encode_model(body), "dyn_in.onnx");
    EXPECT_NE(message.find("x_dyn"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("non-fixed"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("N"), std::string::npos) << "message: " << message;
}

// Missing file: the offending path itself appears in the message.

TEST_F(EngineLoaderNegative, MissingModelFileNamesPath)
{
    const std::string path =
        (std::filesystem::temp_directory_path() /
            "engine_loader_tests" / "no_such_model.onnx")
            .string();
    std::string message;
    try {
        engine::LoadGraphFromFile(path);
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "missing model file did not throw";
    EXPECT_NE(message.find(path), std::string::npos) << "message: " << message;
}
