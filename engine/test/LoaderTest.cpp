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

std::string encode_attribute_string(const std::string& name,
    const std::string& value)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_length_delimited(a, 4, value);
    put_varint_field(a, 20, 3); // STRING
    return a;
}

std::string encode_attribute_ints(const std::string& name,
    const std::vector<int64_t>& values)
{
    std::string a;
    put_length_delimited(a, 1, name);
    for (const int64_t value : values) {
        put_varint_field(a, 8, static_cast<uint64_t>(value));
    }
    put_varint_field(a, 20, 7); // INTS
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

// Embedded TensorProto payload for a Constant 'value' attribute, stored via
// raw_data (mirrors the survey oracle, which carries all Constants raw).
std::string encode_constant_tensor(int64_t elem_type,
    const std::vector<int64_t>& dims, const std::string& raw)
{
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, static_cast<uint64_t>(elem_type));
    put_length_delimited(t, 9, raw);
    return t;
}

std::string raw_bytes_float(const std::vector<float>& values)
{
    return std::string(reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(float));
}

std::string raw_bytes_int64(const std::vector<int64_t>& values)
{
    return std::string(reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(int64_t));
}

// A Constant 'value' attribute (AttributeProto type TENSOR = 4, tensor = 5).
std::string encode_attribute_tensor(const std::string& name,
    const std::string& tensor_bytes)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_sub(a, 5, tensor_bytes);
    put_varint_field(a, 20, 4); // TENSOR
    return a;
}

std::string encode_constant_node(const std::string& name,
    const std::string& output, const std::string& tensor_bytes)
{
    return encode_node(name, "Constant", {}, {output},
        {encode_attribute_tensor("value", tensor_bytes)});
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

// Hermetic positive: a float32 Constant folds into initializers. The Constant
// node vanishes from graph.nodes and the Add consumer keeps naming the folded
// tensor, which now resolves as an initializer.
TEST_F(EngineLoader, HermeticConstantFoldingFloat)
{
    const std::vector<float> values{1.0F, -2.5F, 3.25F};
    std::string body;
    put_sub(body, 1,
        encode_constant_node("const_f0", "c0",
            encode_constant_tensor(1 /*FLOAT*/, {3}, raw_bytes_float(values))));
    put_sub(body, 1, encode_node("add0", "Add", {"x", "c0"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {3}));
    put_sub(body, 12, encode_value_info("y", 1, {3}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("const_fold_float.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().name, "add0");
    EXPECT_EQ(graph.nodes.front().op_type, "Add");
    EXPECT_EQ(graph.nodes.front().inputs,
        (std::vector<std::string>{"x", "c0"}));
    ASSERT_EQ(graph.initializers.count("c0"), 1U);
    const engine::Initializer& folded = graph.initializers.at("c0");
    EXPECT_EQ(folded.dtype, engine::DataType::Float32);
    EXPECT_EQ(folded.dims, (std::vector<std::int64_t>{3}));
    EXPECT_EQ(std::get<std::vector<float>>(folded.values), values);
}

// Hermetic positive: an int64 Constant folds into initializers and feeds
// Reshape, mirroring the survey oracle's int64 Constants.
TEST_F(EngineLoader, HermeticConstantFoldingInt64)
{
    const std::vector<int64_t> values{1, 512};
    std::string body;
    put_sub(body, 1,
        encode_constant_node("const_i0", "shape",
            encode_constant_tensor(7 /*INT64*/, {2}, raw_bytes_int64(values))));
    put_sub(body, 1, encode_node("reshape0", "Reshape", {"x", "shape"},
                      {"y"}, {encode_attribute_int("allowzero", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 48}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("const_fold_int64.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Reshape");
    ASSERT_EQ(graph.initializers.count("shape"), 1U);
    const engine::Initializer& folded = graph.initializers.at("shape");
    EXPECT_EQ(folded.dtype, engine::DataType::Int64);
    EXPECT_EQ(folded.dims, (std::vector<std::int64_t>{2}));
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(folded.values), values);
}

// Hermetic positive: Flatten loads with its axis attribute.
TEST_F(EngineLoader, HermeticFlattenLoads)
{
    std::string body;
    put_sub(body, 1, encode_node("flat0", "Flatten", {"x"}, {"y"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 12}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("flatten_model.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Flatten");
    ASSERT_EQ(graph.nodes.front().attributes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().attributes.front().name, "axis");
    EXPECT_EQ(
        std::get<std::int64_t>(graph.nodes.front().attributes.front().value),
        1);
}

// Hermetic positive: intermediate value-info entries may be float32, int64,
// or bool, while graph inputs/outputs stay float32-only (enforced below).
TEST_F(EngineLoader, HermeticBoolValueInfoAccepted)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));
    put_sub(body, 13, encode_value_info("flag", 9 /*BOOL*/, {2}));
    put_sub(body, 13, encode_value_info("idx", 7 /*INT64*/, {2}));
    put_sub(body, 13, encode_value_info("act", 1 /*FLOAT*/, {2}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("bool_vi.onnx", encode_model(body)));
    ASSERT_EQ(graph.tensors.count("flag"), 1U);
    EXPECT_EQ(graph.tensors.at("flag").dtype, engine::DataType::Bool);
    ASSERT_EQ(graph.tensors.count("idx"), 1U);
    EXPECT_EQ(graph.tensors.at("idx").dtype, engine::DataType::Int64);
    ASSERT_EQ(graph.tensors.count("act"), 1U);
    EXPECT_EQ(graph.tensors.at("act").dtype, engine::DataType::Float32);
}

// Hermetic positive: the no-attribute elementwise ops (Mul/Div/Sub/Sigmoid)
// chain without attributes.
TEST_F(EngineLoader, HermeticNewElementwiseLoads)
{
    std::string body;
    put_sub(body, 1, encode_node("sig0", "Sigmoid", {"x"}, {"s"}, {}));
    put_sub(body, 1, encode_node("mul0", "Mul", {"x", "s"}, {"m"}, {}));
    put_sub(body, 1, encode_node("div0", "Div", {"m", "x"}, {"d"}, {}));
    put_sub(body, 1, encode_node("sub0", "Sub", {"d", "m"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 3, 4, 4}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("new_elem.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 4U);
    EXPECT_EQ(graph.nodes[0].op_type, "Sigmoid");
    EXPECT_EQ(graph.nodes[1].op_type, "Mul");
    EXPECT_EQ(graph.nodes[2].op_type, "Div");
    EXPECT_EQ(graph.nodes[3].op_type, "Sub");
    for (const engine::Node& node : graph.nodes) {
        EXPECT_TRUE(node.attributes.empty()) << node.op_type;
    }
}

// Hermetic positive: Split carries axis plus a split-sizes input, Concat
// carries axis over two inputs.
TEST_F(EngineLoader, HermeticConcatSplitLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("split", {2}, {16, 16}));
    put_sub(body, 1, encode_node("split0", "Split", {"x", "split"}, {"a", "b"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 1, encode_node("concat0", "Concat", {"a", "b"}, {"y"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 32, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 32, 4, 4}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("concat_split.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "Split");
    ASSERT_EQ(graph.nodes[0].attributes.size(), 1U);
    EXPECT_EQ(
        std::get<std::int64_t>(graph.nodes[0].attributes.front().value), 1);
    EXPECT_EQ(graph.nodes[1].op_type, "Concat");
    ASSERT_EQ(graph.nodes[1].attributes.size(), 1U);
    EXPECT_EQ(
        std::get<std::int64_t>(graph.nodes[1].attributes.front().value), 1);
}

// Hermetic positive: Unsqueeze/Expand take no attributes; their axes/shape
// tables arrive as inputs.
TEST_F(EngineLoader, HermeticUnsqueezeExpandLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {-1}));
    put_sub(body, 5, encode_initializer_int64("eshape", {3}, {1, 2, 2}));
    put_sub(body, 1, encode_node("unsq0", "Unsqueeze", {"x", "axes"}, {"u"},
                      {}));
    put_sub(body, 1, encode_node("exp0", "Expand", {"u", "eshape"}, {"y"},
                      {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 2}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("unsq_expand.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "Unsqueeze");
    EXPECT_TRUE(graph.nodes[0].attributes.empty());
    EXPECT_EQ(graph.nodes[1].op_type, "Expand");
    EXPECT_TRUE(graph.nodes[1].attributes.empty());
}

// Hermetic positive: Transpose carries perm; Gather/GatherElements carry axis
// with indices as inputs.
TEST_F(EngineLoader, HermeticTransposeGatherLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("idx", {2}, {0, 1}));
    put_sub(body, 1, encode_node("tr0", "Transpose", {"x"}, {"t"},
                      {encode_attribute_ints("perm", {0, 2, 1})}));
    put_sub(body, 1, encode_node("g0", "Gather", {"t", "idx"}, {"g"},
                      {encode_attribute_int("axis", 0)}));
    put_sub(body, 1, encode_node("ge0", "GatherElements", {"t", "idx"}, {"h"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("h", 1, {2, 3, 4}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("tr_gather.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 3U);
    EXPECT_EQ(graph.nodes[0].op_type, "Transpose");
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(
                  graph.nodes[0].attributes.front().value),
        (std::vector<std::int64_t>{0, 2, 1}));
    EXPECT_EQ(graph.nodes[1].op_type, "Gather");
    EXPECT_EQ(graph.nodes[2].op_type, "GatherElements");
}

// Hermetic positive: Cast pins `to` to FLOAT (1) and INT64 (7).
TEST_F(EngineLoader, HermeticCastLoads)
{
    std::string body;
    put_sub(body, 1, encode_node("cast0", "Cast", {"x"}, {"c"},
                      {encode_attribute_int("to", 7)}));
    put_sub(body, 1, encode_node("cast1", "Cast", {"c"}, {"y"},
                      {encode_attribute_int("to", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {4}));
    put_sub(body, 12, encode_value_info("y", 1, {4}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("cast.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(std::get<std::int64_t>(graph.nodes[0].attributes.front().value),
        7);
    EXPECT_EQ(std::get<std::int64_t>(graph.nodes[1].attributes.front().value),
        1);
}

// Hermetic positive: Softmax carries axis; Resize carries the surveyed nearest
// policy (mode/asymmetric/floor/-0.75) with scales as third input.
TEST_F(EngineLoader, HermeticSoftmaxResizeLoads)
{
    std::string body;
    put_sub(body, 5,
        encode_initializer("scales", {4}, {1.0F, 1.0F, 2.0F, 2.0F}));
    put_sub(body, 1, encode_node("sm0", "Softmax", {"x"}, {"s"},
                      {encode_attribute_int("axis", -1)}));
    put_sub(body, 1, encode_node("rs0", "Resize", {"x", "", "scales"}, {"y"},
                      {encode_attribute_string("mode", "nearest"),
                          encode_attribute_string(
                              "coordinate_transformation_mode", "asymmetric"),
                          encode_attribute_string("nearest_mode", "floor"),
                          encode_attribute_float("cubic_coeff_a", -0.75F)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 1, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 1, 8, 8}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("sm_resize.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "Softmax");
    EXPECT_EQ(graph.nodes[1].op_type, "Resize");
    EXPECT_EQ(graph.nodes[1].inputs,
        (std::vector<std::string>{"x", "", "scales"}));
    ASSERT_EQ(graph.nodes[1].attributes.size(), 4U);
}

// Hermetic positive: Slice takes no attributes (starts/ends/axes as inputs);
// TopK takes axis/largest/sorted with K as an Int64 constant input.
TEST_F(EngineLoader, HermeticSliceTopKLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("starts", {1}, {0}));
    put_sub(body, 5, encode_initializer_int64("ends", {1}, {5}));
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {1}));
    put_sub(body, 5, encode_initializer_int64("k", {1}, {3}));
    put_sub(body, 1, encode_node("sl0", "Slice",
                      {"x", "starts", "ends", "axes"}, {"s"}, {}));
    put_sub(body, 1, encode_node("tk0", "TopK", {"s", "k"}, {"v", "i"},
                      {encode_attribute_int("axis", -1),
                          encode_attribute_int("largest", 1),
                          encode_attribute_int("sorted", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 10}));
    put_sub(body, 12, encode_value_info("v", 1, {1, 3}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("slice_topk.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "Slice");
    EXPECT_TRUE(graph.nodes[0].attributes.empty());
    EXPECT_EQ(graph.nodes[1].op_type, "TopK");
    ASSERT_EQ(graph.nodes[1].attributes.size(), 3U);
}

// Hermetic positive: ConstantOfShape carries an INT64 tensor `value`; Shape,
// Equal, and Where take no attributes.
TEST_F(EngineLoader, HermeticConstantOfShapeEqualWhereShapeLoads)
{
    const std::vector<int64_t> fill{1};
    std::string body;
    put_sub(body, 5, encode_initializer_int64("cshape", {1}, {3}));
    put_sub(body, 1, encode_node("cos0", "ConstantOfShape", {"cshape"},
                      {"fill"},
                      {encode_attribute_tensor("value",
                          encode_constant_tensor(7 /*INT64*/, {1},
                              raw_bytes_int64(fill)))}));
    put_sub(body, 1, encode_node("sh0", "Shape", {"x"}, {"r"}, {}));
    put_sub(body, 1, encode_node("eq0", "Equal", {"x", "y"}, {"e"}, {}));
    put_sub(body, 1, encode_node("w0", "Where", {"e", "x", "y"}, {"z"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 11, encode_value_info("y", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("z", 1, {2, 3}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("cos_eq_where.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 4U);
    EXPECT_EQ(graph.nodes[0].op_type, "ConstantOfShape");
    ASSERT_EQ(graph.nodes[0].attributes.size(), 1U);
    EXPECT_EQ(std::get<std::int64_t>(graph.nodes[0].attributes.front().value),
        1);
    EXPECT_EQ(graph.nodes[1].op_type, "Shape");
    EXPECT_EQ(graph.nodes[2].op_type, "Equal");
    EXPECT_EQ(graph.nodes[3].op_type, "Where");
}

// Hermetic positive: ReduceMax carries keepdims with axes as input; Mod
// carries fmod=0.
TEST_F(EngineLoader, HermeticReduceMaxModLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {-1}));
    put_sub(body, 1, encode_node("rm0", "ReduceMax", {"x", "axes"}, {"r"},
                      {encode_attribute_int("keepdims", 0)}));
    put_sub(body, 1, encode_node("mod0", "Mod", {"x", "x"}, {"y"},
                      {encode_attribute_int("fmod", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("rm_mod.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "ReduceMax");
    EXPECT_EQ(graph.nodes[1].op_type, "Mod");
}

// Hermetic positive: Conv admits kernel_shape (carried by the detection
// fixture on every Conv); shape inference derives dims from the weight.
TEST_F(EngineLoader, HermeticConvKernelShapeAdmitted)
{
    std::string body;
    put_sub(body, 5, encode_initializer("w", {4, 3, 3, 3},
                       std::vector<float>(4 * 3 * 3 * 3, 0.0F)));
    put_sub(body, 1, encode_node("conv0", "Conv", {"x", "w"}, {"y"},
                      {encode_attribute_ints("kernel_shape", {3, 3}),
                          encode_attribute_ints("strides", {1, 1}),
                          encode_attribute_ints("pads", {1, 1, 1, 1})}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 8, 8}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 4, 8, 8}));

    const engine::Graph graph = engine::LoadGraphFromFile(
        write_model_bytes("conv_ks.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Conv");
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

// Non-FP32 I/O: BOOL graph input and INT64/BOOL graph outputs are rejected;
// graph I/O stays float32-only while bool intermediates are allowed.
TEST_F(EngineLoaderNegative, BoolGraphInputRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("probe_bool", "Relu",
        {"x_bool"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x_bool", 9 /*BOOL*/, {2, 2}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 2}));

    const std::string message =
        load_error_message(encode_model(body), "bool_in.onnx");
    EXPECT_NE(message.find("x_bool"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("BOOL"), std::string::npos) << "message: " << message;
}

TEST_F(EngineLoaderNegative, NonFloatGraphOutputRejected)
{
    for (const auto& [file, elem, tag] :
        {std::tuple<std::string, int64_t, std::string>{
                 "int_out.onnx", 7, "INT64"},
            std::tuple<std::string, int64_t, std::string>{
                "bool_out.onnx", 9, "BOOL"}}) {
        std::string body;
        put_sub(body, 1, encode_node("probe_out", "Relu",
            {"x"}, {"y_out"}, {}));
        put_sub(body, 11, encode_value_info("x", 1, {2, 2}));
        put_sub(body, 12, encode_value_info("y_out", elem, {2, 2}));

        const std::string message = load_error_message(encode_model(body), file);
        EXPECT_NE(message.find("y_out"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find(tag), std::string::npos)
            << "message: " << message;
    }
}

// A Constant of any dtype outside FLOAT/INT64 is a load rejection naming the
// node, the op type, and the element type.
TEST_F(EngineLoaderNegative, ConstantOtherDtypeRejected)
{
    for (const auto& [file, elem, tag] :
        {std::tuple<std::string, int64_t, std::string>{
                 "const_i32.onnx", 6, "INT32"},
            std::tuple<std::string, int64_t, std::string>{
                "const_bool.onnx", 9, "BOOL"}}) {
        const std::string raw(4, '\x01');
        std::string body;
        put_sub(body, 1,
            encode_constant_node("const_bad", "c_bad",
                encode_constant_tensor(elem, {4}, raw)));
        put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
        put_sub(body, 11, encode_value_info("x", 1, {4}));
        put_sub(body, 12, encode_value_info("y", 1, {4}));

        const std::string message =
            load_error_message(encode_model(body), file);
        EXPECT_NE(message.find("const_bad"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("Constant"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find(tag), std::string::npos)
            << "message: " << message;
    }
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

// Cast `to` outside {FLOAT, INT64} rejects with node + op context.

TEST_F(EngineLoaderNegative, CastToOutsidePolicyRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("cast_bad", "Cast", {"x"}, {"y"},
        {encode_attribute_int("to", 6)})); // INT32
    put_sub(body, 11, encode_value_info("x", 1, {4}));
    put_sub(body, 12, encode_value_info("y", 1, {4}));

    const std::string message =
        load_error_message(encode_model(body), "cast_to.onnx");
    EXPECT_NE(message.find("cast_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Cast"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("to"), std::string::npos) << "message: " << message;
}

// Resize outside the surveyed nearest policy rejects with node + op context.

TEST_F(EngineLoaderNegative, ResizeModeOutsidePolicyRejected)
{
    for (const auto& [file, attr] :
        {std::tuple<std::string, std::string>{"rs_mode.onnx",
             encode_attribute_string("mode", "linear")},
            std::tuple<std::string, std::string>{"rs_nm.onnx",
                encode_attribute_string("nearest_mode", "round_prefer_floor")},
            std::tuple<std::string, std::string>{"rs_ctm.onnx",
                encode_attribute_string(
                    "coordinate_transformation_mode", "half_pixel")}}) {
        std::string body;
        put_sub(body, 5,
            encode_initializer("scales", {4}, {1.0F, 1.0F, 2.0F, 2.0F}));
        put_sub(body, 1, encode_node("rs_bad", "Resize", {"x", "", "scales"},
                          {"y"}, {attr}));
        put_sub(body, 11, encode_value_info("x", 1, {1, 1, 4, 4}));
        put_sub(body, 12, encode_value_info("y", 1, {1, 1, 8, 8}));

        const std::string message =
            load_error_message(encode_model(body), file);
        EXPECT_NE(message.find("rs_bad"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("Resize"), std::string::npos)
            << "message: " << message;
    }
}

// Mod with fmod=1 (float remainder) rejects; the fixture pins fmod=0.

TEST_F(EngineLoaderNegative, ModFmodOutsidePolicyRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("mod_bad", "Mod", {"x", "x"}, {"y"},
        {encode_attribute_int("fmod", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {4}));
    put_sub(body, 12, encode_value_info("y", 1, {4}));

    const std::string message =
        load_error_message(encode_model(body), "mod_fmod.onnx");
    EXPECT_NE(message.find("mod_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Mod"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("fmod"), std::string::npos) << "message: " << message;
}

// TopK outside largest=1/sorted=1 rejects with node + op context.

TEST_F(EngineLoaderNegative, TopKPolicyOutsidePolicyRejected)
{
    const std::vector<std::string> largest_zero{encode_attribute_int("axis", -1),
        encode_attribute_int("largest", 0), encode_attribute_int("sorted", 1)};
    const std::vector<std::string> sorted_zero{encode_attribute_int("axis", -1),
        encode_attribute_int("largest", 1), encode_attribute_int("sorted", 0)};
    for (const auto& [file, attrs] :
        {std::tuple<std::string, std::vector<std::string>>{
             "tk_l.onnx", largest_zero},
            std::tuple<std::string, std::vector<std::string>>{
                "tk_s.onnx", sorted_zero}}) {
        std::string body;
        put_sub(body, 5, encode_initializer_int64("k", {1}, {2}));
        put_sub(body, 1,
            encode_node("tk_bad", "TopK", {"x", "k"}, {"v", "i"}, attrs));
        put_sub(body, 11, encode_value_info("x", 1, {2, 5}));
        put_sub(body, 12, encode_value_info("v", 1, {2, 2}));

        const std::string message =
            load_error_message(encode_model(body), file);
        EXPECT_NE(message.find("tk_bad"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("TopK"), std::string::npos)
            << "message: " << message;
    }
}

// TopK whose K is not an Int64 constant is a load rejection, never a dynamic
// shape: here K is a graph input, so no initializer provides it.

TEST_F(EngineLoaderNegative, TopKNonConstantKRejectedAtLoad)
{
    std::string body;
    put_sub(body, 1, encode_node("tk_dyn", "TopK", {"x", "k"}, {"v", "i"},
                      {encode_attribute_int("axis", -1),
                          encode_attribute_int("largest", 1),
                          encode_attribute_int("sorted", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 5}));
    put_sub(body, 11, encode_value_info("k", 1, {1}));
    put_sub(body, 12, encode_value_info("v", 1, {2, 2}));

    const std::string message =
        load_error_message(encode_model(body), "topk_dyn_k.onnx");
    EXPECT_NE(message.find("tk_dyn"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("TopK"), std::string::npos) << "message: " << message;
}

// ConstantOfShape whose `value` is not an INT64 tensor rejects.

TEST_F(EngineLoaderNegative, ConstantOfShapeFloatValueRejected)
{
    const std::vector<float> fill{0.0F};
    std::string body;
    put_sub(body, 5, encode_initializer_int64("cshape", {1}, {3}));
    put_sub(body, 1, encode_node("cos_bad", "ConstantOfShape", {"cshape"},
                      {"fill"},
                      {encode_attribute_tensor("value",
                          encode_constant_tensor(1 /*FLOAT*/, {1},
                              raw_bytes_float(fill)))}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body), "cos_float.onnx");
    EXPECT_NE(message.find("cos_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("ConstantOfShape"), std::string::npos)
        << "message: " << message;
}

// Slice takes no attributes at opset 18 (starts/ends/axes/steps are inputs).

TEST_F(EngineLoaderNegative, SliceAttributeRejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("starts", {1}, {0}));
    put_sub(body, 5, encode_initializer_int64("ends", {1}, {5}));
    put_sub(body, 1, encode_node("sl_bad", "Slice", {"x", "starts", "ends"},
                      {"y"}, {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 10}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 5}));

    const std::string message =
        load_error_message(encode_model(body), "slice_attr.onnx");
    EXPECT_NE(message.find("sl_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Slice"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("axis"), std::string::npos) << "message: " << message;
}

// A no-attribute new op (Mul) still rejects any attribute with full context.

TEST_F(EngineLoaderNegative, UnsupportedAttributeOnNewOp)
{
    std::string body;
    put_sub(body, 1, encode_node("mul_foo", "Mul", {"x", "x"}, {"y"},
        {encode_attribute_name_only("alpha")}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body), "mul_attr.onnx");
    EXPECT_NE(message.find("mul_foo"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Mul"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("alpha"), std::string::npos) << "message: " << message;
}
