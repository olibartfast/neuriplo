// Static shape-inference tests.
//
// Two suites, split for ctest filters (suite names must not carry the ctest
// NAME colons, so the filters live in the add_test COMMAND):
//   EngineShapes.*           — positives: hermetic hand-encoded graphs plus the
//                              real ResNet-18 fixture when it is resolvable
//   EngineShapesNegative.*   — rejection classes with node/op message asserts
//
// The hermetic cases are hand-encoded model bytes and reuse the protobuf wire
// encoders from the loader tests, so they exercise the real loader -> IR ->
// shape-inference path. The fixture cases run on the opset-18 ResNet-18 export
// when it is resolvable via NEURIPLO_NATIVE_FIXTURE (env var or CMake cache
// variable), and a missing fixture FAILS the test — never skipped or mocked.
// The per-shape seam (same loaded Graph, two input shapes, no reload) is proven
// hermetically on a hand-encoded Relu graph; the fixture additionally asserts
// that its batch-1-pinned Reshape rejects a batch-2 inference.

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
#include "engine/Shapes.hpp"

namespace {

// ---------------------------------------------------------------------------
// Hand-rolled protobuf wire encoders (test-side twin of the loader's reader).
// Field numbers mirror ModelLoader.cpp; no public header carries any.
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

std::string encode_opset18()
{
    std::string leaving; // domain (field 1) omitted; default ONNX domain
    put_varint_field(leaving, 2, 18);
    return leaving;
}

std::string encode_value_info(const std::string& name, int64_t elem_type,
    const std::vector<int64_t>& dims)
{
    std::string vi;
    put_length_delimited(vi, 1, name);
    std::string tensor_type;
    put_varint_field(tensor_type, 1, static_cast<uint64_t>(elem_type));
    std::string shape;
    for (const int64_t dim : dims) {
        std::string dim_msg;
        put_varint_field(dim_msg, 1, static_cast<uint64_t>(dim));
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

void put_fixed32_field(std::string& out, int field_number, uint32_t bits)
{
    put_tag(out, field_number, 5);
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((bits >> (8 * i)) & 0xFFU));
    }
}

std::string encode_attribute_float(const std::string& name, float value)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_fixed32_field(a, 2, *reinterpret_cast<const uint32_t*>(&value));
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

std::string encode_initializer_int64(const std::string& name,
    const std::vector<int64_t>& dims, const std::vector<int64_t>& values)
{
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 7); // INT64
    put_length_delimited(t, 8, name);
    const std::string raw(reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(int64_t));
    put_length_delimited(t, 9, raw);
    return t;
}

// Embedded TensorProto payload for a ConstantOfShape 'value' attribute.
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

std::string raw_bytes_int64(const std::vector<int64_t>& values)
{
    return std::string(reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(int64_t));
}

// A Tensor-typed node attribute (AttributeProto type TENSOR = 4, tensor = 5).
std::string encode_attribute_tensor(const std::string& name,
    const std::string& tensor_bytes)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_sub(a, 5, tensor_bytes);
    put_varint_field(a, 20, 4); // TENSOR
    return a;
}

std::string encode_model(const std::string& graph_body)
{
    std::string model;
    put_sub(model, 8, encode_opset18());
    put_sub(model, 7, graph_body);
    return model;
}

std::string write_model_bytes(const std::string& name, const std::string& bytes)
{
    const std::string dir =
        (std::filesystem::temp_directory_path() / "engine_shapes_tests")
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

engine::Graph load_graph(const std::string& name, const std::string& bytes)
{
    return engine::LoadGraphFromFile(write_model_bytes(name, bytes));
}

std::vector<float> zeros(size_t count)
{
    return std::vector<float>(count, 0.0F);
}

std::string first_output_of(const engine::Graph& graph,
    const std::string& op_type)
{
    for (const engine::Node& node : graph.nodes) {
        if (node.op_type == op_type && !node.outputs.empty()) {
            return node.outputs.front();
        }
    }
    return {};
}

const char* fixture_env_name() { return "NEURIPLO_NATIVE_FIXTURE"; }

} // namespace

// ===========================================================================
// EngineShapes.* — positive cases
// ===========================================================================

class EngineShapes : public ::testing::Test {
protected:
    static std::string fixture_path()
    {
        const char* from_env = std::getenv(fixture_env_name());
        return from_env != nullptr ? std::string(from_env) : std::string();
    }
};

// Add: NumPy multidirectional broadcast of [1,3,4,4] and [3,1,1].

TEST_F(EngineShapes, AddBroadcast)
{
    std::string body;
    put_sub(body, 1, encode_node("add0", "Add", {"x", "y"}, {"z"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 11, encode_value_info("y", 1, {3, 1, 1}));
    put_sub(body, 12, encode_value_info("z", 1, {1, 3, 4, 4}));

    const engine::Graph graph =
        load_graph("add_broadcast.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {1, 3, 4, 4}}, {"y", {3, 1, 1}}});
    ASSERT_EQ(shapes.tensors.count("z"), 1U);
    EXPECT_EQ(shapes.tensors.at("z").dims, (std::vector<int64_t>{1, 3, 4, 4}));
    EXPECT_EQ(shapes.tensors.at("z").dtype, engine::DataType::Float32);
}

// Relu: output shape equals input shape.

TEST_F(EngineShapes, ReluCopiesInputShape)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 5, 6, 7}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 5, 6, 7}));

    const engine::Graph graph = load_graph("relu_copy.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 5, 6, 7}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 5, 6, 7}));
}

// MatMul: batched product plus 1-D operand promotion.

TEST_F(EngineShapes, MatMulBroadcastAndPromotion)
{
    {
        std::string body;
        put_sub(body, 1, encode_node("mm0", "MatMul", {"a", "b"}, {"y"}, {}));
        put_sub(body, 11, encode_value_info("a", 1, {2, 3, 4}));
        put_sub(body, 11, encode_value_info("b", 1, {4, 5}));
        put_sub(body, 12, encode_value_info("y", 1, {2, 3, 5}));
        const engine::Graph graph =
            load_graph("matmul_batch.onnx", encode_model(body));
        const engine::InferredShapes shapes =
            engine::InferShapes(graph, {{"a", {2, 3, 4}}, {"b", {4, 5}}});
        ASSERT_EQ(shapes.tensors.count("y"), 1U);
        EXPECT_EQ(shapes.tensors.at("y").dims,
            (std::vector<int64_t>{2, 3, 5}));
    }
    {
        // 1-D lhs and 1-D rhs promote to a scalar result.
        std::string body;
        put_sub(body, 1, encode_node("mm1", "MatMul", {"a", "b"}, {"y"}, {}));
        put_sub(body, 11, encode_value_info("a", 1, {4}));
        put_sub(body, 11, encode_value_info("b", 1, {4}));
        put_sub(body, 12, encode_value_info("y", 1, {}));
        const engine::Graph graph =
            load_graph("matmul_vec.onnx", encode_model(body));
        const engine::InferredShapes shapes =
            engine::InferShapes(graph, {{"a", {4}}, {"b", {4}}});
        ASSERT_EQ(shapes.tensors.count("y"), 1U);
        EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{}));
    }
    {
        // 1-D rhs: the promoted trailing dim is squeezed from the result.
        std::string body;
        put_sub(body, 1, encode_node("mm2", "MatMul", {"a", "b"}, {"y"}, {}));
        put_sub(body, 11, encode_value_info("a", 1, {2, 3}));
        put_sub(body, 11, encode_value_info("b", 1, {3}));
        put_sub(body, 12, encode_value_info("y", 1, {2}));
        const engine::Graph graph =
            load_graph("matmul_rhs_vec.onnx", encode_model(body));
        const engine::InferredShapes shapes =
            engine::InferShapes(graph, {{"a", {2, 3}}, {"b", {3}}});
        ASSERT_EQ(shapes.tensors.count("y"), 1U);
        EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2}));
    }
}

// Reshape: 0 copies the input dim and -1 is inferred from the element count.

TEST_F(EngineShapes, ReshapeZeroAndInferred)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("shape", {2}, {0, -1}));
    put_sub(body, 1, encode_node("reshape0", "Reshape", {"x", "shape"}, {"y"},
                      {encode_attribute_int("allowzero", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 48}));

    const engine::Graph graph =
        load_graph("reshape_zero.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {1, 3, 4, 4}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{1, 48}));
}

// Reshape: a target that increases rank reshapes without copying batch.

TEST_F(EngineShapes, ReshapeIncreasesRank)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("shape", {3}, {1, 2, 3}));
    put_sub(body, 1, encode_node("reshape0", "Reshape", {"x", "shape"}, {"y"},
                      {encode_attribute_int("allowzero", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3}));

    const engine::Graph graph =
        load_graph("reshape_rank.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{1, 2, 3}));
}

// ReduceMean: keepdims=0 drops the reduced axes; keepdims=1 sets them to 1.

TEST_F(EngineShapes, ReduceMeanKeepdimsOn)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {2}, {-1, -2}));
    put_sub(body, 1, encode_node("mean0", "ReduceMean", {"x", "axes"}, {"y"},
                      {encode_attribute_int("keepdims", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3, 1, 1}));

    const engine::Graph graph =
        load_graph("mean_keep.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3, 4, 4}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 3, 1, 1}));
}

TEST_F(EngineShapes, ReduceMeanKeepdimsOff)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {2}, {-1, -2}));
    put_sub(body, 1, encode_node("mean0", "ReduceMean", {"x", "axes"}, {"y"},
                      {encode_attribute_int("keepdims", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

    const engine::Graph graph =
        load_graph("mean_drop.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3, 4, 4}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 3}));
}

// Conv: non-default stride and asymmetric pads.

TEST_F(EngineShapes, ConvStrideAndPad)
{
    std::string body;
    put_sub(body, 5, encode_initializer("w", {4, 3, 3, 3}, zeros(4 * 3 * 3 * 3)));
    put_sub(body, 1, encode_node("conv0", "Conv", {"x", "w"}, {"y"},
                      {encode_attribute_ints("strides", {2, 2}),
                          encode_attribute_ints("pads", {1, 1, 1, 1})}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 8, 8}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 4, 4, 4}));

    const engine::Graph graph = load_graph("conv_stride.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {1, 3, 8, 8}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{1, 4, 4, 4}));
}

// Conv: SAME_UPPER derives ceil(in/stride) output with asymmetric pads.

TEST_F(EngineShapes, ConvSameUpper)
{
    std::string body;
    put_sub(body, 5,
        encode_initializer("w", {2, 1, 3, 3}, zeros(2 * 1 * 3 * 3)));
    put_sub(body, 1, encode_node("conv0", "Conv", {"x", "w"}, {"y"},
                      {encode_attribute_ints("strides", {2, 2}),
                          encode_attribute_string("auto_pad", "SAME_UPPER")}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 1, 7, 7}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 4, 4}));

    const engine::Graph graph =
        load_graph("conv_same.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {1, 1, 7, 7}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{1, 2, 4, 4}));
}

// MaxPool: ceil_mode rounds the output up rather than down.

TEST_F(EngineShapes, MaxPoolCeilMode)
{
    std::string body;
    put_sub(body, 1, encode_node("pool0", "MaxPool", {"x"}, {"y"},
                      {encode_attribute_ints("kernel_shape", {2, 2}),
                          encode_attribute_ints("strides", {2, 2}),
                          encode_attribute_int("ceil_mode", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2, 5, 5}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3, 3}));

    const engine::Graph graph = load_graph("pool_ceil.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {1, 2, 5, 5}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{1, 2, 3, 3}));
}

// MaxPool: the default floor behavior on the same window.

TEST_F(EngineShapes, MaxPoolFloorDefault)
{
    std::string body;
    put_sub(body, 1, encode_node("pool0", "MaxPool", {"x"}, {"y"},
                      {encode_attribute_ints("kernel_shape", {2, 2}),
                          encode_attribute_ints("strides", {2, 2})}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2, 5, 5}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 2, 2}));

    const engine::Graph graph = load_graph("pool_floor.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {1, 2, 5, 5}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{1, 2, 2, 2}));
}

// Gemm: transA/transB select which physical dims are rows and columns.

TEST_F(EngineShapes, GemmTransposes)
{
    std::string body;
    put_sub(body, 5, encode_initializer("b", {4, 3}, zeros(4 * 3)));
    put_sub(body, 1, encode_node("gemm0", "Gemm", {"a", "b"}, {"y"},
                      {encode_attribute_int("transA", 1),
                          encode_attribute_int("transB", 1)}));
    put_sub(body, 11, encode_value_info("a", 1, {3, 2}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 4}));

    const engine::Graph graph = load_graph("gemm_t.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"a", {3, 2}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 4}));
}

// Gemm: default transposes (both 0).

TEST_F(EngineShapes, GemmDefaults)
{
    std::string body;
    put_sub(body, 5, encode_initializer("b", {3, 4}, zeros(3 * 4)));
    put_sub(body, 1, encode_node("gemm0", "Gemm", {"a", "b"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("a", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 4}));

    const engine::Graph graph =
        load_graph("gemm_default.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"a", {2, 3}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 4}));
}

// Mul/Div/Sub/Mod: NumPy multidirectional broadcast of [1,3,4,4] and [3,1,1].

TEST_F(EngineShapes, NewElementwiseBroadcast)
{
    std::string body;
    put_sub(body, 1, encode_node("mul0", "Mul", {"x", "y"}, {"m"}, {}));
    put_sub(body, 1, encode_node("div0", "Div", {"m", "y"}, {"d"}, {}));
    put_sub(body, 1, encode_node("sub0", "Sub", {"d", "m"}, {"s"}, {}));
    put_sub(body, 1, encode_node("mod0", "Mod", {"s", "y"}, {"z"},
                      {encode_attribute_int("fmod", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 11, encode_value_info("y", 1, {3, 1, 1}));
    put_sub(body, 12, encode_value_info("z", 1, {1, 3, 4, 4}));

    const engine::Graph graph =
        load_graph("new_elem.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {1, 3, 4, 4}}, {"y", {3, 1, 1}}});
    for (const std::string& name : {"m", "d", "s", "z"}) {
        ASSERT_EQ(shapes.tensors.count(name), 1U) << name;
        EXPECT_EQ(shapes.tensors.at(name).dims,
            (std::vector<int64_t>{1, 3, 4, 4}))
            << name;
        EXPECT_EQ(shapes.tensors.at(name).dtype, engine::DataType::Float32)
            << name;
    }
}

// Mod: int64 inputs propagate int64 to the output (Y2-repair: dtype follows
// the first input; the Mod kernel is int64-only).

TEST_F(EngineShapes, ModInt64PropagatesDtype)
{
    std::string body;
    put_sub(body, 1, encode_node("c0", "Cast", {"x"}, {"a"},
                      {encode_attribute_int("to", 7)}));
    put_sub(body, 5, encode_initializer_int64("k", {3}, {2, 3, 4}));
    put_sub(body, 1, encode_node("mod0", "Mod", {"a", "k"}, {"z"},
                      {encode_attribute_int("fmod", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("z", 1, {2, 3}));

    const engine::Graph graph =
        load_graph("mod_int64.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3}}});
    ASSERT_EQ(shapes.tensors.count("z"), 1U);
    EXPECT_EQ(shapes.tensors.at("z").dims, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(shapes.tensors.at("z").dtype, engine::DataType::Int64);
}

// Sigmoid/Softmax: output shape equals input shape.

TEST_F(EngineShapes, SigmoidSoftmaxCopyInputShape)
{
    std::string body;
    put_sub(body, 1, encode_node("sm0", "Softmax", {"x"}, {"s"},
                      {encode_attribute_int("axis", -1)}));
    put_sub(body, 1, encode_node("sig0", "Sigmoid", {"s"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 5}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 5}));

    const engine::Graph graph =
        load_graph("sig_sm.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 5}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 5}));
    EXPECT_EQ(shapes.tensors.at("s").dims, (std::vector<int64_t>{2, 5}));
}

// Concat: dims sum along axis, including a negative axis.

TEST_F(EngineShapes, ConcatAxis)
{
    {
        std::string body;
        put_sub(body, 1, encode_node("c0", "Concat", {"a", "b"}, {"y"},
                          {encode_attribute_int("axis", 1)}));
        put_sub(body, 11, encode_value_info("a", 1, {1, 2, 4}));
        put_sub(body, 11, encode_value_info("b", 1, {1, 3, 4}));
        put_sub(body, 12, encode_value_info("y", 1, {1, 5, 4}));
        const engine::Graph graph =
            load_graph("concat1.onnx", encode_model(body));
        const engine::InferredShapes shapes = engine::InferShapes(graph,
            {{"a", {1, 2, 4}}, {"b", {1, 3, 4}}});
        ASSERT_EQ(shapes.tensors.count("y"), 1U);
        EXPECT_EQ(shapes.tensors.at("y").dims,
            (std::vector<int64_t>{1, 5, 4}));
    }
    {
        std::string body;
        put_sub(body, 1, encode_node("c1", "Concat", {"a", "b"}, {"y"},
                          {encode_attribute_int("axis", -1)}));
        put_sub(body, 11, encode_value_info("a", 1, {2, 3}));
        put_sub(body, 11, encode_value_info("b", 1, {2, 5}));
        put_sub(body, 12, encode_value_info("y", 1, {2, 8}));
        const engine::Graph graph =
            load_graph("concat_neg.onnx", encode_model(body));
        const engine::InferredShapes shapes =
            engine::InferShapes(graph, {{"a", {2, 3}}, {"b", {2, 5}}});
        ASSERT_EQ(shapes.tensors.count("y"), 1U);
        EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 8}));
    }
}

// Split: explicit sizes from an Int64 constant, or an even split without one.

TEST_F(EngineShapes, SplitSizedAndEqual)
{
    {
        std::string body;
        put_sub(body, 5, encode_initializer_int64("split", {2}, {2, 3}));
        put_sub(body, 1, encode_node("sp0", "Split", {"x", "split"},
                          {"a", "b"}, {encode_attribute_int("axis", 1)}));
        put_sub(body, 11, encode_value_info("x", 1, {1, 5}));
        put_sub(body, 12, encode_value_info("a", 1, {1, 2}));
        const engine::Graph graph =
            load_graph("split_sized.onnx", encode_model(body));
        const engine::InferredShapes shapes =
            engine::InferShapes(graph, {{"x", {1, 5}}});
        ASSERT_EQ(shapes.tensors.count("a"), 1U);
        EXPECT_EQ(shapes.tensors.at("a").dims, (std::vector<int64_t>{1, 2}));
        ASSERT_EQ(shapes.tensors.count("b"), 1U);
        EXPECT_EQ(shapes.tensors.at("b").dims, (std::vector<int64_t>{1, 3}));
    }
    {
        std::string body;
        put_sub(body, 1, encode_node("sp1", "Split", {"x"}, {"a", "b"},
                          {encode_attribute_int("axis", 1)}));
        put_sub(body, 11, encode_value_info("x", 1, {1, 6}));
        put_sub(body, 12, encode_value_info("a", 1, {1, 3}));
        const engine::Graph graph =
            load_graph("split_equal.onnx", encode_model(body));
        const engine::InferredShapes shapes =
            engine::InferShapes(graph, {{"x", {1, 6}}});
        EXPECT_EQ(shapes.tensors.at("a").dims, (std::vector<int64_t>{1, 3}));
        EXPECT_EQ(shapes.tensors.at("b").dims, (std::vector<int64_t>{1, 3}));
    }
}

// Unsqueeze: ones inserted at the axes positions ([2,3] + [0,-1] -> [1,2,3,1]).

TEST_F(EngineShapes, UnsqueezeInsertsOnes)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {2}, {0, -1}));
    put_sub(body, 1, encode_node("u0", "Unsqueeze", {"x", "axes"}, {"y"},
                      {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3, 1}));

    const engine::Graph graph =
        load_graph("unsqueeze.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims,
        (std::vector<int64_t>{1, 2, 3, 1}));
}

// Expand: [3,1] broadcasts to the [2,3,4] target shape.

TEST_F(EngineShapes, ExpandBroadcastsToShape)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("eshape", {3}, {2, 3, 4}));
    put_sub(body, 1, encode_node("e0", "Expand", {"x", "eshape"}, {"y"},
                      {}));
    put_sub(body, 11, encode_value_info("x", 1, {3, 1}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3, 4}));

    const engine::Graph graph = load_graph("expand.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {3, 1}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 3, 4}));
}

// Transpose: perm [0,2,1] maps [2,3,4] to [2,4,3].

TEST_F(EngineShapes, TransposePermutes)
{
    std::string body;
    put_sub(body, 1, encode_node("t0", "Transpose", {"x"}, {"y"},
                      {encode_attribute_ints("perm", {0, 2, 1})}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 4, 3}));

    const engine::Graph graph =
        load_graph("transpose.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3, 4}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 4, 3}));
}

// GatherElements: output shape equals the indices shape.

TEST_F(EngineShapes, GatherElementsMatchesIndices)
{
    std::string body;
    put_sub(body, 1, encode_node("ge0", "GatherElements", {"x", "idx"}, {"y"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 11, encode_value_info("idx", 1, {2, 1, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));

    const engine::Graph graph =
        load_graph("gelem.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3, 4}}, {"idx", {2, 1, 4}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 1, 4}));
    EXPECT_EQ(shapes.tensors.at("y").dtype, engine::DataType::Float32);
}

// Gather: [4,5,6] along axis 1 with [2] indices -> [4,2,6].

TEST_F(EngineShapes, GatherInsertsIndices)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("idx", {1}, {1}));
    put_sub(body, 1, encode_node("g0", "Gather", {"x", "idx"}, {"y"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {4, 5, 6}));
    put_sub(body, 12, encode_value_info("y", 1, {4, 1, 6}));

    const engine::Graph graph = load_graph("gather.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {4, 5, 6}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{4, 1, 6}));
    EXPECT_EQ(shapes.tensors.at("y").dtype, engine::DataType::Float32);
}

// Cast: same dims, dtype follows `to` (7 -> int64).

TEST_F(EngineShapes, CastChangesDtype)
{
    std::string body;
    put_sub(body, 1, encode_node("c0", "Cast", {"x"}, {"y"},
                      {encode_attribute_int("to", 7)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

    const engine::Graph graph = load_graph("cast.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(shapes.tensors.at("y").dtype, engine::DataType::Int64);
}

// Resize (nearest): [1,3,8,8] with scales [1,1,2,2] -> [1,3,16,16].

TEST_F(EngineShapes, ResizeScalesNearest)
{
    std::string body;
    put_sub(body, 5,
        encode_initializer("scales", {4}, {1.0F, 1.0F, 2.0F, 2.0F}));
    put_sub(body, 1, encode_node("rs0", "Resize", {"x", "", "scales"}, {"y"},
                      {encode_attribute_string("mode", "nearest")}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 8, 8}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 3, 16, 16}));

    const engine::Graph graph = load_graph("resize.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {1, 3, 8, 8}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims,
        (std::vector<int64_t>{1, 3, 16, 16}));
}

// Slice: [1,10] with starts [2], ends [7], axes [1] -> [1,5].

TEST_F(EngineShapes, SliceExtractsWindow)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("starts", {1}, {2}));
    put_sub(body, 5, encode_initializer_int64("ends", {1}, {7}));
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {1}));
    put_sub(body, 1, encode_node("sl0", "Slice",
                      {"x", "starts", "ends", "axes"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 10}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 5}));

    const engine::Graph graph = load_graph("slice.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {1, 10}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{1, 5}));
}

// TopK: constant scalar K replaces the axis dim on both outputs.

TEST_F(EngineShapes, TopKSelectsK)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("k", {1}, {2}));
    put_sub(body, 1, encode_node("tk0", "TopK", {"x", "k"}, {"v", "i"},
                      {encode_attribute_int("axis", -1),
                          encode_attribute_int("largest", 1),
                          encode_attribute_int("sorted", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 5}));
    put_sub(body, 12, encode_value_info("v", 1, {2, 2}));

    const engine::Graph graph = load_graph("topk.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 5}}});
    ASSERT_EQ(shapes.tensors.count("v"), 1U);
    EXPECT_EQ(shapes.tensors.at("v").dims, (std::vector<int64_t>{2, 2}));
    ASSERT_EQ(shapes.tensors.count("i"), 1U);
    EXPECT_EQ(shapes.tensors.at("i").dims, (std::vector<int64_t>{2, 2}));
}

// ConstantOfShape: the shape input's values become the int64 output dims.

TEST_F(EngineShapes, ConstantOfShapeUsesShapeInput)
{
    const std::vector<int64_t> fill{1};
    std::string body;
    put_sub(body, 5, encode_initializer_int64("cshape", {1}, {3}));
    put_sub(body, 1, encode_node("cos0", "ConstantOfShape", {"cshape"},
                      {"o"},
                      {encode_attribute_tensor("value",
                          encode_constant_tensor(7 /*INT64*/, {1},
                              raw_bytes_int64(fill)))}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("o", 1, {3}));

    const engine::Graph graph = load_graph("cos.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2}}});
    ASSERT_EQ(shapes.tensors.count("o"), 1U);
    EXPECT_EQ(shapes.tensors.at("o").dims, (std::vector<int64_t>{3}));
    EXPECT_EQ(shapes.tensors.at("o").dtype, engine::DataType::Int64);
}

// Equal: broadcast inputs, bool output.

TEST_F(EngineShapes, EqualBroadcastsToBool)
{
    std::string body;
    put_sub(body, 1, encode_node("eq0", "Equal", {"x", "y"}, {"e"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 11, encode_value_info("y", 1, {3, 1, 1}));
    put_sub(body, 12, encode_value_info("e", 1, {1, 3, 4, 4}));

    const engine::Graph graph = load_graph("equal.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {1, 3, 4, 4}}, {"y", {3, 1, 1}}});
    ASSERT_EQ(shapes.tensors.count("e"), 1U);
    EXPECT_EQ(shapes.tensors.at("e").dims,
        (std::vector<int64_t>{1, 3, 4, 4}));
    EXPECT_EQ(shapes.tensors.at("e").dtype, engine::DataType::Bool);
}

// Where: three-way broadcast of condition, X, and Y.

TEST_F(EngineShapes, WhereBroadcastsThree)
{
    std::string body;
    put_sub(body, 1, encode_node("w0", "Where", {"c", "x", "y"}, {"z"},
                      {}));
    put_sub(body, 11, encode_value_info("c", 1, {1, 3, 1}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 1, 4}));
    put_sub(body, 11, encode_value_info("y", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("z", 1, {2, 3, 4}));

    const engine::Graph graph = load_graph("where.onnx", encode_model(body));
    const engine::InferredShapes shapes = engine::InferShapes(graph,
        {{"c", {1, 3, 1}}, {"x", {2, 1, 4}}, {"y", {2, 3, 4}}});
    ASSERT_EQ(shapes.tensors.count("z"), 1U);
    EXPECT_EQ(shapes.tensors.at("z").dims, (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(shapes.tensors.at("z").dtype, engine::DataType::Float32);
}

// Shape: a rank-3 input yields int64 [3].

TEST_F(EngineShapes, ShapeOutputsRank)
{
    std::string body;
    put_sub(body, 1, encode_node("sh0", "Shape", {"x"}, {"r"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("r", 1, {3}));

    const engine::Graph graph = load_graph("shape.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3, 4}}});
    ASSERT_EQ(shapes.tensors.count("r"), 1U);
    EXPECT_EQ(shapes.tensors.at("r").dims, (std::vector<int64_t>{3}));
    EXPECT_EQ(shapes.tensors.at("r").dtype, engine::DataType::Int64);
}

// ReduceMax: keepdims=0 drops axis 1 ([2,3,4] -> [2,4]).

TEST_F(EngineShapes, ReduceMaxDropsAxes)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {1}));
    put_sub(body, 1, encode_node("rm0", "ReduceMax", {"x", "axes"}, {"y"},
                      {encode_attribute_int("keepdims", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 4}));

    const engine::Graph graph =
        load_graph("rmax.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3, 4}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{2, 4}));
}

// Flatten: axis=2 folds [2,3,4,5] into [6,20].

TEST_F(EngineShapes, FlattenAxis)
{
    std::string body;
    put_sub(body, 1, encode_node("fl0", "Flatten", {"x"}, {"y"},
                      {encode_attribute_int("axis", 2)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4, 5}));
    put_sub(body, 12, encode_value_info("y", 1, {6, 20}));

    const engine::Graph graph =
        load_graph("flatten.onnx", encode_model(body));
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"x", {2, 3, 4, 5}}});
    ASSERT_EQ(shapes.tensors.count("y"), 1U);
    EXPECT_EQ(shapes.tensors.at("y").dims, (std::vector<int64_t>{6, 20}));
}

// --- Fixture cases ---------------------------------------------------------

TEST_F(EngineShapes, FixtureInferShapes)
{
    const std::string fixture = fixture_path();
    ASSERT_FALSE(fixture.empty())
        << "missing fixture: set " << fixture_env_name()
        << " to the opset-18 ResNet-18 ONNX model; a missing fixture must "
           "fail, never skip";
    ASSERT_TRUE(std::filesystem::exists(fixture))
        << "fixture path from " << fixture_env_name()
        << " does not exist: " << fixture;

    const engine::Graph graph = engine::LoadGraphFromFile(fixture);
    const engine::InferredShapes shapes =
        engine::InferShapes(graph, {{"input", {1, 3, 224, 224}}});

    const std::string conv_out = first_output_of(graph, "Conv");
    ASSERT_FALSE(conv_out.empty());
    ASSERT_EQ(shapes.tensors.count(conv_out), 1U);
    EXPECT_EQ(shapes.tensors.at(conv_out).dims,
        (std::vector<int64_t>{1, 64, 112, 112}));
    EXPECT_EQ(shapes.tensors.at(conv_out).dtype, engine::DataType::Float32);

    ASSERT_EQ(graph.outputs.size(), 1U);
    const std::string out_name = graph.outputs.front().first;
    ASSERT_EQ(shapes.tensors.count(out_name), 1U);
    EXPECT_EQ(shapes.tensors.at(out_name).dims,
        (std::vector<int64_t>{1, 1000}));

    // Every tensor a node consumes or produces must resolve.
    for (const engine::Node& node : graph.nodes) {
        for (const std::string& input : node.inputs) {
            if (!input.empty()) {
                EXPECT_EQ(shapes.tensors.count(input), 1U) << input;
            }
        }
        for (const std::string& output : node.outputs) {
            EXPECT_EQ(shapes.tensors.count(output), 1U) << output;
        }
    }
}

// The fixture's Reshape target is [1,512] — batch-1 in shape space — so a
// batch-2 inference must FAIL at that node, not be rescued.
TEST_F(EngineShapes, FixtureBatchTwoPinnedReshapeRejected)
{
    const std::string fixture = fixture_path();
    ASSERT_FALSE(fixture.empty())
        << "missing fixture: set " << fixture_env_name()
        << " to the opset-18 ResNet-18 ONNX model; a missing fixture must "
           "fail, never skip";
    ASSERT_TRUE(std::filesystem::exists(fixture))
        << "fixture path from " << fixture_env_name()
        << " does not exist: " << fixture;

    const engine::Graph graph = engine::LoadGraphFromFile(fixture);

    std::string reshape_name;
    for (const engine::Node& node : graph.nodes) {
        if (node.op_type == "Reshape") {
            reshape_name = node.name;
            break;
        }
    }
    ASSERT_FALSE(reshape_name.empty()) << "fixture declares no Reshape node";

    std::string message;
    try {
        engine::InferShapes(graph, {{"input", {2, 3, 224, 224}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty())
        << "batch-2 inference on the batch-pinned fixture did not throw";
    EXPECT_NE(message.find(reshape_name), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Reshape"), std::string::npos)
        << "message: " << message;
}

// Same loaded graph, two different input shapes, no reload: the shape half of
// the per-shape seam ([R-12]/[V-13]). Hand-encoded so it needs no fixture.
TEST_F(EngineShapes, PerShapeSeamWithoutReload)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 3, 4, 4}));

    const engine::Graph graph =
        load_graph("per_shape_seam.onnx", encode_model(body));

    const engine::InferredShapes first =
        engine::InferShapes(graph, {{"x", {1, 3, 4, 4}}});
    ASSERT_EQ(first.tensors.count("y"), 1U);
    EXPECT_EQ(first.tensors.at("y").dims, (std::vector<int64_t>{1, 3, 4, 4}));

    const engine::InferredShapes second =
        engine::InferShapes(graph, {{"x", {4, 3, 4, 4}}});
    ASSERT_EQ(second.tensors.count("y"), 1U);
    EXPECT_EQ(second.tensors.at("y").dims, (std::vector<int64_t>{4, 3, 4, 4}));
}

// ===========================================================================
// EngineShapesNegative.* — rejections whose messages must name the node/op.
// ===========================================================================

class EngineShapesNegative : public ::testing::Test {
};

// Reshape target product that cannot match the input element count.

TEST_F(EngineShapesNegative, ReshapeProductMismatchNamesNode)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("shape", {2}, {5, 5}));
    put_sub(body, 1, encode_node("reshape_bad", "Reshape", {"x", "shape"},
                      {"y"}, {encode_attribute_int("allowzero", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {5, 5}));

    const engine::Graph graph =
        load_graph("reshape_mismatch.onnx", encode_model(body));
    std::string message;
    try {
        engine::InferShapes(graph, {{"x", {2, 3, 4, 4}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "reshape product mismatch did not throw";
    EXPECT_NE(message.find("reshape_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Reshape"), std::string::npos)
        << "message: " << message;
}

// ReduceMean axis outside [-rank, rank-1].

TEST_F(EngineShapesNegative, ReduceMeanAxisOutOfRangeNamesNode)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {5}));
    put_sub(body, 1, encode_node("mean_bad", "ReduceMean", {"x", "axes"},
                      {"y"}, {encode_attribute_int("keepdims", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3, 4, 4}));

    const engine::Graph graph =
        load_graph("mean_oor.onnx", encode_model(body));
    std::string message;
    try {
        engine::InferShapes(graph, {{"x", {2, 3, 4, 4}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "out-of-range axis did not throw";
    EXPECT_NE(message.find("mean_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("ReduceMean"), std::string::npos)
        << "message: " << message;
}

// ReduceMean with a duplicated axis.

TEST_F(EngineShapesNegative, ReduceMeanDuplicateAxisNamesNode)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {2}, {1, 1}));
    put_sub(body, 1, encode_node("mean_dup", "ReduceMean", {"x", "axes"},
                      {"y"}, {encode_attribute_int("keepdims", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3, 4, 4}));

    const engine::Graph graph =
        load_graph("mean_dup.onnx", encode_model(body));
    std::string message;
    try {
        engine::InferShapes(graph, {{"x", {2, 3, 4, 4}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "duplicate axis did not throw";
    EXPECT_NE(message.find("mean_dup"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("ReduceMean"), std::string::npos)
        << "message: " << message;
}

// A declared graph input missing from input_dims names that input.

TEST_F(EngineShapesNegative, InputAbsentFromInputDimsNamesInput)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 3, 4, 4}));

    const engine::Graph graph =
        load_graph("missing_dims.onnx", encode_model(body));
    std::string message;
    try {
        engine::InferShapes(graph, {});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "missing input dims did not throw";
    EXPECT_NE(message.find("x"), std::string::npos) << "message: " << message;
}

// A node input that no input, initializer, or earlier node provides. The
// loader rejects this at load, so the graph is assembled directly to reach the
// shape walk's own check.

TEST_F(EngineShapesNegative, UnresolvedNodeInputNamesNode)
{
    engine::Graph graph;
    engine::Node node;
    node.name = "relu_bad";
    node.op_type = "Relu";
    node.inputs = {"missing"};
    node.outputs = {"y"};
    graph.nodes.push_back(node);
    graph.inputs.push_back(
        {"x", engine::TensorInfo{engine::DataType::Float32, {1, 2}}});

    std::string message;
    try {
        engine::InferShapes(graph, {{"x", {1, 2}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "unresolved node input did not throw";
    EXPECT_NE(message.find("relu_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Relu"), std::string::npos)
        << "message: " << message;
}

// An operator outside the supported set names the node and op type.

TEST_F(EngineShapesNegative, UnknownOpNamesNodeAndOpType)
{
    engine::Graph graph;
    engine::Node node;
    node.name = "weird_node";
    node.op_type = "Fancy";
    node.inputs = {"x"};
    node.outputs = {"y"};
    graph.nodes.push_back(node);
    graph.inputs.push_back(
        {"x", engine::TensorInfo{engine::DataType::Float32, {1, 2}}});

    std::string message;
    try {
        engine::InferShapes(graph, {{"x", {1, 2}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "unknown op did not throw";
    EXPECT_NE(message.find("weird_node"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Fancy"), std::string::npos)
        << "message: " << message;
}

// TopK with a non-constant K input: the loader already rejects this, so the
// graph is assembled directly to reach the shape walk's own check.

TEST_F(EngineShapesNegative, TopKNonConstantKNamesNode)
{
    engine::Graph graph;
    engine::Node add;
    add.name = "k_src";
    add.op_type = "Add";
    add.inputs = {"x", "x"};
    add.outputs = {"k"};
    engine::Node topk;
    topk.name = "tk_bad";
    topk.op_type = "TopK";
    topk.inputs = {"x", "k"};
    topk.outputs = {"v", "i"};
    topk.attributes.push_back(engine::Attribute("axis", int64_t(-1)));
    topk.attributes.push_back(engine::Attribute("largest", int64_t(1)));
    topk.attributes.push_back(engine::Attribute("sorted", int64_t(1)));
    graph.nodes.push_back(add);
    graph.nodes.push_back(topk);
    graph.inputs.push_back(
        {"x", engine::TensorInfo{engine::DataType::Float32, {2, 5}}});

    std::string message;
    try {
        engine::InferShapes(graph, {{"x", {2, 5}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "non-constant TopK K did not throw";
    EXPECT_NE(message.find("tk_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("TopK"), std::string::npos)
        << "message: " << message;
}

// Slice with a computed (non-constant) starts input names the node.

TEST_F(EngineShapesNegative, SliceDynamicStartsNamesNode)
{
    engine::Graph graph;
    engine::Node add;
    add.name = "starts_src";
    add.op_type = "Add";
    add.inputs = {"x", "x"};
    add.outputs = {"starts"};
    engine::Node slice;
    slice.name = "sl_bad";
    slice.op_type = "Slice";
    slice.inputs = {"x", "starts", "ends"};
    slice.outputs = {"y"};
    graph.nodes.push_back(add);
    graph.nodes.push_back(slice);
    graph.inputs.push_back(
        {"x", engine::TensorInfo{engine::DataType::Float32, {1, 10}}});
    engine::Initializer ends;
    ends.dtype = engine::DataType::Int64;
    ends.dims = {1};
    ends.values = std::vector<int64_t>{5};
    graph.initializers["ends"] = ends;

    std::string message;
    try {
        engine::InferShapes(graph, {{"x", {1, 10}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "dynamic Slice starts did not throw";
    EXPECT_NE(message.find("sl_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Slice"), std::string::npos)
        << "message: " << message;
}

// Split sizes that do not sum to the axis dim name the node.

TEST_F(EngineShapesNegative, SplitSizeMismatchNamesNode)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("split", {2}, {2, 2}));
    put_sub(body, 1, encode_node("sp_bad", "Split", {"x", "split"},
                      {"a", "b"}, {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 5}));
    put_sub(body, 12, encode_value_info("a", 1, {1, 2}));

    const engine::Graph graph =
        load_graph("split_mismatch.onnx", encode_model(body));
    std::string message;
    try {
        engine::InferShapes(graph, {{"x", {1, 5}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "split size mismatch did not throw";
    EXPECT_NE(message.find("sp_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Split"), std::string::npos)
        << "message: " << message;
}

// Resize with neither scales nor sizes names the node.

TEST_F(EngineShapesNegative, ResizeMissingInputsNamesNode)
{
    std::string body;
    put_sub(body, 1, encode_node("rs_bad", "Resize", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 8, 8}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 3, 16, 16}));

    const engine::Graph graph =
        load_graph("resize_noscale.onnx", encode_model(body));
    std::string message;
    try {
        engine::InferShapes(graph, {{"x", {1, 3, 8, 8}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "scaleless Resize did not throw";
    EXPECT_NE(message.find("rs_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Resize"), std::string::npos)
        << "message: " << message;
}

// Concat inputs that differ off the concat axis name the node.

TEST_F(EngineShapesNegative, ConcatRankMismatchNamesNode)
{
    std::string body;
    put_sub(body, 1, encode_node("c_bad", "Concat", {"a", "b"}, {"y"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("a", 1, {1, 2, 4}));
    put_sub(body, 11, encode_value_info("b", 1, {1, 3, 5}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 5, 4}));

    const engine::Graph graph =
        load_graph("concat_mismatch.onnx", encode_model(body));
    std::string message;
    try {
        engine::InferShapes(graph, {{"a", {1, 2, 4}}, {"b", {1, 3, 5}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "concat mismatch did not throw";
    EXPECT_NE(message.find("c_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Concat"), std::string::npos)
        << "message: " << message;
}

// Mod inputs that disagree on dtype name the node (loader pins no dtype
// agreement, so the shape walk enforces it).

TEST_F(EngineShapesNegative, ModDtypeMismatchNamesNode)
{
    engine::Graph graph;
    engine::Node cast;
    cast.name = "c0";
    cast.op_type = "Cast";
    cast.inputs = {"x"};
    cast.outputs = {"a"};
    cast.attributes.push_back(engine::Attribute("to", int64_t(7)));
    engine::Node mod;
    mod.name = "mod_bad";
    mod.op_type = "Mod";
    mod.inputs = {"a", "x"};
    mod.outputs = {"z"};
    mod.attributes.push_back(engine::Attribute("fmod", int64_t(0)));
    graph.nodes.push_back(cast);
    graph.nodes.push_back(mod);
    graph.inputs.push_back(
        {"x", engine::TensorInfo{engine::DataType::Float32, {2, 3}}});

    std::string message;
    try {
        engine::InferShapes(graph, {{"x", {2, 3}}});
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "Mod dtype mismatch did not throw";
    EXPECT_NE(message.find("mod_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Mod"), std::string::npos)
        << "message: " << message;
}
