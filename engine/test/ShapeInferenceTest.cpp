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
