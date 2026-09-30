// Arena memory-planner tests.
//
// Two suites, split for ctest filters (suite names must not carry the ctest
// NAME colons, so the filters live in the add_test COMMAND):
//   EnginePlan.*           — positives: reuse/overlap, alignment, sequential
//                            arena shrink, external names, graph outputs, the
//                            per-shape seam, and the real fixture plan
//   EnginePlanNegative.*   — rejection classes with tensor/node message asserts
//
// The per-shape seam ([V-13], plan half) is proven hermetically on a loaded
// two-Conv graph planned at two spatial sizes; the fixture case plans the
// opset-18 ResNet-18 export resolved via NEURIPLO_NATIVE_FIXTURE (env var or
// CMake cache variable) and a missing fixture FAILS — never skipped or mocked.

#include "engine/Graph.hpp"
#include "engine/ModelLoader.hpp"
#include "engine/Plan.hpp"
#include "engine/Shapes.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Hand-rolled protobuf wire encoders (test-side twin of the loader's reader).
// Field numbers mirror ModelLoader.cpp; no public header carries any.
// ---------------------------------------------------------------------------

void put_varint(std::string& out, uint64_t value) {
    while (value >= 0x80U) {
        out.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

void put_tag(std::string& out, int field_number, uint8_t wire_type) {
    put_varint(out, (static_cast<uint64_t>(field_number) << 3) | wire_type);
}

void put_length_delimited(std::string& out, int field_number, const std::string& payload) {
    put_tag(out, field_number, 2);
    put_varint(out, payload.size());
    out.append(payload);
}

void put_sub(std::string& out, int field_number, const std::string& sub) {
    put_length_delimited(out, field_number, sub);
}

void put_varint_field(std::string& out, int field_number, uint64_t value) {
    put_tag(out, field_number, 0);
    put_varint(out, value);
}

std::string encode_opset18() {
    std::string leaving; // domain (field 1) omitted; default ONNX domain
    put_varint_field(leaving, 2, 18);
    return leaving;
}

std::string encode_value_info(const std::string& name, int64_t elem_type, const std::vector<int64_t>& dims) {
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

std::string encode_node(const std::string& name, const std::string& op_type, const std::vector<std::string>& inputs,
                        const std::vector<std::string>& outputs, const std::vector<std::string>& attr_bytes) {
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

std::string encode_attribute_ints(const std::string& name, const std::vector<int64_t>& values) {
    std::string a;
    put_length_delimited(a, 1, name);
    for (const int64_t value : values) {
        put_varint_field(a, 8, static_cast<uint64_t>(value));
    }
    put_varint_field(a, 20, 7); // INTS
    return a;
}

std::string encode_initializer(const std::string& name, const std::vector<int64_t>& dims,
                               const std::vector<float>& values) {
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 1); // FLOAT
    put_length_delimited(t, 8, name);
    const std::string raw(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    put_length_delimited(t, 9, raw);
    return t;
}

std::string encode_model(const std::string& graph_body) {
    std::string model;
    put_sub(model, 8, encode_opset18());
    put_sub(model, 7, graph_body);
    return model;
}

std::string write_model_bytes(const std::string& name, const std::string& bytes) {
    const std::string dir = (std::filesystem::temp_directory_path() / "engine_plan_tests").string();
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

engine::Graph load_graph(const std::string& name, const std::string& bytes) {
    return engine::LoadGraphFromFile(write_model_bytes(name, bytes));
}

std::vector<float> zeros(std::size_t count) { return std::vector<float>(count, 0.0F); }

engine::TensorInfo f32(std::vector<int64_t> dims) {
    engine::TensorInfo info;
    info.dtype = engine::DataType::Float32;
    info.dims = std::move(dims);
    return info;
}

engine::Node relu_node(const std::string& name, const std::string& input, const std::string& output) {
    engine::Node node;
    node.name = name;
    node.op_type = "Relu";
    node.inputs = {input};
    node.outputs = {output};
    return node;
}

// Two planned byte ranges share at least one byte.
bool ranges_overlap(const engine::BufferAssignment& a, const engine::BufferAssignment& b) {
    return a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}

const char* fixture_env_name() { return "NEURIPLO_NATIVE_FIXTURE"; }

} // namespace

// ===========================================================================
// EnginePlan.* — positive cases
// ===========================================================================

class EnginePlan : public ::testing::Test {
  protected:
    static std::string fixture_path() {
        const char* from_env = std::getenv(fixture_env_name());
        return from_env != nullptr ? std::string(from_env) : std::string();
    }
};

// Two independent Relu chains, x -> a0 -> aout and y -> b0 -> bout, all rank-1
// width 1x64. Under inclusive liveness a0 is live over [0, 1] while b0 is live
// over [2, 3]: b0 starts after a0's last use, so their lifetimes are genuinely
// disjoint and they share offset 0. The chain outputs aout [1, 3] and b0 [2, 3]
// overlap in time, so those must not share bytes.
TEST_F(EnginePlan, DisjointLifetimesShareOffset) {
    engine::Graph graph;
    graph.inputs.push_back({"x", f32({1, 64})});
    graph.inputs.push_back({"y", f32({1, 64})});
    graph.nodes.push_back(relu_node("relu0", "x", "a0"));
    graph.nodes.push_back(relu_node("relu1", "a0", "aout"));
    graph.nodes.push_back(relu_node("relu2", "y", "b0"));
    graph.nodes.push_back(relu_node("relu3", "b0", "bout"));
    graph.outputs.push_back({"aout", f32({1, 64})});
    graph.outputs.push_back({"bout", f32({1, 64})});

    engine::InferredShapes shapes;
    shapes.tensors["x"] = f32({1, 64});
    shapes.tensors["y"] = f32({1, 64});
    shapes.tensors["a0"] = f32({1, 64});
    shapes.tensors["aout"] = f32({1, 64});
    shapes.tensors["b0"] = f32({1, 64});
    shapes.tensors["bout"] = f32({1, 64});

    const engine::MemoryPlan plan = engine::PlanMemory(graph, shapes);
    ASSERT_EQ(plan.buffers.size(), 4U);
    EXPECT_EQ(plan.buffers.at("a0").offset, plan.buffers.at("b0").offset)
        << "disjoint inclusive lifetimes should reuse: a0=" << plan.buffers.at("a0").offset
        << " b0=" << plan.buffers.at("b0").offset;
    EXPECT_FALSE(ranges_overlap(plan.buffers.at("aout"), plan.buffers.at("b0")))
        << "aout=" << plan.buffers.at("aout").offset << "+" << plan.buffers.at("aout").size
        << " b0=" << plan.buffers.at("b0").offset << "+" << plan.buffers.at("b0").size;
}

// `a` stays live until Add consumes it at node 2, while `b` is defined at node
// 1 and consumed at node 2: their live ranges intersect, so they cannot share.
TEST_F(EnginePlan, OverlappingLifetimesDoNotOverlap) {
    engine::Graph graph;
    graph.inputs.push_back({"x", f32({1, 64})});
    graph.nodes.push_back(relu_node("relu0", "x", "a"));
    graph.nodes.push_back(relu_node("relu1", "a", "b"));
    engine::Node add;
    add.name = "add0";
    add.op_type = "Add";
    add.inputs = {"a", "b"};
    add.outputs = {"c"};
    graph.nodes.push_back(add);
    graph.outputs.push_back({"c", f32({1, 64})});

    engine::InferredShapes shapes;
    shapes.tensors["x"] = f32({1, 64});
    shapes.tensors["a"] = f32({1, 64});
    shapes.tensors["b"] = f32({1, 64});
    shapes.tensors["c"] = f32({1, 64});

    const engine::MemoryPlan plan = engine::PlanMemory(graph, shapes);
    ASSERT_EQ(plan.buffers.size(), 3U);
    const engine::BufferAssignment& a = plan.buffers.at("a");
    const engine::BufferAssignment& b = plan.buffers.at("b");
    EXPECT_NE(a.offset, b.offset);
    const bool disjoint = a.offset + a.size <= b.offset || b.offset + b.size <= a.offset;
    EXPECT_TRUE(disjoint) << "a=" << a.offset << "+" << a.size << " b=" << b.offset << "+" << b.size;
}

// Every offset is 64-byte aligned and every buffer fits inside the arena.
TEST_F(EnginePlan, OffsetsAlignedAndWithinArena) {
    engine::Graph graph;
    graph.inputs.push_back({"x", f32({1, 64})});
    graph.nodes.push_back(relu_node("relu0", "x", "a"));
    graph.nodes.push_back(relu_node("relu1", "a", "b"));
    engine::Node add;
    add.name = "add0";
    add.op_type = "Add";
    add.inputs = {"a", "b"};
    add.outputs = {"c"};
    graph.nodes.push_back(add);
    graph.outputs.push_back({"c", f32({1, 64})});

    engine::InferredShapes shapes;
    shapes.tensors["x"] = f32({1, 64});
    shapes.tensors["a"] = f32({1, 64});
    shapes.tensors["b"] = f32({1, 64});
    shapes.tensors["c"] = f32({1, 64});

    const engine::MemoryPlan plan = engine::PlanMemory(graph, shapes);
    ASSERT_FALSE(plan.buffers.empty());
    for (const auto& entry : plan.buffers) {
        const engine::BufferAssignment& buffer = entry.second;
        EXPECT_EQ(buffer.offset % 64, 0) << entry.first;
        EXPECT_LE(buffer.offset + buffer.size, plan.arena_size) << entry.first;
    }
    EXPECT_GT(plan.arena_size, 0);
}

// A three-link Relu chain x -> a0 -> a1 -> a2. Under inclusive liveness
// a0=[0,1], a1=[1,2], a2=[2,2]; consecutive links touch at the boundary so
// they cannot share, but a0 and a2 are disjoint and reuse the same bytes. The
// arena is therefore strictly smaller than the sum of its tensors ([V-5] reuse
// proof).
TEST_F(EnginePlan, SequentialArenaSmallerThanSum) {
    engine::Graph graph;
    graph.inputs.push_back({"x", f32({1, 64})});
    graph.nodes.push_back(relu_node("relu0", "x", "a0"));
    graph.nodes.push_back(relu_node("relu1", "a0", "a1"));
    graph.nodes.push_back(relu_node("relu2", "a1", "a2"));
    graph.outputs.push_back({"a2", f32({1, 64})});

    engine::InferredShapes shapes;
    shapes.tensors["x"] = f32({1, 64});
    shapes.tensors["a0"] = f32({1, 64});
    shapes.tensors["a1"] = f32({1, 64});
    shapes.tensors["a2"] = f32({1, 64});

    const engine::MemoryPlan plan = engine::PlanMemory(graph, shapes);
    int64_t sum = 0;
    for (const auto& entry : plan.buffers) {
        sum += entry.second.size;
    }
    EXPECT_GT(sum, 0);
    EXPECT_LT(plan.arena_size, sum);
}

// Graph inputs and embedded constants are external: they never get a buffer.
TEST_F(EnginePlan, InputsAndInitializersHaveNoBuffer) {
    engine::Graph graph;
    graph.inputs.push_back({"x", f32({1, 3, 8, 8})});
    engine::Initializer weight;
    weight.dtype = engine::DataType::Float32;
    weight.dims = {4, 3, 3, 3};
    weight.values = zeros(4 * 3 * 3 * 3);
    graph.initializers["w"] = weight;

    engine::Node conv;
    conv.name = "conv0";
    conv.op_type = "Conv";
    conv.inputs = {"x", "w"};
    conv.outputs = {"y"};
    graph.nodes.push_back(conv);
    graph.outputs.push_back({"y", f32({1, 4, 8, 8})});

    engine::InferredShapes shapes;
    shapes.tensors["x"] = f32({1, 3, 8, 8});
    shapes.tensors["w"] = f32({4, 3, 3, 3});
    shapes.tensors["y"] = f32({1, 4, 8, 8});

    const engine::MemoryPlan plan = engine::PlanMemory(graph, shapes);
    EXPECT_EQ(plan.buffers.count("x"), 0U);
    EXPECT_EQ(plan.buffers.count("w"), 0U);
    EXPECT_EQ(plan.buffers.count("y"), 1U);
}

// A declared graph output is an arena tensor and gets a buffer.
TEST_F(EnginePlan, GraphOutputHasBuffer) {
    engine::Graph graph;
    graph.inputs.push_back({"x", f32({2, 5})});
    graph.nodes.push_back(relu_node("relu0", "x", "y"));
    graph.outputs.push_back({"y", f32({2, 5})});

    engine::InferredShapes shapes;
    shapes.tensors["x"] = f32({2, 5});
    shapes.tensors["y"] = f32({2, 5});

    const engine::MemoryPlan plan = engine::PlanMemory(graph, shapes);
    ASSERT_EQ(plan.buffers.count("y"), 1U);
    EXPECT_GT(plan.buffers.at("y").size, 0);
}

// The inclusive-liveness safety invariant the executor depends on: no node
// output may share bytes with any input of its defining node (so a
// read-then-write kernel never overwrites its own input). Checked on a
// hermetic node and over every node of the real fixture plan.
TEST_F(EnginePlan, OutputNeverAliasesItsInput) {
    // Hermetic: relu1 consumes arena tensor `a` and defines `b`; the planner
    // must keep their byte ranges disjoint.
    {
        engine::Graph graph;
        graph.inputs.push_back({"x", f32({1, 8})});
        graph.nodes.push_back(relu_node("relu0", "x", "a"));
        graph.nodes.push_back(relu_node("relu1", "a", "b"));
        graph.outputs.push_back({"b", f32({1, 8})});

        engine::InferredShapes shapes;
        shapes.tensors["x"] = f32({1, 8});
        shapes.tensors["a"] = f32({1, 8});
        shapes.tensors["b"] = f32({1, 8});

        const engine::MemoryPlan plan = engine::PlanMemory(graph, shapes);
        ASSERT_EQ(plan.buffers.count("a"), 1U);
        ASSERT_EQ(plan.buffers.count("b"), 1U);
        EXPECT_FALSE(ranges_overlap(plan.buffers.at("a"), plan.buffers.at("b")))
            << "a=" << plan.buffers.at("a").offset << "+" << plan.buffers.at("a").size
            << " b=" << plan.buffers.at("b").offset << "+" << plan.buffers.at("b").size;
    }

    // Real fixture: the same invariant for every node with a buffered input and
    // a buffered output. A missing fixture FAILS, never skips.
    const std::string fixture = fixture_path();
    ASSERT_FALSE(fixture.empty()) << "missing fixture: set " << fixture_env_name()
                                  << " to the opset-18 ResNet-18 ONNX model; a missing fixture must "
                                     "fail, never skip";
    ASSERT_TRUE(std::filesystem::exists(fixture))
        << "fixture path from " << fixture_env_name() << " does not exist: " << fixture;

    const engine::Graph graph = engine::LoadGraphFromFile(fixture);
    const engine::InferredShapes shapes = engine::InferShapes(graph, {{"input", {1, 3, 224, 224}}});
    const engine::MemoryPlan plan = engine::PlanMemory(graph, shapes);

    for (const engine::Node& node : graph.nodes) {
        for (const std::string& input : node.inputs) {
            const auto in_buffer = plan.buffers.find(input);
            if (in_buffer == plan.buffers.end()) {
                continue; // graph input or initializer: external, no arena buffer
            }
            for (const std::string& output : node.outputs) {
                const auto out_buffer = plan.buffers.find(output);
                if (out_buffer == plan.buffers.end()) {
                    continue;
                }
                EXPECT_FALSE(ranges_overlap(in_buffer->second, out_buffer->second))
                    << "node '" << node.name << "' (" << node.op_type << ") output '" << output
                    << "' aliases input '" << input << "'";
            }
        }
    }
}

// [V-13] plan half: one loaded graph, two inferred shapes, two plans, no
// reload. Hand-encoded so it needs no fixture.
TEST_F(EnginePlan, PerShapeSeamWithoutReload) {
    std::string body;
    put_sub(body, 5, encode_initializer("w0", {2, 3, 3, 3}, zeros(2 * 3 * 3 * 3)));
    put_sub(body, 5, encode_initializer("w1", {4, 2, 3, 3}, zeros(4 * 2 * 3 * 3)));
    put_sub(body, 1, encode_node("conv0", "Conv", {"x", "w0"}, {"a"}, {encode_attribute_ints("pads", {1, 1, 1, 1})}));
    put_sub(body, 1, encode_node("conv1", "Conv", {"a", "w1"}, {"b"}, {encode_attribute_ints("pads", {1, 1, 1, 1})}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 8, 8}));
    put_sub(body, 12, encode_value_info("b", 1, {1, 4, 8, 8}));

    const engine::Graph graph = load_graph("plan_per_shape.onnx", encode_model(body));

    const engine::InferredShapes first = engine::InferShapes(graph, {{"x", {1, 3, 8, 8}}});
    const engine::MemoryPlan first_plan = engine::PlanMemory(graph, first);
    ASSERT_FALSE(first_plan.buffers.empty());
    ASSERT_EQ(first_plan.buffers.count("a"), 1U);
    ASSERT_EQ(first_plan.buffers.count("b"), 1U);
    EXPECT_EQ(first_plan.buffers.at("a").size, 1 * 2 * 8 * 8 * 4);
    EXPECT_EQ(first_plan.buffers.at("b").size, 1 * 4 * 8 * 8 * 4);
    EXPECT_GT(first_plan.arena_size, 0);

    const engine::InferredShapes second = engine::InferShapes(graph, {{"x", {1, 3, 16, 16}}});
    const engine::MemoryPlan second_plan = engine::PlanMemory(graph, second);
    ASSERT_FALSE(second_plan.buffers.empty());
    ASSERT_EQ(second_plan.buffers.count("a"), 1U);
    ASSERT_EQ(second_plan.buffers.count("b"), 1U);
    EXPECT_EQ(second_plan.buffers.at("a").size, 1 * 2 * 16 * 16 * 4);
    EXPECT_EQ(second_plan.buffers.at("b").size, 1 * 4 * 16 * 16 * 4);
    EXPECT_NE(first_plan.arena_size, second_plan.arena_size);
}

// The real fixture plans to a non-empty arena.
TEST_F(EnginePlan, FixturePlanNonEmpty) {
    const std::string fixture = fixture_path();
    ASSERT_FALSE(fixture.empty()) << "missing fixture: set " << fixture_env_name()
                                  << " to the opset-18 ResNet-18 ONNX model; a missing fixture must "
                                     "fail, never skip";
    ASSERT_TRUE(std::filesystem::exists(fixture))
        << "fixture path from " << fixture_env_name() << " does not exist: " << fixture;

    const engine::Graph graph = engine::LoadGraphFromFile(fixture);
    const engine::InferredShapes shapes = engine::InferShapes(graph, {{"input", {1, 3, 224, 224}}});
    const engine::MemoryPlan plan = engine::PlanMemory(graph, shapes);

    EXPECT_FALSE(plan.buffers.empty());
    EXPECT_GT(plan.arena_size, 0);
    for (const auto& entry : plan.buffers) {
        EXPECT_GT(entry.second.size, 0) << entry.first;
    }
}

// ===========================================================================
// EnginePlanNegative.* — rejections whose messages must name the tensor/node.
// ===========================================================================

class EnginePlanNegative : public ::testing::Test {};

// A node output absent from the inferred shapes names that tensor.
TEST_F(EnginePlanNegative, MissingShapeNamesTensor) {
    engine::Graph graph;
    graph.inputs.push_back({"x", f32({1, 4})});
    graph.nodes.push_back(relu_node("relu0", "x", "missing_tensor"));
    graph.outputs.push_back({"missing_tensor", f32({1, 4})});

    engine::InferredShapes shapes;
    shapes.tensors["x"] = f32({1, 4});

    std::string message;
    try {
        engine::PlanMemory(graph, shapes);
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "missing inferred shape did not throw";
    EXPECT_NE(message.find("missing_tensor"), std::string::npos) << "message: " << message;
}

// An empty graph cannot be planned.
TEST_F(EnginePlanNegative, EmptyGraphRejected) {
    engine::Graph graph;
    engine::InferredShapes shapes;

    std::string message;
    try {
        engine::PlanMemory(graph, shapes);
    } catch (const engine::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "empty graph did not throw";
    EXPECT_NE(message.find("no nodes"), std::string::npos) << "message: " << message;
}
