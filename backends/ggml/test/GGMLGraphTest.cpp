#include "GGMLGraph.hpp"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

using ggml_graph::DType;
using ggml_graph::Graph;
using ggml_graph::Op;

std::vector<uint8_t> f32_blob(std::initializer_list<float> values) {
    std::vector<float> data(values);
    std::vector<uint8_t> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    return bytes;
}

std::vector<uint8_t> i64_blob(std::initializer_list<int64_t> values) {
    std::vector<int64_t> data(values);
    std::vector<uint8_t> bytes(data.size() * sizeof(int64_t));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    return bytes;
}

std::vector<float> read_f32(const ggml_graph::Output& output) {
    std::vector<float> values(output.data.size() / sizeof(float));
    std::memcpy(values.data(), output.data.data(), output.data.size());
    return values;
}

std::vector<int32_t> read_i32(const ggml_graph::Output& output) {
    std::vector<int32_t> values(output.data.size() / sizeof(int32_t));
    std::memcpy(values.data(), output.data.data(), output.data.size());
    return values;
}

void expect_near(const std::vector<float>& got, const std::vector<float>& expected, float tolerance = 1e-5f) {
    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], tolerance) << "index " << i;
    }
}

} // namespace

TEST(GGMLGraphTest, MultiInputAdd) {
    Graph graph;
    const int a = graph.add_input("a", DType::F32, {2});
    const int b = graph.add_input("b", DType::F32, {2});
    const int sum = graph.add_op(Op::Add, {a, b});
    graph.mark_output(sum);

    const auto outputs = graph.compute({f32_blob({3, 4}), f32_blob({5, 6})});
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].dtype, DType::F32);
    EXPECT_EQ(outputs[0].shape, (std::vector<int64_t>{2}));
    expect_near(read_f32(outputs[0]), {8, 10});
}

TEST(GGMLGraphTest, Int64InputCastAndGather) {
    Graph graph;
    // ONNX data [N=3, D=2] stored as ggml [D=2, N=3].
    const int data = graph.add_const_f32("data", {2, 3}, {1, 2, 3, 4, 5, 6});
    const int indices = graph.add_input("indices", DType::I64, {2});
    const int indices_i32 = graph.add_op(Op::CastI64ToI32, {indices});
    const int gathered = graph.add_op(Op::Gather, {data, indices_i32});
    graph.mark_output(gathered);

    const auto outputs = graph.compute({i64_blob({2, 0})});
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].dtype, DType::F32);
    EXPECT_EQ(outputs[0].shape, (std::vector<int64_t>{2, 2}));
    expect_near(read_f32(outputs[0]), {5, 6, 1, 2});
}

TEST(GGMLGraphTest, MultipleOutputs) {
    Graph graph;
    const int a = graph.add_input("a", DType::F32, {2});
    const int b = graph.add_input("b", DType::F32, {2});
    const int sum = graph.add_op(Op::Add, {a, b});
    const int product = graph.add_op(Op::Mul, {a, b});
    graph.mark_output(sum);
    graph.mark_output(product);

    const auto outputs = graph.compute({f32_blob({3, 4}), f32_blob({5, 6})});
    ASSERT_EQ(outputs.size(), 2u);
    expect_near(read_f32(outputs[0]), {8, 10});
    expect_near(read_f32(outputs[1]), {15, 24});
}

TEST(GGMLGraphTest, TinyTransformerBlock) {
    Graph graph;
    const int x = graph.add_input("x", DType::F32, {2, 2});
    const int ln_w = graph.add_const_f32("ln_w", {2}, {1, 1});
    const int ln_b = graph.add_const_f32("ln_b", {2}, {0, 0});
    const int w1 = graph.add_const_f32("w1", {2, 3}, {1, 0, 0, 1, 1, 1});
    const int b1 = graph.add_const_f32("b1", {3}, {0, 0, 0});
    const int w2 = graph.add_const_f32("w2", {3, 2}, {1, 1, 0, 0, 1, 1});
    const int b2 = graph.add_const_f32("b2", {2}, {0, 0});
    const int norm = graph.add_op(Op::LayerNorm, {x, ln_w, ln_b}, {1e-5f});
    const int hidden = graph.add_op(Op::Linear, {norm, w1, b1});
    const int activated = graph.add_op(Op::Gelu, {hidden});
    const int projected = graph.add_op(Op::Linear, {activated, w2, b2});
    const int out = graph.add_op(Op::Add, {x, projected});
    graph.mark_output(out);

    const auto outputs = graph.compute({f32_blob({1, 2, 3, 4})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {1.6826661f, 2.8413231f, 3.6826661f, 4.8413231f}, 1e-5f);
}

TEST(GGMLGraphTest, RejectsBadContracts) {
    Graph graph;
    const int a = graph.add_input("a", DType::F32, {2});
    const int b = graph.add_input("b", DType::F32, {2});
    const int sum = graph.add_op(Op::Add, {a, b});
    graph.mark_output(sum);

    EXPECT_THROW(graph.compute({f32_blob({1, 2})}), std::invalid_argument);
    EXPECT_THROW(graph.compute({f32_blob({1, 2}), f32_blob({3, 4}), f32_blob({5, 6})}), std::invalid_argument);
    EXPECT_THROW(graph.compute({f32_blob({1}), f32_blob({3, 4})}), std::invalid_argument);

    Graph empty;
    EXPECT_THROW(empty.compute({}), std::invalid_argument);

    Graph gather;
    const int data = gather.add_const_f32("data", {2, 2}, {1, 2, 3, 4});
    const int f32_indices = gather.add_input("indices", DType::F32, {2});
    EXPECT_THROW(gather.add_op(Op::Gather, {data, f32_indices}), std::invalid_argument);
}
