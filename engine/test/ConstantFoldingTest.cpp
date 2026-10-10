// Load-time constant folding: hand-built graphs, folded with the CPU device.

#include "engine/ConstantFolding.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

engine::Node MakeNode(const std::string& name, const std::string& op_type, std::vector<std::string> inputs,
                      std::vector<std::string> outputs) {
    engine::Node node;
    node.name = name;
    node.op_type = op_type;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    return node;
}

engine::Initializer FloatInit(std::vector<int64_t> dims, std::vector<float> values) {
    engine::Initializer init;
    init.dtype = engine::DataType::Float32;
    init.dims = std::move(dims);
    init.values = std::move(values);
    return init;
}

} // namespace

// A constant-only chain folds into initializers; the node fed by a graph
// input stays.
TEST(EngineConstantFolding, FoldsConstantChainKeepsLiveNodes) {
    engine::Graph graph;
    graph.inputs.push_back({"x", engine::TensorInfo{engine::DataType::Float32, {2}}});
    graph.outputs.push_back({"y", engine::TensorInfo{engine::DataType::Float32, {2}}});
    graph.initializers["a"] = FloatInit({2}, {1.5F, -0.5F});
    graph.nodes.push_back(MakeNode("floor0", "Floor", {"a"}, {"b"}));
    graph.nodes.push_back(MakeNode("neg_scale", "Mul", {"b", "b"}, {"c"}));
    graph.nodes.push_back(MakeNode("add0", "Add", {"x", "c"}, {"y"}));

    EXPECT_EQ(engine::FoldConstants(graph, engine::CpuDevice()), 2);

    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().name, "add0");
    ASSERT_EQ(graph.initializers.count("c"), 1U);
    EXPECT_EQ(std::get<std::vector<float>>(graph.initializers.at("b").values), (std::vector<float>{1.0F, -1.0F}));
    EXPECT_EQ(std::get<std::vector<float>>(graph.initializers.at("c").values), (std::vector<float>{1.0F, 1.0F}));
}

// A constant-only node that produces a graph output is left for the executor,
// which reads graph outputs from the arena.
TEST(EngineConstantFolding, LeavesGraphOutputProducers) {
    engine::Graph graph;
    graph.outputs.push_back({"y", engine::TensorInfo{engine::DataType::Float32, {1}}});
    graph.initializers["a"] = FloatInit({1}, {2.5F});
    graph.nodes.push_back(MakeNode("floor0", "Floor", {"a"}, {"y"}));

    EXPECT_EQ(engine::FoldConstants(graph, engine::CpuDevice()), 0);
    EXPECT_EQ(graph.nodes.size(), 1U);
}

// A node with no CPU kernel is left in place for the loader's usual rejection.
TEST(EngineConstantFolding, LeavesUnknownOps) {
    engine::Graph graph;
    graph.initializers["a"] = FloatInit({1}, {2.5F});
    graph.nodes.push_back(MakeNode("mystery", "NotAnOp", {"a"}, {"b"}));

    EXPECT_EQ(engine::FoldConstants(graph, engine::CpuDevice()), 0);
    EXPECT_EQ(graph.nodes.size(), 1U);
}
