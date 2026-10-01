// Sequential-executor tests.
//
// Two suites, split for ctest filters:
//   EngineExecutor.*          — hermetic Add+Relu with hand-computed output, a
//                               counting device proving one arena allocation
//                               across construction plus two runs, and the
//                               ResNet-18 fixture end to end.
//   EngineExecutorNegative.*  — missing input, wrong input size, and an op with
//                               no kernel, each throwing InferenceException.
//
// The hermetic graph reuses the hand-rolled protobuf wire encoders from the
// loader tests: the field numbers mirror ModelLoader.cpp and no public header
// carries any. The fixture is resolved via NEURIPLO_NATIVE_FIXTURE; a missing
// fixture FAILS, never skips.

#include "engine/Executor.hpp"

#include "engine/Device.hpp"
#include "engine/Graph.hpp"
#include "engine/ModelLoader.hpp"

#include <cmath>
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
// Hand-rolled protobuf wire encoders (test-side twin of the reader).
//
// Field numbers: ModelProto graph=7 opset_import=8; GraphProto node=1
// initializer=5 input=11 output=12; NodeProto input=1 output=2 name=3
// op_type=4; AttributeProto name=1; TensorProto dims=1 data_type=2 name=8
// raw_data=9; OpsetId version=2; ValueInfoProto name=1 type=2; TypeProto
// tensor_type=1; TensorType elem_type=1 shape=2; TensorShapeProto dim=1;
// Dimension dim_value=1.
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
    std::string leaving;
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
                        const std::vector<std::string>& outputs) {
    std::string n;
    for (const std::string& input : inputs) {
        put_length_delimited(n, 1, input);
    }
    for (const std::string& output : outputs) {
        put_length_delimited(n, 2, output);
    }
    put_length_delimited(n, 3, name);
    put_length_delimited(n, 4, op_type);
    return n;
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
    const std::string dir = (std::filesystem::temp_directory_path() / "engine_executor_tests").string();
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

// A hermetic Add(x, c) -> y followed by Relu(y) -> z, with the float constant
// `c` supplied as an initializer. The graph input x has one element per entry
// in `constant`; the graph output is z.
engine::Graph load_hermetic_add_relu(const std::vector<float>& constant) {
    const int64_t width = static_cast<int64_t>(constant.size());
    std::string body;
    put_sub(body, 5, encode_initializer("c", {width}, constant));
    put_sub(body, 1, encode_node("add0", "Add", {"x", "c"}, {"y"}));
    put_sub(body, 1, encode_node("relu0", "Relu", {"y"}, {"z"}));
    put_sub(body, 11, encode_value_info("x", 1, {width}));
    put_sub(body, 12, encode_value_info("z", 1, {width}));
    return engine::LoadGraphFromFile(write_model_bytes("add_relu.onnx", encode_model(body)));
}

// Add(x, c) -> t0 followed by `relu_links` Relu links, output tN. Under
// inclusive liveness the links touch at their boundaries, so only alternating
// tensors can reuse; a chain of at least three links makes the 64-byte-aligned
// arena strictly smaller than the sum of the intermediate buffers.
engine::Graph load_hermetic_chain(const std::vector<float>& constant, int relu_links) {
    const int64_t width = static_cast<int64_t>(constant.size());
    std::string body;
    put_sub(body, 5, encode_initializer("c", {width}, constant));
    put_sub(body, 1, encode_node("add0", "Add", {"x", "c"}, {"t0"}));
    std::string previous = "t0";
    for (int i = 1; i <= relu_links; ++i) {
        const std::string next = "t" + std::to_string(i);
        put_sub(body, 1, encode_node("relu" + std::to_string(i), "Relu", {previous}, {next}));
        previous = next;
    }
    put_sub(body, 11, encode_value_info("x", 1, {width}));
    put_sub(body, 12, encode_value_info(previous, 1, {width}));
    return engine::LoadGraphFromFile(write_model_bytes("chain.onnx", encode_model(body)));
}

const char* fixture_env_name() { return "NEURIPLO_NATIVE_FIXTURE"; }

std::string fixture_path() {
    const char* from_env = std::getenv(fixture_env_name());
    return from_env != nullptr ? std::string(from_env) : std::string();
}

// The process-lifetime CPU device as a mutable reference, so the non-const
// allocator and transfer accessors are usable.
engine::Device& cpu_device() { return const_cast<engine::Device&>(engine::CpuDevice()); }

// Counts allocator calls while delegating to the CPU allocator.
class CountingAllocator final : public engine::Allocator {
  public:
    int allocate_calls = 0;
    int release_calls = 0;

    void* allocate(int64_t bytes) override {
        ++allocate_calls;
        return cpu_device().allocator().allocate(bytes);
    }

    void release(void* buffer) override {
        ++release_calls;
        cpu_device().allocator().release(buffer);
    }
};

// A kernel table with no entries, used to prove the executor reports a missing
// kernel rather than silently doing nothing.
class EmptyKernelTable final : public engine::KernelTable {
  public:
    engine::KernelFn find(const std::string&) const override { return nullptr; }
};

// A test device whose allocator counts calls. When `empty_kernels` is true it
// advertises no kernels; otherwise it forwards to the CPU device's table.
class TestDevice final : public engine::Device {
  public:
    explicit TestDevice(bool empty_kernels) : kernels_(empty_kernels ? nullptr : &engine::CpuDevice().kernels()) {}

    const char* name() const override { return "test-cpu"; }
    engine::Allocator& allocator() override { return allocator_; }
    engine::Transfer& transfer() override { return cpu_device().transfer(); }
    const engine::KernelTable& kernels() const override { return kernels_ != nullptr ? *kernels_ : empty_table_; }

    const CountingAllocator& counting() const { return allocator_; }

  private:
    CountingAllocator allocator_;
    const engine::KernelTable* kernels_;
    EmptyKernelTable empty_table_;
};

} // namespace

// ===========================================================================
// EngineExecutor.* — positive cases
// ===========================================================================
TEST(EngineExecutor, HermeticAddReluMatchesHandComputed) {
    const engine::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    engine::Model model(graph, {{"x", {2}}}, engine::CpuDevice());

    const engine::InferenceResult result = model.Run({{"x", {1.0F, -2.0F}}});
    ASSERT_EQ(result.outputs.count("z"), 1U);
    EXPECT_EQ(result.outputs.at("z"), (std::vector<float>{0.5F, 0.0F}));
}

TEST(EngineExecutor, ArenaAllocatedOnceAndReused) {
    // A three-link chain under inclusive liveness: only alternating tensors can
    // share, so the 64-byte-aligned arena stays strictly smaller than the sum of
    // the intermediate buffers and the reuse proof is not swamped by alignment.
    const std::vector<float> constant(16, 0.0F);
    const engine::Graph graph = load_hermetic_chain(constant, 2);
    TestDevice device(false);
    {
        engine::Model model(graph, {{"x", {16}}}, device);
        const std::vector<float> input(16, 1.0F);
        const engine::InferenceResult first = model.Run({{"x", input}});
        const engine::InferenceResult second = model.Run({{"x", input}});
        EXPECT_EQ(first.outputs.at("t2"), second.outputs.at("t2"));
        EXPECT_EQ(device.counting().allocate_calls, 1);
        EXPECT_EQ(device.counting().release_calls, 0);

        int64_t intermediate_total = 0;
        for (const auto& buffer : model.plan().buffers) {
            intermediate_total += buffer.second.size;
        }
        EXPECT_LT(model.plan().arena_size, intermediate_total);
    }
    EXPECT_EQ(device.counting().allocate_calls, 1);
    EXPECT_EQ(device.counting().release_calls, 1);
}

TEST(EngineExecutor, FixtureResNet18RunsAndIsFinite) {
    const std::string fixture = fixture_path();
    ASSERT_FALSE(fixture.empty()) << "missing fixture: set " << fixture_env_name()
                                  << " to the opset-18 ResNet-18 ONNX model; a missing fixture must fail, never skip";
    ASSERT_TRUE(std::filesystem::exists(fixture)) << "fixture does not exist: " << fixture;

    const engine::Graph graph = engine::LoadGraphFromFile(fixture);
    ASSERT_FALSE(graph.inputs.empty());
    ASSERT_FALSE(graph.outputs.empty());
    ASSERT_EQ(graph.inputs.front().second.dims, (std::vector<int64_t>{1, 3, 224, 224}));

    const std::string input_name = graph.inputs.front().first;
    const std::string output_name = graph.outputs.front().first;
    engine::Model model(graph, {{input_name, {1, 3, 224, 224}}}, engine::CpuDevice());

    const std::vector<float> input(static_cast<std::size_t>(1 * 3 * 224 * 224), 0.01F);
    const engine::InferenceResult result = model.Run({{input_name, input}});

    ASSERT_EQ(result.outputs.count(output_name), 1U);
    const std::vector<float>& output = result.outputs.at(output_name);
    EXPECT_EQ(output.size(), 1000U);
    for (std::size_t i = 0; i < output.size(); ++i) {
        EXPECT_TRUE(std::isfinite(output[i])) << "output[" << i << "] = " << output[i];
    }
}

// ===========================================================================
// EngineExecutorNegative.* — rejections
// ===========================================================================

TEST(EngineExecutorNegative, MissingInputThrows) {
    const engine::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    engine::Model model(graph, {{"x", {2}}}, engine::CpuDevice());
    EXPECT_THROW(model.Run({}), engine::InferenceException);
}

TEST(EngineExecutorNegative, WrongInputSizeThrows) {
    const engine::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    engine::Model model(graph, {{"x", {2}}}, engine::CpuDevice());
    EXPECT_THROW(model.Run({{"x", {1.0F}}}), engine::InferenceException);
}

TEST(EngineExecutorNegative, OpWithoutKernelThrows) {
    const engine::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    ASSERT_FALSE(graph.nodes.empty());
    TestDevice device(true); // advertises no kernels
    engine::Model model(graph, {{"x", {2}}}, device);

    try {
        model.Run({{"x", {1.0F, -2.0F}}});
        FAIL() << "expected InferenceException for an op with no kernel";
    } catch (const engine::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find(graph.nodes.front().name), std::string::npos) << message;
        EXPECT_NE(message.find(graph.nodes.front().op_type), std::string::npos) << message;
    }
}
