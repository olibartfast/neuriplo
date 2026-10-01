// CPU kernel tests: Relu, Add, Reshape, ReduceMean, Mul, Div, Sub, Sigmoid,
// Cast, Softmax, ReduceMax, Mod, Equal, Where, Concat, Split, Unsqueeze,
// Expand, Transpose, Slice, Gather, GatherElements, Resize, Flatten, Shape,
// ConstantOfShape, and TopK.
//
// Expected values are hand-computed small cases written inline, not captured
// from another runtime. Kernels are reached through the CPU device's kernel
// table (CpuDevice().kernels().find(...)), the same way the executor will.

#include "engine/Device.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

// A float32 tensor view over `data`, with its concrete shape.
engine::TensorView FloatView(std::vector<float>& data, std::vector<int64_t> dims) {
    engine::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = engine::DataType::Float32;
    view.dims = std::move(dims);
    return view;
}

// An int64 tensor view over `data`, with its concrete shape.
engine::TensorView Int64View(std::vector<int64_t>& data, std::vector<int64_t> dims) {
    engine::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = engine::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

// A bool tensor view over `data` (one byte per element, 0 or 1), with its
// concrete shape.
engine::TensorView BoolView(std::vector<std::uint8_t>& data, std::vector<int64_t> dims) {
    engine::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = engine::DataType::Bool;
    view.dims = std::move(dims);
    return view;
}

// Reads a bool output buffer back as bytes (0 or 1) for comparison.
std::vector<std::uint8_t> BoolBytes(const std::vector<std::uint8_t>& data) {
    const bool* flags = reinterpret_cast<const bool*>(data.data());
    std::vector<std::uint8_t> out(data.size(), 0);
    for (std::size_t i = 0; i < data.size(); ++i) {
        out[i] = flags[i] ? std::uint8_t{1} : std::uint8_t{0};
    }
    return out;
}

engine::Node MakeNode(const std::string& name, const std::string& op_type) {
    engine::Node node;
    node.name = name;
    node.op_type = op_type;
    return node;
}

} // namespace

// Relu clamps a mixed-sign vector at zero.
TEST(EngineKernels, ReluMixedSignVector) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Relu");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {-2.0f, -0.5f, 0.0f, 3.0f, 1.5f};
    std::vector<float> out(in.size(), 0.0f);
    engine::Node node = MakeNode("relu0", "Relu");
    std::vector<engine::TensorView> inputs = {FloatView(in, {5})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {5})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{0.0f, 0.0f, 0.0f, 3.0f, 1.5f}));
}

// Add broadcasts a trailing [3] input across a [2,3] input.
TEST(EngineKernels, AddBroadcastsLowerRank) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Add");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("add0", "Add");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f}));
}

// Add broadcasts [2,1] against [1,3] into [2,3].
TEST(EngineKernels, AddBroadcastsBothAxes) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Add");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("add1", "Add");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 1}), FloatView(b, {1, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f}));
}

// Reshape reorders [2,3] into [3,2] without touching the flat buffer.
TEST(EngineKernels, ReshapeReordersDims) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(in.size(), 0.0f);
    engine::Node node = MakeNode("reshape0", "Reshape");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

// Reshape changes rank to a flat [6] and validates an int64 shape operand.
TEST(EngineKernels, ReshapeRankChangeWithShapeOperand) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(in.size(), 0.0f);
    std::vector<int64_t> shape = {6};
    engine::Node node = MakeNode("reshape1", "Reshape");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(shape, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {6})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

// ReduceMean over axis 1 of [2,3] with the default keepdims == 1.
TEST(EngineKernels, ReduceMeanOneAxisKeepdimsDefault) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ReduceMean");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1};
    engine::Node node = MakeNode("reduce0", "ReduceMean");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{2.0f, 5.0f}));
}

// ReduceMean over axis 1 with keepdims == 0 squeezes the reduced axis.
TEST(EngineKernels, ReduceMeanOneAxisKeepdimsZero) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ReduceMean");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1};
    engine::Node node = MakeNode("reduce1", "ReduceMean");
    node.attributes.emplace_back("keepdims", int64_t{0});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{2.0f, 5.0f}));
}

// ReduceMean with no axes input reduces every axis, leaving [1,1] by default.
TEST(EngineKernels, ReduceMeanAllAxes) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ReduceMean");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(1, 0.0f);
    engine::Node node = MakeNode("reduce2", "ReduceMean");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{2.5f}));
}

// Gemm computes the plain matrix product without transposes or scaling.
TEST(EngineKernels, GemmPlainProduct) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("gemm0", "Gemm");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{58.0f, 64.0f, 139.0f, 154.0f}));
}

// Gemm transA transposes a [K,M] lhs, leaving the same product as the plain case.
TEST(EngineKernels, GemmTransA) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("gemm_transA", "Gemm");
    node.attributes.emplace_back("transA", int64_t{1});
    std::vector<engine::TensorView> inputs = {FloatView(a, {3, 2}), FloatView(b, {3, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{89.0f, 98.0f, 116.0f, 128.0f}));
}

// Gemm transB transposes a [N,K] rhs, matching the plain product.
TEST(EngineKernels, GemmTransB) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("gemm_transB", "Gemm");
    node.attributes.emplace_back("transB", int64_t{1});
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {2, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{58.0f, 64.0f, 139.0f, 154.0f}));
}

// Gemm scales by alpha, adds beta times a rank-1 [N] bias, and broadcasts it.
TEST(EngineKernels, GemmAlphaBetaRankOneBias) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> c = {10.0f, 20.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("gemm_scale", "Gemm");
    node.attributes.emplace_back("alpha", 2.0f);
    node.attributes.emplace_back("beta", 0.5f);
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 2}), FloatView(b, {2, 2}), FloatView(c, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{43.0f, 54.0f, 91.0f, 110.0f}));
}

// MatMul on two rank-2 operands contracts the shared inner dimension.
TEST(EngineKernels, MatMulTwoDimensional) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("matmul0", "MatMul");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{58.0f, 64.0f, 139.0f, 154.0f}));
}

// MatMul on rank-3 operands multiplies each batch independently.
TEST(EngineKernels, MatMulBatchedThreeDimensional) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    std::vector<float> b = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f};
    std::vector<float> out(8, 0.0f);
    engine::Node node = MakeNode("matmul_batch", "MatMul");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 2, 3}), FloatView(b, {2, 3, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{4.0f, 5.0f, 10.0f, 11.0f, 2.0f, 0.0f, 0.0f, 2.0f}));
}

// MatMul promotes a 1-D lhs to [1,K] and drops the promoted axis.
TEST(EngineKernels, MatMulPromotesOneDimensionalLhs) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(2, 0.0f);
    engine::Node node = MakeNode("matmul_lhs1d", "MatMul");
    std::vector<engine::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {3, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{4.0f, 5.0f}));
}

// MatMul promotes a 1-D rhs to [K,1] and drops the promoted axis.
TEST(EngineKernels, MatMulPromotesOneDimensionalRhs) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {1.0f, 0.0f, 1.0f};
    std::vector<float> out(2, 0.0f);
    engine::Node node = MakeNode("matmul_rhs1d", "MatMul");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{4.0f, 10.0f}));
}

// Conv with a 1x1 identity weight copies the input through unchanged.
TEST(EngineKernels, ConvIdentityOneByOne) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> w = {1.0f};
    std::vector<float> out(9, 0.0f);
    engine::Node node = MakeNode("conv_id", "Conv");
    std::vector<engine::TensorView> inputs = {FloatView(x, {1, 1, 3, 3}), FloatView(w, {1, 1, 1, 1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1, 3, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}));
}

// Conv with a 2x2 all-ones kernel sums each 2x2 input window.
TEST(EngineKernels, ConvKnownTwoByTwoKernel) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("conv_2x2", "Conv");
    std::vector<engine::TensorView> inputs = {FloatView(x, {1, 1, 3, 3}), FloatView(w, {1, 1, 2, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{12.0f, 16.0f, 24.0f, 28.0f}));
}

// Conv adds a rank-1 [M] bias to every output channel.
TEST(EngineKernels, ConvWithBias) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> b = {0.5f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("conv_bias", "Conv");
    std::vector<engine::TensorView> inputs = {FloatView(x, {1, 1, 3, 3}), FloatView(w, {1, 1, 2, 2}),
                                              FloatView(b, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{12.5f, 16.5f, 24.5f, 28.5f}));
}

// Conv with group == C is depthwise: each output channel sees only its own
// input channel.
TEST(EngineKernels, ConvDepthwiseGroupEqualsChannels) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> w = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    std::vector<float> out(2, 0.0f);
    engine::Node node = MakeNode("conv_dw", "Conv");
    node.attributes.emplace_back("group", int64_t{2});
    std::vector<engine::TensorView> inputs = {FloatView(x, {1, 2, 2, 2}), FloatView(w, {2, 1, 2, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 2, 1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 50.0f}));
}

// Conv with stride 2 and a one-cell pad samples the padded input on the
// computation lattice.
TEST(EngineKernels, ConvStrideAndPadding) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("conv_stride_pad", "Conv");
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    std::vector<engine::TensorView> inputs = {FloatView(x, {1, 1, 4, 4}), FloatView(w, {1, 1, 3, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{14.0f, 30.0f, 57.0f, 99.0f}));
}

// MaxPool with a 2x2 window and stride 2 takes the max of each disjoint block.
TEST(EngineKernels, MaxPoolTwoByTwoStrideTwo) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("MaxPool");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("pool_2x2", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    std::vector<engine::TensorView> inputs = {FloatView(x, {1, 1, 4, 4})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{6.0f, 8.0f, 14.0f, 16.0f}));
}

// MaxPool rounds the output up with ceil_mode and ignores the padded border.
TEST(EngineKernels, MaxPoolPaddingAndCeilMode) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("MaxPool");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("pool_pad_ceil", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("pads", std::vector<int64_t>{0, 0, 1, 1});
    node.attributes.emplace_back("ceil_mode", int64_t{1});
    std::vector<engine::TensorView> inputs = {FloatView(x, {1, 1, 3, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 6.0f, 8.0f, 9.0f}));
}

// Mul broadcasts a trailing [3] input across a [2,3] input.
TEST(EngineKernels, MulBroadcastsLowerRank) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Mul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("mul0", "Mul");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{10.0f, 40.0f, 90.0f, 40.0f, 100.0f, 180.0f}));
}

// Mul broadcasts [2,1] against [1,3] into [2,3].
TEST(EngineKernels, MulBroadcastsBothAxes) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Mul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {2.0f, 3.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("mul1", "Mul");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 1}), FloatView(b, {1, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{20.0f, 40.0f, 60.0f, 30.0f, 60.0f, 90.0f}));
}

// Div divides elementwise without broadcasting.
TEST(EngineKernels, DivPlainQuotient) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Div");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {6.0f, 8.0f, 10.0f};
    std::vector<float> b = {2.0f, 4.0f, 5.0f};
    std::vector<float> out(3, 0.0f);
    engine::Node node = MakeNode("div0", "Div");
    std::vector<engine::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{3.0f, 2.0f, 2.0f}));
}

// Div broadcasts a trailing [3] divisor across a [2,3] dividend.
TEST(EngineKernels, DivBroadcastsLowerRank) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Div");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {6.0f, 12.0f, 18.0f, 24.0f, 30.0f, 36.0f};
    std::vector<float> b = {2.0f, 3.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("div1", "Div");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{3.0f, 4.0f, 3.0f, 12.0f, 10.0f, 6.0f}));
}

// Div by zero follows IEEE float semantics: inf, -inf, nan, never a throw.
TEST(EngineKernels, DivByZeroYieldsInfAndNan) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Div");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, -1.0f, 0.0f, 6.0f};
    std::vector<float> b = {0.0f, 0.0f, 0.0f, 3.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("div_zero", "Div");
    std::vector<engine::TensorView> inputs = {FloatView(a, {4}), FloatView(b, {4})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_TRUE(std::isinf(out[0]) && out[0] > 0.0f);
    EXPECT_TRUE(std::isinf(out[1]) && out[1] < 0.0f);
    EXPECT_TRUE(std::isnan(out[2]));
    EXPECT_EQ(out[3], 2.0f);
}

// Sub subtracts elementwise without broadcasting.
TEST(EngineKernels, SubPlainDifference) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Sub");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {5.0f, 7.0f, 9.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(3, 0.0f);
    engine::Node node = MakeNode("sub0", "Sub");
    std::vector<engine::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{4.0f, 5.0f, 6.0f}));
}

// Sub broadcasts [2,1] against [1,3] into [2,3].
TEST(EngineKernels, SubBroadcastsBothAxes) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Sub");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {10.0f, 20.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("sub1", "Sub");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 1}), FloatView(b, {1, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{9.0f, 8.0f, 7.0f, 19.0f, 18.0f, 17.0f}));
}

// Sigmoid maps 0 to 0.5 and symmetric inputs to complementary outputs.
TEST(EngineKernels, SigmoidBasicValues) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Sigmoid");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {0.0f, 2.0f, -2.0f};
    std::vector<float> out(3, 0.0f);
    engine::Node node = MakeNode("sigmoid0", "Sigmoid");
    std::vector<engine::TensorView> inputs = {FloatView(in, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_FLOAT_EQ(out[0], 0.5f);
    EXPECT_NEAR(out[1], 0.88079708f, 1e-6f);
    EXPECT_NEAR(out[2], 0.11920292f, 1e-6f);
}

// Sigmoid saturates: large inputs approach 1, large negatives approach 0.
TEST(EngineKernels, SigmoidSaturatesForLargeMagnitudes) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Sigmoid");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {10.0f, -10.0f};
    std::vector<float> out(2, 0.0f);
    engine::Node node = MakeNode("sigmoid1", "Sigmoid");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_NEAR(out[0], 0.9999546f, 1e-6f);
    EXPECT_NEAR(out[1], 0.0000453979f, 1e-8f);
}

// Cast converts float32 to int64 with truncation toward zero.
TEST(EngineKernels, CastFloatToInt64Truncates) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.9f, -2.7f, 0.0f, 3.0f};
    std::vector<int64_t> out(4, 0);
    engine::Node node = MakeNode("cast0", "Cast");
    node.attributes.emplace_back("to", int64_t{7});
    std::vector<engine::TensorView> inputs = {FloatView(in, {4})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, -2, 0, 3}));
}

// Cast converts int64 to float32 exactly for small magnitudes.
TEST(EngineKernels, CastInt64ToFloat) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, -2, 0};
    std::vector<float> out(3, 0.0f);
    engine::Node node = MakeNode("cast1", "Cast");
    node.attributes.emplace_back("to", int64_t{1});
    std::vector<engine::TensorView> inputs = {Int64View(in, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, -2.0f, 0.0f}));
}

// Cast converts float32 to bool: only exact zero maps to false.
TEST(EngineKernels, CastFloatToBool) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {0.0f, 2.5f, -1.0f};
    std::vector<std::uint8_t> out(3, 0);
    engine::Node node = MakeNode("cast2", "Cast");
    node.attributes.emplace_back("to", int64_t{9});
    std::vector<engine::TensorView> inputs = {FloatView(in, {3})};
    std::vector<engine::TensorView> outputs = {BoolView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{0, 1, 1}));
}

// Cast converts bool to int64 as 1/0.
TEST(EngineKernels, CastBoolToInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0, 1};
    std::vector<int64_t> out(3, 0);
    engine::Node node = MakeNode("cast3", "Cast");
    node.attributes.emplace_back("to", int64_t{7});
    std::vector<engine::TensorView> inputs = {BoolView(in, {3})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 0, 1}));
}

// Cast converts int64 to bool: only zero maps to false.
TEST(EngineKernels, CastInt64ToBool) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {0, 5, -3};
    std::vector<std::uint8_t> out(3, 0);
    engine::Node node = MakeNode("cast4", "Cast");
    node.attributes.emplace_back("to", int64_t{9});
    std::vector<engine::TensorView> inputs = {Int64View(in, {3})};
    std::vector<engine::TensorView> outputs = {BoolView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{0, 1, 1}));
}

// Cast converts bool to float32 as 1.0/0.0.
TEST(EngineKernels, CastBoolToFloat) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0};
    std::vector<float> out(2, 0.0f);
    engine::Node node = MakeNode("cast5", "Cast");
    node.attributes.emplace_back("to", int64_t{1});
    std::vector<engine::TensorView> inputs = {BoolView(in, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 0.0f}));
}

// Softmax over the default last axis normalizes each row of [2,3].
TEST(EngineKernels, SoftmaxDefaultLastAxis) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Softmax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("softmax0", "Softmax");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_NEAR(out[0], 0.09003057f, 1e-6f);
    EXPECT_NEAR(out[1], 0.24472847f, 1e-6f);
    EXPECT_NEAR(out[2], 0.66524096f, 1e-6f);
    EXPECT_NEAR(out[3], 0.09003057f, 1e-6f);
    EXPECT_NEAR(out[4], 0.24472847f, 1e-6f);
    EXPECT_NEAR(out[5], 0.66524096f, 1e-6f);
}

// Softmax over axis 0 normalizes each column of [2,3]; every column differs
// by 3, so every column shares the same pair of outputs.
TEST(EngineKernels, SoftmaxFirstAxis) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Softmax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("softmax1", "Softmax");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_NEAR(out[0], 0.04742587f, 1e-6f);
    EXPECT_NEAR(out[1], 0.04742587f, 1e-6f);
    EXPECT_NEAR(out[2], 0.04742587f, 1e-6f);
    EXPECT_NEAR(out[3], 0.95257413f, 1e-6f);
    EXPECT_NEAR(out[4], 0.95257413f, 1e-6f);
    EXPECT_NEAR(out[5], 0.95257413f, 1e-6f);
}

// ReduceMax over axis 1 of [2,3] with the default keepdims == 1.
TEST(EngineKernels, ReduceMaxOneAxisKeepdimsDefault) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1};
    engine::Node node = MakeNode("reducemax0", "ReduceMax");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 6.0f}));
}

// ReduceMax over axis 1 with keepdims == 0 squeezes the reduced axis.
TEST(EngineKernels, ReduceMaxOneAxisKeepdimsZero) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1};
    engine::Node node = MakeNode("reducemax1", "ReduceMax");
    node.attributes.emplace_back("keepdims", int64_t{0});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 6.0f}));
}

// ReduceMax with no axes input reduces every axis, leaving [1,1] by default.
TEST(EngineKernels, ReduceMaxAllAxes) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f};
    std::vector<float> out(1, 0.0f);
    engine::Node node = MakeNode("reducemax2", "ReduceMax");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f}));
}

// ReduceMax accepts a negative axis: -1 names the last axis of [2,3].
TEST(EngineKernels, ReduceMaxNegativeAxis) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {-1};
    engine::Node node = MakeNode("reducemax3", "ReduceMax");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 6.0f}));
}

// Mod takes the truncating remainder on int64; the sign follows the dividend.
TEST(EngineKernels, ModDividendSign) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Mod");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {7, -7, 7, -7};
    std::vector<int64_t> b = {3, 3, -3, -3};
    std::vector<int64_t> out(4, 0);
    engine::Node node = MakeNode("mod0", "Mod");
    std::vector<engine::TensorView> inputs = {Int64View(a, {4}), Int64View(b, {4})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, -1, 1, -1}));
}

// Mod broadcasts [2,1] against [3] into [2,3].
TEST(EngineKernels, ModBroadcastsLowerRank) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Mod");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {3, 4, 5};
    std::vector<int64_t> out(6, 0);
    engine::Node node = MakeNode("mod1", "Mod");
    std::vector<engine::TensorView> inputs = {Int64View(a, {2, 1}), Int64View(b, {3})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 3, 2, 2, 0, 3}));
}

// Equal compares float32 vectors elementwise into a bool output.
TEST(EngineKernels, EqualFloatVectors) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Equal");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 0.0f, 3.0f};
    std::vector<std::uint8_t> out(3, 0);
    engine::Node node = MakeNode("equal0", "Equal");
    std::vector<engine::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {3})};
    std::vector<engine::TensorView> outputs = {BoolView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 0, 1}));
}

// Equal broadcasts int64 [2,1] against [3] into a [2,3] bool output.
TEST(EngineKernels, EqualInt64Broadcast) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Equal");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {5, 6};
    std::vector<int64_t> b = {5, 5, 5};
    std::vector<std::uint8_t> out(6, 0);
    engine::Node node = MakeNode("equal1", "Equal");
    std::vector<engine::TensorView> inputs = {Int64View(a, {2, 1}), Int64View(b, {3})};
    std::vector<engine::TensorView> outputs = {BoolView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 1, 1, 0, 0, 0}));
}

// Where selects elementwise between two flat vectors.
TEST(EngineKernels, WhereFlatSelect) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Where");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> cond = {1, 0, 1};
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(3, 0.0f);
    engine::Node node = MakeNode("where0", "Where");
    std::vector<engine::TensorView> inputs = {BoolView(cond, {3}), FloatView(x, {3}), FloatView(y, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 20.0f, 3.0f}));
}

// Where broadcasts all three inputs: cond [2,1], X [3], scalar Y into [2,3].
TEST(EngineKernels, WhereBroadcastsAllThree) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Where");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> cond = {1, 0};
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y = {9.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("where1", "Where");
    std::vector<engine::TensorView> inputs = {BoolView(cond, {2, 1}), FloatView(x, {3}), FloatView(y, {})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 9.0f, 9.0f, 9.0f}));
}

// The op-type table resolves the implemented kernels and rejects unknown ops.
TEST(EngineKernels, TableResolvesKnownOpsAndRejectsUnknown) {
    const engine::KernelTable& table = engine::CpuDevice().kernels();
    EXPECT_NE(table.find("Relu"), nullptr);
    EXPECT_NE(table.find("Add"), nullptr);
    EXPECT_NE(table.find("Reshape"), nullptr);
    EXPECT_NE(table.find("ReduceMean"), nullptr);
    EXPECT_NE(table.find("Gemm"), nullptr);
    EXPECT_NE(table.find("MatMul"), nullptr);
    EXPECT_NE(table.find("Conv"), nullptr);
    EXPECT_NE(table.find("MaxPool"), nullptr);
    EXPECT_NE(table.find("Mul"), nullptr);
    EXPECT_NE(table.find("Div"), nullptr);
    EXPECT_NE(table.find("Sub"), nullptr);
    EXPECT_NE(table.find("Sigmoid"), nullptr);
    EXPECT_NE(table.find("Cast"), nullptr);
    EXPECT_NE(table.find("Softmax"), nullptr);
    EXPECT_NE(table.find("ReduceMax"), nullptr);
    EXPECT_NE(table.find("Mod"), nullptr);
    EXPECT_NE(table.find("Equal"), nullptr);
    EXPECT_NE(table.find("Where"), nullptr);
    EXPECT_NE(table.find("Concat"), nullptr);
    EXPECT_NE(table.find("Split"), nullptr);
    EXPECT_NE(table.find("Unsqueeze"), nullptr);
    EXPECT_NE(table.find("Expand"), nullptr);
    EXPECT_NE(table.find("Transpose"), nullptr);
    EXPECT_NE(table.find("Slice"), nullptr);
    EXPECT_NE(table.find("Gather"), nullptr);
    EXPECT_NE(table.find("GatherElements"), nullptr);
    EXPECT_NE(table.find("Resize"), nullptr);
    EXPECT_NE(table.find("Flatten"), nullptr);
    EXPECT_NE(table.find("Shape"), nullptr);
    EXPECT_NE(table.find("ConstantOfShape"), nullptr);
    EXPECT_NE(table.find("TopK"), nullptr);
    EXPECT_EQ(table.find("NoSuchOp"), nullptr);
}

// A float32-expected op rejects an int64 data input, naming the op.
TEST(EngineKernelsNegative, ReluRejectsDtypeMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Relu");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3};
    std::vector<float> out(3, 0.0f);
    engine::Node node = MakeNode("relu_bad", "Relu");
    std::vector<engine::TensorView> inputs = {Int64View(in, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Relu"), std::string::npos) << error.what();
    }
}

// Add rejects the wrong input arity, naming the op.
TEST(EngineKernelsNegative, AddRejectsWrongArity) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Add");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(3, 0.0f);
    engine::Node node = MakeNode("add_bad", "Add");
    std::vector<engine::TensorView> inputs = {FloatView(a, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Add"), std::string::npos) << error.what();
    }
}

// Reshape rejects an element-count mismatch, naming the op.
TEST(EngineKernelsNegative, ReshapeRejectsSizeMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(5, 0.0f);
    engine::Node node = MakeNode("reshape_bad", "Reshape");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {5})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Reshape"), std::string::npos) << error.what();
    }
}

// Gemm rejects a contraction-dimension mismatch, naming the op.
TEST(EngineKernelsNegative, GemmRejectsInnerDimMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("gemm_bad", "Gemm");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {2, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Gemm"), std::string::npos) << error.what();
    }
}

// MatMul rejects a contraction-dimension mismatch, naming the op.
TEST(EngineKernelsNegative, MatMulRejectsInnerDimMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("matmul_bad", "MatMul");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {2, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("MatMul"), std::string::npos) << error.what();
    }
}

// Conv rejects a weight whose channel dimension disagrees with input/group.
TEST(EngineKernelsNegative, ConvRejectsChannelMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("conv_bad", "Conv");
    std::vector<engine::TensorView> inputs = {FloatView(x, {1, 2, 3, 2}), FloatView(w, {1, 1, 2, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1, 2, 1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Conv"), std::string::npos) << error.what();
    }
}

// MaxPool rejects a declared output shape that does not match its window.
TEST(EngineKernelsNegative, MaxPoolRejectsShapeMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("MaxPool");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> out(9, 0.0f);
    engine::Node node = MakeNode("pool_bad", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    std::vector<engine::TensorView> inputs = {FloatView(x, {1, 1, 3, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1, 3, 3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("MaxPool"), std::string::npos) << error.what();
    }
}

// Mul rejects an int64 data input, naming the op.
TEST(EngineKernelsNegative, MulRejectsDtypeMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Mul");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {1, 2, 3};
    std::vector<int64_t> b = {1, 2, 3};
    std::vector<int64_t> out(3, 0);
    engine::Node node = MakeNode("mul_bad", "Mul");
    std::vector<engine::TensorView> inputs = {Int64View(a, {3}), Int64View(b, {3})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Mul"), std::string::npos) << error.what();
    }
}

// Div rejects shapes that cannot broadcast, naming the op.
TEST(EngineKernelsNegative, DivRejectsBroadcastMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Div");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 2.0f};
    std::vector<float> out(3, 0.0f);
    engine::Node node = MakeNode("div_bad", "Div");
    std::vector<engine::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Div"), std::string::npos) << error.what();
    }
}

// Sub rejects the wrong input arity, naming the op.
TEST(EngineKernelsNegative, SubRejectsWrongArity) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Sub");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(3, 0.0f);
    engine::Node node = MakeNode("sub_bad", "Sub");
    std::vector<engine::TensorView> inputs = {FloatView(a, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Sub"), std::string::npos) << error.what();
    }
}

// Sigmoid rejects an int64 data input, naming the op.
TEST(EngineKernelsNegative, SigmoidRejectsDtypeMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Sigmoid");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3};
    std::vector<float> out(3, 0.0f);
    engine::Node node = MakeNode("sigmoid_bad", "Sigmoid");
    std::vector<engine::TensorView> inputs = {Int64View(in, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Sigmoid"), std::string::npos) << error.what();
    }
}

// Cast rejects an unsupported `to` code, naming the op.
TEST(EngineKernelsNegative, CastRejectsUnsupportedTo) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    engine::Node node = MakeNode("cast_bad", "Cast");
    node.attributes.emplace_back("to", int64_t{6});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Cast"), std::string::npos) << error.what();
    }
}

// Cast rejects a missing `to` attribute, naming the op.
TEST(EngineKernelsNegative, CastRejectsMissingTo) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    engine::Node node = MakeNode("cast_noto", "Cast");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Cast"), std::string::npos) << error.what();
    }
}

// Softmax rejects an out-of-range axis, naming the op.
TEST(EngineKernelsNegative, SoftmaxRejectsAxisOutOfRange) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Softmax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("softmax_bad", "Softmax");
    node.attributes.emplace_back("axis", int64_t{2});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Softmax"), std::string::npos) << error.what();
    }
}

// ReduceMax rejects a duplicated axis, naming the op.
TEST(EngineKernelsNegative, ReduceMaxRejectsDuplicateAxis) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(1, 0.0f);
    std::vector<int64_t> axes = {0, 0};
    engine::Node node = MakeNode("reducemax_bad", "ReduceMax");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(axes, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ReduceMax"), std::string::npos) << error.what();
    }
}

// Mod rejects a zero divisor, naming the op.
TEST(EngineKernelsNegative, ModRejectsZeroDivisor) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Mod");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {3, 0};
    std::vector<int64_t> out(2, 0);
    engine::Node node = MakeNode("mod_bad", "Mod");
    std::vector<engine::TensorView> inputs = {Int64View(a, {2}), Int64View(b, {2})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Mod"), std::string::npos) << error.what();
    }
}

// Mod rejects fmod=1, naming the op.
TEST(EngineKernelsNegative, ModRejectsFmodOne) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Mod");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {3, 4};
    std::vector<int64_t> out(2, 0);
    engine::Node node = MakeNode("mod_fmod", "Mod");
    node.attributes.emplace_back("fmod", int64_t{1});
    std::vector<engine::TensorView> inputs = {Int64View(a, {2}), Int64View(b, {2})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Mod"), std::string::npos) << error.what();
    }
}

// Equal rejects mixed float/int inputs, naming the op.
TEST(EngineKernelsNegative, EqualRejectsMixedDtypes) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Equal");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<int64_t> b = {1, 2};
    std::vector<std::uint8_t> out(2, 0);
    engine::Node node = MakeNode("equal_bad", "Equal");
    std::vector<engine::TensorView> inputs = {FloatView(a, {2}), Int64View(b, {2})};
    std::vector<engine::TensorView> outputs = {BoolView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Equal"), std::string::npos) << error.what();
    }
}

// Where rejects a non-bool condition, naming the op.
TEST(EngineKernelsNegative, WhereRejectsNonBoolCondition) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Where");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> cond = {1.0f, 0.0f};
    std::vector<float> x = {1.0f, 2.0f};
    std::vector<float> y = {10.0f, 20.0f};
    std::vector<float> out(2, 0.0f);
    engine::Node node = MakeNode("where_bad", "Where");
    std::vector<engine::TensorView> inputs = {FloatView(cond, {2}), FloatView(x, {2}), FloatView(y, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Where"), std::string::npos) << error.what();
    }
}

// Concat joins [2,2] and [2,1] along axis 1 into [2,3].
TEST(EngineKernels, ConcatAxisOne) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("concat0", "Concat");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<engine::TensorView> inputs = {FloatView(a, {2, 2}), FloatView(b, {2, 1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f}));
}

// Concat with axis -1 names the last axis; int64 data path.
TEST(EngineKernels, ConcatNegativeAxisInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {1, 2};
    std::vector<int64_t> b = {3, 4, 5, 6};
    std::vector<int64_t> out(6, 0);
    engine::Node node = MakeNode("concat1", "Concat");
    node.attributes.emplace_back("axis", int64_t{-1});
    std::vector<engine::TensorView> inputs = {Int64View(a, {2, 1}), Int64View(b, {2, 2})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 3, 4, 2, 5, 6}));
}

// Split divides [2,4] into unequal [2,1] and [2,3] parts along axis 1.
TEST(EngineKernels, SplitUnequalSizes) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Split");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> out0(2, 0.0f);
    std::vector<float> out1(6, 0.0f);
    std::vector<int64_t> sizes = {1, 3};
    engine::Node node = MakeNode("split0", "Split");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 4}), Int64View(sizes, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out0, {2, 1}), FloatView(out1, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out0, (std::vector<float>{1.0f, 5.0f}));
    EXPECT_EQ(out1, (std::vector<float>{2.0f, 3.0f, 4.0f, 6.0f, 7.0f, 8.0f}));
}

// Split with no sizes input divides evenly; negative axis, int64 data.
TEST(EngineKernels, SplitEvenNegativeAxisInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Split");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4};
    std::vector<int64_t> out0(2, 0);
    std::vector<int64_t> out1(2, 0);
    engine::Node node = MakeNode("split1", "Split");
    node.attributes.emplace_back("axis", int64_t{-2});
    std::vector<engine::TensorView> inputs = {Int64View(in, {4, 1})};
    std::vector<engine::TensorView> outputs = {Int64View(out0, {2, 1}), Int64View(out1, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out0, (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(out1, (std::vector<int64_t>{3, 4}));
}

// Unsqueeze inserts a size-1 axis at position 1 of [2,3].
TEST(EngineKernels, UnsqueezeMiddleAxis) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Unsqueeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    std::vector<int64_t> axes = {1};
    engine::Node node = MakeNode("unsqueeze0", "Unsqueeze");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 1, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

// Unsqueeze with axis -1 appends a trailing singleton; int64 data path.
TEST(EngineKernels, UnsqueezeNegativeAxisInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Unsqueeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {10, 20};
    std::vector<int64_t> out(2, 0);
    std::vector<int64_t> axes = {-1};
    engine::Node node = MakeNode("unsqueeze1", "Unsqueeze");
    std::vector<engine::TensorView> inputs = {Int64View(in, {2}), Int64View(axes, {1})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{10, 20}));
}

// Expand broadcasts [1,3] to [2,3] by repeating the leading axis.
TEST(EngineKernels, ExpandLeadingAxis) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Expand");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(6, 0.0f);
    std::vector<int64_t> shape = {2, 3};
    engine::Node node = MakeNode("expand0", "Expand");
    std::vector<engine::TensorView> inputs = {FloatView(in, {1, 3}), Int64View(shape, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f}));
}

// Expand to a higher-rank target right-aligns; int64 data path.
TEST(EngineKernels, ExpandHigherRankInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Expand");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {4, 5, 6};
    std::vector<int64_t> out(6, 0);
    std::vector<int64_t> shape = {2, 1, 3};
    engine::Node node = MakeNode("expand1", "Expand");
    std::vector<engine::TensorView> inputs = {Int64View(in, {3}), Int64View(shape, {3})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2, 1, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{4, 5, 6, 4, 5, 6}));
}

// Transpose with perm [1,0] swaps [2,3] to [3,2].
TEST(EngineKernels, TransposeSwapTwoDimensional) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Transpose");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("transpose0", "Transpose");
    node.attributes.emplace_back("perm", std::vector<int64_t>{1, 0});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));
}

// Transpose with no perm reverses [2,1,3] to [3,1,2]; int64 data path.
TEST(EngineKernels, TransposeDefaultReverseInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Transpose");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> out(6, 0);
    engine::Node node = MakeNode("transpose1", "Transpose");
    std::vector<engine::TensorView> inputs = {Int64View(in, {2, 1, 3})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {3, 1, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 4, 2, 5, 3, 6}));
}

// Slice takes [1,4) with step 1 from a length-6 vector.
TEST(EngineKernels, SlicePlainRange) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    std::vector<float> out(3, 0.0f);
    std::vector<int64_t> starts = {1};
    std::vector<int64_t> ends = {4};
    std::vector<int64_t> axes = {0};
    std::vector<int64_t> steps = {1};
    engine::Node node = MakeNode("slice0", "Slice");
    std::vector<engine::TensorView> inputs = {FloatView(in, {6}), Int64View(starts, {1}), Int64View(ends, {1}),
                                              Int64View(axes, {1}), Int64View(steps, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{20.0f, 30.0f, 40.0f}));
}

// Slice clamps out-of-range bounds and strides the second axis; int64 data.
TEST(EngineKernels, SliceClampAndStepInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int64_t> out(4, 0);
    std::vector<int64_t> starts = {-100, 1};
    std::vector<int64_t> ends = {100, 4};
    std::vector<int64_t> axes = {0, 1};
    std::vector<int64_t> steps = {1, 2};
    engine::Node node = MakeNode("slice1", "Slice");
    std::vector<engine::TensorView> inputs = {Int64View(in, {2, 5}), Int64View(starts, {2}), Int64View(ends, {2}),
                                              Int64View(axes, {2}), Int64View(steps, {2})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{2, 4, 7, 9}));
}

// Gather picks rows 2 and 0 of a [3,2] matrix along axis 0.
TEST(EngineKernels, GatherRowsAxisZero) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Gather");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<int64_t> indices = {2, 0};
    engine::Node node = MakeNode("gather0", "Gather");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<engine::TensorView> inputs = {FloatView(in, {3, 2}), Int64View(indices, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 6.0f, 1.0f, 2.0f}));
}

// Gather with axis -1 and a wrapped negative index; int64 data path.
TEST(EngineKernels, GatherNegativeAxisWrapInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Gather");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> out(4, 0);
    std::vector<int64_t> indices = {2, -3};
    engine::Node node = MakeNode("gather1", "Gather");
    node.attributes.emplace_back("axis", int64_t{-1});
    std::vector<engine::TensorView> inputs = {Int64View(in, {2, 3}), Int64View(indices, {2})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{3, 1, 6, 4}));
}

// GatherElements selects per-position entries along axis 1 of [2,3].
TEST(EngineKernels, GatherElementsAxisOne) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("GatherElements");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<int64_t> indices = {2, 0, 1, 2};
    engine::Node node = MakeNode("gatherelem0", "GatherElements");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(indices, {2, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{3.0f, 1.0f, 5.0f, 6.0f}));
}

// GatherElements with axis -1 and wrapped indices; int64 data path.
TEST(EngineKernels, GatherElementsNegativeAxisInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("GatherElements");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4};
    std::vector<int64_t> out(2, 0);
    std::vector<int64_t> indices = {1, -2};
    engine::Node node = MakeNode("gatherelem1", "GatherElements");
    node.attributes.emplace_back("axis", int64_t{-1});
    std::vector<engine::TensorView> inputs = {Int64View(in, {2, 2}), Int64View(indices, {2, 1})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{2, 3}));
}

// Resize doubles a [1,1,2,2] input to [1,1,4,4] with nearest-floor sampling.
TEST(EngineKernels, ResizeNearestDouble) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(16, 0.0f);
    std::vector<float> scales = {1.0f, 1.0f, 2.0f, 2.0f};
    engine::Node node = MakeNode("resize0", "Resize");
    std::vector<engine::TensorView> inputs = {FloatView(in, {1, 1, 2, 2}), engine::TensorView{},
                                              FloatView(scales, {4})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1, 4, 4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 3.0f,
                                        3.0f, 4.0f, 4.0f}));
}

// Resize triples one axis of a [1,2] row; int64 data path.
TEST(EngineKernels, ResizeTripleWidthInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {5, 6};
    std::vector<int64_t> out(6, 0);
    std::vector<float> scales = {1.0f, 3.0f};
    engine::Node node = MakeNode("resize1", "Resize");
    std::vector<engine::TensorView> inputs = {Int64View(in, {1, 2}), engine::TensorView{},
                                              FloatView(scales, {2})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {1, 6})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{5, 5, 5, 6, 6, 6}));
}

// Flatten collapses [2,3,4] to [2,12] with axis 1.
TEST(EngineKernels, FlattenAxisOne) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Flatten");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(24, 0.0f);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = static_cast<float>(i + 1);
    }
    std::vector<float> out(24, 0.0f);
    engine::Node node = MakeNode("flatten0", "Flatten");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3, 4})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 12})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, in);
}

// Flatten with axis -1 keeps a trailing singleton; int64 data path.
TEST(EngineKernels, FlattenNegativeAxisInt64) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Flatten");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> out(6, 0);
    engine::Node node = MakeNode("flatten1", "Flatten");
    node.attributes.emplace_back("axis", int64_t{-1});
    std::vector<engine::TensorView> inputs = {Int64View(in, {2, 3})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {6, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 2, 3, 4, 5, 6}));
}

// Shape emits the [2,3,4] dims as an int64 shape tensor.
TEST(EngineKernels, ShapeThreeDimensional) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Shape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(24, 1.0f);
    std::vector<int64_t> out(3, 0);
    engine::Node node = MakeNode("shape0", "Shape");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3, 4})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{2, 3, 4}));
}

// Shape reads the rank of a bool input.
TEST(EngineKernels, ShapeBoolInput) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Shape");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0, 1, 0};
    std::vector<int64_t> out(2, 0);
    engine::Node node = MakeNode("shape1", "Shape");
    std::vector<engine::TensorView> inputs = {BoolView(in, {2, 2})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{2, 2}));
}

// ConstantOfShape fills [2,3] with the int64 value attribute.
TEST(EngineKernels, ConstantOfShapeFill) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {2, 3};
    std::vector<int64_t> out(6, 0);
    engine::Node node = MakeNode("cos0", "ConstantOfShape");
    node.attributes.emplace_back("value", int64_t{7});
    std::vector<engine::TensorView> inputs = {Int64View(shape, {2})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{7, 7, 7, 7, 7, 7}));
}

// ConstantOfShape with an empty shape input produces a scalar.
TEST(EngineKernels, ConstantOfShapeScalar) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape;
    std::vector<int64_t> out(1, 0);
    engine::Node node = MakeNode("cos1", "ConstantOfShape");
    node.attributes.emplace_back("value", int64_t{-3});
    std::vector<engine::TensorView> inputs = {Int64View(shape, {0})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{-3}));
}

// TopK picks the 2 largest of 5 values, descending, with int64 indices.
TEST(EngineKernels, TopKTwoLargest) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("TopK");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f};
    std::vector<float> values(2, 0.0f);
    std::vector<int64_t> k = {2};
    std::vector<int64_t> indices(2, 0);
    engine::Node node = MakeNode("topk0", "TopK");
    std::vector<engine::TensorView> inputs = {FloatView(in, {5}), Int64View(k, {})};
    std::vector<engine::TensorView> outputs = {FloatView(values, {2}), Int64View(indices, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(values, (std::vector<float>{5.0f, 4.0f}));
    EXPECT_EQ(indices, (std::vector<int64_t>{1, 3}));
}

// TopK with K < dim selects per row of [2,4] along the default last axis.
TEST(EngineKernels, TopKRowsSmallerK) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("TopK");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 4.0f, 2.0f, 3.0f, 8.0f, 5.0f, 7.0f, 6.0f};
    std::vector<float> values(6, 0.0f);
    std::vector<int64_t> k = {3};
    std::vector<int64_t> indices(6, 0);
    engine::Node node = MakeNode("topk1", "TopK");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 4}), Int64View(k, {})};
    std::vector<engine::TensorView> outputs = {FloatView(values, {2, 3}), Int64View(indices, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(values, (std::vector<float>{4.0f, 3.0f, 2.0f, 8.0f, 7.0f, 6.0f}));
    EXPECT_EQ(indices, (std::vector<int64_t>{1, 3, 2, 0, 2, 3}));
}

// Concat rejects inputs that differ off the concat axis, naming the op.
TEST(EngineKernelsNegative, ConcatRejectsOffAxisMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {3.0f, 4.0f, 5.0f};
    std::vector<float> out(5, 0.0f);
    engine::Node node = MakeNode("concat_bad", "Concat");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<engine::TensorView> inputs = {FloatView(a, {1, 2}), FloatView(b, {1, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 5})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Concat"), std::string::npos) << error.what();
    }
}

// Concat rejects an out-of-range axis, naming the op.
TEST(EngineKernelsNegative, ConcatRejectsAxisOutOfRange) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    engine::Node node = MakeNode("concat_axis", "Concat");
    node.attributes.emplace_back("axis", int64_t{2});
    std::vector<engine::TensorView> inputs = {FloatView(a, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Concat"), std::string::npos) << error.what();
    }
}

// Split rejects sizes that do not sum to the axis dim, naming the op.
TEST(EngineKernelsNegative, SplitRejectsSizeSumMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Split");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out0(1, 0.0f);
    std::vector<float> out1(2, 0.0f);
    std::vector<int64_t> sizes = {1, 2};
    engine::Node node = MakeNode("split_bad", "Split");
    std::vector<engine::TensorView> inputs = {FloatView(in, {4}), Int64View(sizes, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out0, {1}), FloatView(out1, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Split"), std::string::npos) << error.what();
    }
}

// Split rejects an axis dim that does not divide evenly, naming the op.
TEST(EngineKernelsNegative, SplitRejectsUnevenDefault) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Split");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out0(1, 0.0f);
    std::vector<float> out1(1, 0.0f);
    engine::Node node = MakeNode("split_even", "Split");
    std::vector<engine::TensorView> inputs = {FloatView(in, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out0, {1}), FloatView(out1, {1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Split"), std::string::npos) << error.what();
    }
}

// Unsqueeze rejects duplicated axes, naming the op.
TEST(EngineKernelsNegative, UnsqueezeRejectsDuplicateAxes) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Unsqueeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1, 1};
    engine::Node node = MakeNode("unsqueeze_bad", "Unsqueeze");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2}), Int64View(axes, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 1, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Unsqueeze"), std::string::npos) << error.what();
    }
}

// Unsqueeze rejects an out-of-range axis, naming the op.
TEST(EngineKernelsNegative, UnsqueezeRejectsAxisOutOfRange) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Unsqueeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {5};
    engine::Node node = MakeNode("unsqueeze_axis", "Unsqueeze");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2}), Int64View(axes, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Unsqueeze"), std::string::npos) << error.what();
    }
}

// Expand rejects a target the input cannot broadcast to, naming the op.
TEST(EngineKernelsNegative, ExpandRejectsUnbroadcastable) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Expand");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(6, 0.0f);
    std::vector<int64_t> shape = {3, 2};
    engine::Node node = MakeNode("expand_bad", "Expand");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(shape, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Expand"), std::string::npos) << error.what();
    }
}

// Transpose rejects a perm that is not a permutation, naming the op.
TEST(EngineKernelsNegative, TransposeRejectsBadPerm) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Transpose");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    engine::Node node = MakeNode("transpose_bad", "Transpose");
    node.attributes.emplace_back("perm", std::vector<int64_t>{0, 0});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Transpose"), std::string::npos) << error.what();
    }
}

// Slice rejects starts/ends of different lengths, naming the op.
TEST(EngineKernelsNegative, SliceRejectsStartsEndsMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> starts = {0, 0};
    std::vector<int64_t> ends = {2};
    engine::Node node = MakeNode("slice_bad", "Slice");
    std::vector<engine::TensorView> inputs = {FloatView(in, {3}), Int64View(starts, {2}), Int64View(ends, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Slice"), std::string::npos) << error.what();
    }
}

// Slice rejects a non-positive step, naming the op.
TEST(EngineKernelsNegative, SliceRejectsNonPositiveStep) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(3, 0.0f);
    std::vector<int64_t> starts = {0};
    std::vector<int64_t> ends = {3};
    std::vector<int64_t> axes = {0};
    std::vector<int64_t> steps = {0};
    engine::Node node = MakeNode("slice_step", "Slice");
    std::vector<engine::TensorView> inputs = {FloatView(in, {3}), Int64View(starts, {1}), Int64View(ends, {1}),
                                              Int64View(axes, {1}), Int64View(steps, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Slice"), std::string::npos) << error.what();
    }
}

// Gather rejects an out-of-range index, naming the op.
TEST(EngineKernelsNegative, GatherRejectsIndexOutOfRange) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Gather");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(1, 0.0f);
    std::vector<int64_t> indices = {5};
    engine::Node node = MakeNode("gather_bad", "Gather");
    std::vector<engine::TensorView> inputs = {FloatView(in, {3}), Int64View(indices, {1})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Gather"), std::string::npos) << error.what();
    }
}

// Gather rejects an out-of-range axis, naming the op.
TEST(EngineKernelsNegative, GatherRejectsAxisOutOfRange) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Gather");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<int64_t> indices = {0, 1};
    engine::Node node = MakeNode("gather_axis", "Gather");
    node.attributes.emplace_back("axis", int64_t{2});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(indices, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Gather"), std::string::npos) << error.what();
    }
}

// GatherElements rejects data/indices of different rank, naming the op.
TEST(EngineKernelsNegative, GatherElementsRejectsRankMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("GatherElements");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> indices = {0, 1};
    engine::Node node = MakeNode("gatherelem_bad", "GatherElements");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(indices, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("GatherElements"), std::string::npos) << error.what();
    }
}

// GatherElements rejects an out-of-range index, naming the op.
TEST(EngineKernelsNegative, GatherElementsRejectsIndexOutOfRange) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("GatherElements");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> indices = {0, 7};
    engine::Node node = MakeNode("gatherelem_idx", "GatherElements");
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(indices, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("GatherElements"), std::string::npos) << error.what();
    }
}

// Resize rejects a sizes input when scales are given, naming the op.
TEST(EngineKernelsNegative, ResizeRejectsSizesInput) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<float> scales = {1.0f, 2.0f};
    std::vector<int64_t> sizes = {1, 4};
    engine::Node node = MakeNode("resize_bad", "Resize");
    std::vector<engine::TensorView> inputs = {FloatView(in, {1, 2}), engine::TensorView{}, FloatView(scales, {2}),
                                              Int64View(sizes, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 4})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Resize"), std::string::npos) << error.what();
    }
}

// Resize rejects scales whose rank misses the input rank, naming the op.
TEST(EngineKernelsNegative, ResizeRejectsScalesRankMismatch) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<float> scales = {2.0f, 2.0f, 2.0f};
    engine::Node node = MakeNode("resize_scales", "Resize");
    std::vector<engine::TensorView> inputs = {FloatView(in, {1, 2}), engine::TensorView{},
                                              FloatView(scales, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 4})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Resize"), std::string::npos) << error.what();
    }
}

// Flatten rejects an out-of-range axis, naming the op.
TEST(EngineKernelsNegative, FlattenRejectsAxisOutOfRange) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Flatten");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    engine::Node node = MakeNode("flatten_bad", "Flatten");
    node.attributes.emplace_back("axis", int64_t{3});
    std::vector<engine::TensorView> inputs = {FloatView(in, {2, 2})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1, 4})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Flatten"), std::string::npos) << error.what();
    }
}

// Shape rejects a non-int64 output, naming the op.
TEST(EngineKernelsNegative, ShapeRejectsNonInt64Output) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("Shape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(1, 0.0f);
    engine::Node node = MakeNode("shape_bad", "Shape");
    std::vector<engine::TensorView> inputs = {FloatView(in, {3})};
    std::vector<engine::TensorView> outputs = {FloatView(out, {1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Shape"), std::string::npos) << error.what();
    }
}

// ConstantOfShape rejects a negative shape value, naming the op.
TEST(EngineKernelsNegative, ConstantOfShapeRejectsNegativeDim) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {2, -1};
    std::vector<int64_t> out(2, 0);
    engine::Node node = MakeNode("cos_bad", "ConstantOfShape");
    node.attributes.emplace_back("value", int64_t{0});
    std::vector<engine::TensorView> inputs = {Int64View(shape, {2})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ConstantOfShape"), std::string::npos) << error.what();
    }
}

// ConstantOfShape rejects a missing value attribute, naming the op.
TEST(EngineKernelsNegative, ConstantOfShapeRejectsMissingValue) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {2};
    std::vector<int64_t> out(2, 0);
    engine::Node node = MakeNode("cos_novalue", "ConstantOfShape");
    std::vector<engine::TensorView> inputs = {Int64View(shape, {1})};
    std::vector<engine::TensorView> outputs = {Int64View(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ConstantOfShape"), std::string::npos) << error.what();
    }
}

// TopK rejects K larger than the axis dim, naming the op.
TEST(EngineKernelsNegative, TopKRejectsKExceedsDim) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("TopK");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> values(3, 0.0f);
    std::vector<int64_t> k = {4};
    std::vector<int64_t> indices(3, 0);
    engine::Node node = MakeNode("topk_bad", "TopK");
    std::vector<engine::TensorView> inputs = {FloatView(in, {3}), Int64View(k, {})};
    std::vector<engine::TensorView> outputs = {FloatView(values, {3}), Int64View(indices, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("TopK"), std::string::npos) << error.what();
    }
}

// TopK rejects largest=0, naming the op.
TEST(EngineKernelsNegative, TopKRejectsLargestZero) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("TopK");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> values(2, 0.0f);
    std::vector<int64_t> k = {2};
    std::vector<int64_t> indices(2, 0);
    engine::Node node = MakeNode("topk_largest", "TopK");
    node.attributes.emplace_back("largest", int64_t{0});
    std::vector<engine::TensorView> inputs = {FloatView(in, {3}), Int64View(k, {})};
    std::vector<engine::TensorView> outputs = {FloatView(values, {2}), Int64View(indices, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("TopK"), std::string::npos) << error.what();
    }
}

// TopK rejects a non-scalar K input, naming the op.
TEST(EngineKernelsNegative, TopKRejectsNonScalarK) {
    engine::KernelFn kernel = engine::CpuDevice().kernels().find("TopK");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> values(2, 0.0f);
    std::vector<int64_t> k = {2, 2};
    std::vector<int64_t> indices(2, 0);
    engine::Node node = MakeNode("topk_kscalar", "TopK");
    std::vector<engine::TensorView> inputs = {FloatView(in, {3}), Int64View(k, {2})};
    std::vector<engine::TensorView> outputs = {FloatView(values, {2}), Int64View(indices, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const engine::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("TopK"), std::string::npos) << error.what();
    }
}
