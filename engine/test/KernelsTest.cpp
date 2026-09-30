// CPU kernel tests: Relu, Add, Reshape, and ReduceMean.
//
// Expected values are hand-computed small cases written inline, not captured
// from another runtime. Kernels are reached through the CPU device's kernel
// table (CpuDevice().kernels().find(...)), the same way the executor will.

#include "engine/Device.hpp"

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

// The op-type table resolves the implemented kernels and rejects unknown ops.
TEST(EngineKernels, TableResolvesKnownOpsAndRejectsUnknown) {
    const engine::KernelTable& table = engine::CpuDevice().kernels();
    EXPECT_NE(table.find("Relu"), nullptr);
    EXPECT_NE(table.find("Add"), nullptr);
    EXPECT_NE(table.find("Reshape"), nullptr);
    EXPECT_NE(table.find("ReduceMean"), nullptr);
    EXPECT_NE(table.find("Gemm"), nullptr);
    EXPECT_NE(table.find("MatMul"), nullptr);
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
