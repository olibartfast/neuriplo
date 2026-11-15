#include "GGMLOnnx.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

std::string fixture(const char* name) { return std::string(GGML_TEST_FIXTURE_DIR) + "/" + name; }

} // namespace

TEST(GGMLOnnxTest, MultiInputAdd) {
    auto model = ggml_onnx::load(fixture("multi_input_add.onnx"));
    ASSERT_EQ(model->inputs.size(), 2u);
    EXPECT_EQ(model->inputs[0].name, "a");
    EXPECT_EQ(model->inputs[1].name, "b");
    EXPECT_EQ(model->inputs[0].shape, (std::vector<int64_t>{2, 3}));
    ASSERT_EQ(model->output_names.size(), 1u);
    EXPECT_EQ(model->output_names[0], "y");

    const auto outputs = model->graph.compute({
        f32_blob({1, 2, 3, 4, 5, 6}),
        f32_blob({10, 20, 30, 40, 50, 60}),
    });
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {11.5f, 22.5f, 33.5f, 44.5f, 55.5f, 66.5f});
}

TEST(GGMLOnnxTest, Int64GatherInput) {
    auto model = ggml_onnx::load(fixture("int64_gather.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    EXPECT_EQ(model->inputs[0].name, "indices");
    EXPECT_EQ(model->inputs[0].dtype, ggml_graph::DType::I64);
    EXPECT_EQ(model->inputs[0].shape, (std::vector<int64_t>{2}));

    const auto outputs = model->graph.compute({i64_blob({2, 0})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {5, 6, 1, 2});
}

TEST(GGMLOnnxTest, AttentionBlock) {
    auto model = ggml_onnx::load(fixture("attention_block.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    EXPECT_EQ(model->inputs[0].name, "x");

    const auto outputs = model->graph.compute({f32_blob({1, 2, 3, 4})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {3.3395233f, 2.3395233f, 3.651452f, 2.651452f}, 1e-5f);
}

TEST(GGMLOnnxTest, ShapeAndActivationOps) {
    auto model = ggml_onnx::load(fixture("shape_ops.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);

    const auto outputs = model->graph.compute({f32_blob({1, 2, 3, -1, -2, -3})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {1.0f, 2.0f, 3.0f, 0.7310586f, 0.880797f, 0.95257413f, 0.0f, 0.0f, 0.0f,
                                       0.2689414f, 0.11920292f, 0.047425874f});
}

TEST(GGMLOnnxTest, ConvAndMaxPool) {
    auto model = ggml_onnx::load(fixture("conv_pool.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    std::vector<float> input(32);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }
    std::vector<uint8_t> blob(input.size() * sizeof(float));
    std::memcpy(blob.data(), input.data(), blob.size());

    const auto outputs = model->graph.compute({blob});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]),
                {-8.077398f, -5.1446795f, -5.1364913f, -7.4380975f, -7.796434f, -7.7885165f, -17.035841f, -20.485954f,
                 21.763773f, 7.3586116f, 26.805868f, 11.698821f},
                1e-5f);
}

TEST(GGMLOnnxTest, DepthwiseConv) {
    auto model = ggml_onnx::load(fixture("depthwise_conv.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    std::vector<float> input(32);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }
    std::vector<uint8_t> blob(input.size() * sizeof(float));
    std::memcpy(blob.data(), input.data(), blob.size());

    const auto outputs = model->graph.compute({blob});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]),
                {0.6196508f, 1.0604091f, 1.0453864f, 0.21335392f, 1.2738626f, 2.2858884f,   2.7945983f,   1.0859759f,
                 3.2338116f, 4.3207273f, 4.8294373f, 1.9654365f,  2.6955914f, 1.7821629f,   2.038763f,    0.9041944f,
                 0.692745f,  4.972358f,  5.2223196f, 8.077144f,   5.75691f,   4.964433f,    5.138953f,    8.788653f,
                 7.018058f,  5.662512f,  5.837032f,  10.0291815f, 4.2146206f, -0.88491756f, -0.92741674f, 2.091441f},
                1e-5f);
}

TEST(GGMLOnnxTest, ConstantOfShape) {
    auto model = ggml_onnx::load(fixture("constant_of_shape.onnx"));
    EXPECT_TRUE(model->inputs.empty());
    const auto outputs = model->graph.compute({});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
}

TEST(GGMLOnnxTest, ScaledGemm) {
    auto model = ggml_onnx::load(fixture("gemm_scaled.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    const auto outputs = model->graph.compute({f32_blob({1, 2, 3, 4})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {1.3f, 0.175f, 2.8f, 1.175f});
}

TEST(GGMLOnnxTest, BatchedMatMul) {
    auto model = ggml_onnx::load(fixture("batched_matmul.onnx"));
    ASSERT_EQ(model->inputs.size(), 2u);
    const auto outputs = model->graph.compute({
        f32_blob({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
        f32_blob({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f}),
    });
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {2.2f, 2.8000002f, 4.9f, 6.4f, 22.0f, 24.400002f, 30.1f, 33.4f}, 1e-5f);
}

TEST(GGMLOnnxTest, RankThreeTranspose) {
    auto model = ggml_onnx::load(fixture("transpose3.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    const auto outputs = model->graph.compute(
        {f32_blob({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]),
                {1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24});
}

TEST(GGMLOnnxTest, Erf) {
    auto model = ggml_onnx::load(fixture("erf.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    const auto outputs = model->graph.compute({f32_blob({-2, -1, 0, 1, 2})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {-0.995322287f, -0.842700779f, 0.0f, 0.842700779f, 0.995322287f}, 1e-6f);
}

TEST(GGMLOnnxTest, ResizeNearest) {
    auto model = ggml_onnx::load(fixture("resize_nearest.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    const auto outputs = model->graph.compute({f32_blob({1, 2, 3, 4})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});
}

TEST(GGMLOnnxTest, Split) {
    auto model = ggml_onnx::load(fixture("split.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    const auto outputs = model->graph.compute({f32_blob({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})});
    ASSERT_EQ(outputs.size(), 2u);
    expect_near(read_f32(outputs[0]), {1, 2, 3, 4, 5, 6, 7, 8});
    expect_near(read_f32(outputs[1]), {9, 10, 11, 12, 13, 14, 15, 16});
}

TEST(GGMLOnnxTest, TopK) {
    auto model = ggml_onnx::load(fixture("topk.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    const auto outputs = model->graph.compute({f32_blob({1, 4, 2, 5, 3})});
    ASSERT_EQ(outputs.size(), 2u);
    const std::vector<float> values = read_f32(outputs[0]);
    const std::vector<int32_t> indices = read_i32(outputs[1]);
    ASSERT_EQ(values.size(), 2u);
    ASSERT_EQ(indices.size(), 2u);
    std::vector<float> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    EXPECT_EQ(sorted_values, (std::vector<float>{4, 5}));
    std::vector<int32_t> sorted_indices = indices;
    std::sort(sorted_indices.begin(), sorted_indices.end());
    EXPECT_EQ(sorted_indices, (std::vector<int32_t>{1, 3}));
    const std::vector<float> input = {1, 4, 2, 5, 3};
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_FLOAT_EQ(values[i], input[static_cast<size_t>(indices[i])]);
    }
}

TEST(GGMLOnnxTest, ExpandAndTile) {
    auto model = ggml_onnx::load(fixture("expand_tile.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    const auto outputs = model->graph.compute({f32_blob({1, 2, 3})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
}

TEST(GGMLOnnxTest, GatherElements) {
    auto model = ggml_onnx::load(fixture("gather_elements.onnx"));
    ASSERT_EQ(model->inputs.size(), 2u);
    const auto outputs = model->graph.compute({
        f32_blob({1, 2, 3, 4, 5, 6, 7, 8}),
        i64_blob({1, 0, 3, 2}),
    });
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {3, 2, 7, 6});
}

TEST(GGMLOnnxTest, SliceReverseAndStep) {
    auto model = ggml_onnx::load(fixture("slice_ops.onnx"));
    ASSERT_EQ(model->inputs.size(), 2u);
    const auto outputs = model->graph.compute({
        f32_blob({1, 2, 3, 4, 5, 6, 7, 8}),
        f32_blob({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
    });
    ASSERT_EQ(outputs.size(), 2u);
    expect_near(read_f32(outputs[0]), {7, 8, 5, 6, 3, 4, 1, 2});
    expect_near(read_f32(outputs[1]), {2, 3, 4, 5, 7, 8, 9, 10});
}

TEST(GGMLOnnxTest, Pad) {
    auto model = ggml_onnx::load(fixture("pad.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    const auto outputs = model->graph.compute({f32_blob({1, 2, 3, 4})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0});
}

TEST(GGMLOnnxTest, ReduceMeanMaxSum) {
    auto model = ggml_onnx::load(fixture("reduce_ops.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    std::vector<float> input(24);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }
    std::vector<uint8_t> blob(input.size() * sizeof(float));
    std::memcpy(blob.data(), input.data(), blob.size());
    const auto outputs = model->graph.compute({blob});
    ASSERT_EQ(outputs.size(), 3u);
    expect_near(read_f32(outputs[0]), {2.5f, 6.5f, 10.5f, 14.5f, 18.5f, 22.5f});
    expect_near(read_f32(outputs[1]), {4, 8, 12, 16, 20, 24});
    expect_near(read_f32(outputs[2]), {10, 26, 42, 58, 74, 90});
}

TEST(GGMLOnnxTest, ShapeSubgraphFolding) {
    auto model = ggml_onnx::load(fixture("shape_fold.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    std::vector<float> input(24);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }
    std::vector<uint8_t> blob(input.size() * sizeof(float));
    std::memcpy(blob.data(), input.data(), blob.size());
    const auto outputs = model->graph.compute({blob});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), input);
}

TEST(GGMLOnnxTest, BroadcastBothSides) {
    auto model = ggml_onnx::load(fixture("broadcast_edges.onnx"));
    ASSERT_EQ(model->inputs.size(), 2u);
    const auto outputs = model->graph.compute({
        f32_blob({1, 2}),
        f32_blob({10, 20, 30}),
    });
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {11, 21, 31, 12, 22, 32});
}

TEST(GGMLOnnxTest, GatherElementsRejectsRankTwo) {
    EXPECT_THROW(
        {
            try {
                (void)ggml_onnx::load(fixture("gather_elements_rank2.onnx"));
            } catch (const std::invalid_argument& error) {
                EXPECT_NE(std::string(error.what()).find("rank-3 input with axis=1"), std::string::npos)
                    << error.what();
                throw;
            }
        },
        std::invalid_argument);
}

TEST(GGMLOnnxTest, GatherElementsOutOfRangeBecomesError) {
    auto model = ggml_onnx::load(fixture("gather_elements_oob.onnx"));
    EXPECT_THROW(model->graph.compute({
                     f32_blob({1, 2, 3, 4, 5, 6, 7, 8}),
                     i64_blob({99, 0, 3, 2}),
                 }),
                 std::runtime_error);
}

TEST(GGMLOnnxTest, SliceRejectsStepTwo) {
    EXPECT_THROW(
        {
            try {
                (void)ggml_onnx::load(fixture("slice_step2.onnx"));
            } catch (const std::invalid_argument& error) {
                EXPECT_NE(std::string(error.what()).find("step != 1"), std::string::npos) << error.what();
                throw;
            }
        },
        std::invalid_argument);
}

TEST(GGMLOnnxTest, PadRejectsBeginPadding) {
    EXPECT_THROW(
        {
            try {
                (void)ggml_onnx::load(fixture("pad_begin.onnx"));
            } catch (const std::invalid_argument& error) {
                EXPECT_NE(std::string(error.what()).find("begin padding of zero only"), std::string::npos)
                    << error.what();
                throw;
            }
        },
        std::invalid_argument);
}

TEST(GGMLOnnxTest, GridSampleBilinear) {
    auto model = ggml_onnx::load(fixture("grid_sample.onnx"));
    ASSERT_EQ(model->inputs.size(), 1u);
    const auto outputs = model->graph.compute({f32_blob({1, 2, 3, 4})});
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {1, 2, 3, 4});
}

TEST(GGMLOnnxTest, MatMulVector) {
    auto model = ggml_onnx::load(fixture("matmul_vector.onnx"));
    ASSERT_EQ(model->inputs.size(), 2u);
    const auto outputs = model->graph.compute({
        f32_blob({1, 2, 3, 4, 5, 6}),
        f32_blob({1, 2, 3}),
    });
    ASSERT_EQ(outputs.size(), 1u);
    expect_near(read_f32(outputs[0]), {14, 32});
}

TEST(GGMLOnnxTest, RejectsUnsupportedOp) {
    EXPECT_THROW(
        {
            try {
                (void)ggml_onnx::load(fixture("unsupported_softplus.onnx"));
            } catch (const std::invalid_argument& error) {
                EXPECT_NE(std::string(error.what()).find("unsupported ONNX op 'Softplus"), std::string::npos)
                    << error.what();
                throw;
            }
        },
        std::invalid_argument);
}

TEST(GGMLOnnxTest, RejectsDynamicShape) {
    EXPECT_THROW(
        {
            try {
                (void)ggml_onnx::load(fixture("dynamic_shape.onnx"));
            } catch (const std::invalid_argument& error) {
                EXPECT_NE(std::string(error.what()).find("dynamic shape"), std::string::npos) << error.what();
                throw;
            }
        },
        std::invalid_argument);
}
