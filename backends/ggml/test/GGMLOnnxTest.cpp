#include "GGMLOnnx.hpp"

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

TEST(GGMLOnnxTest, RejectsUnsupportedOp) {
    EXPECT_THROW(
        {
            try {
                (void)ggml_onnx::load(fixture("unsupported_grid_sample.onnx"));
            } catch (const std::invalid_argument& error) {
                EXPECT_NE(std::string(error.what()).find("unsupported ONNX op 'GridSample"), std::string::npos)
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
