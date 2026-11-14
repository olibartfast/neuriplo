#include "GGMLOps.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

struct GgmlFixture {
    struct ggml_context* ctx = nullptr;

    GgmlFixture() {
        ggml_init_params params{};
        params.mem_size = 64 * 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("ggml_init failed");
        }
    }

    ~GgmlFixture() {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    struct ggml_tensor* f32(std::initializer_list<int64_t> dims, const std::vector<float>& data) {
        struct ggml_tensor* t = new_tensor(dims, GGML_TYPE_F32);
        EXPECT_EQ(static_cast<size_t>(ggml_nelements(t)), data.size());
        std::memcpy(t->data, data.data(), data.size() * sizeof(float));
        return t;
    }

    struct ggml_tensor* i32(std::initializer_list<int64_t> dims, const std::vector<int32_t>& data) {
        struct ggml_tensor* t = new_tensor(dims, GGML_TYPE_I32);
        EXPECT_EQ(static_cast<size_t>(ggml_nelements(t)), data.size());
        std::memcpy(t->data, data.data(), data.size() * sizeof(int32_t));
        return t;
    }

    void compute(struct ggml_tensor* output) {
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
    }

  private:
    struct ggml_tensor* new_tensor(std::initializer_list<int64_t> dims, enum ggml_type type) {
        if (dims.size() == 1) {
            return ggml_new_tensor_1d(ctx, type, *dims.begin());
        }
        if (dims.size() == 2) {
            auto it = dims.begin();
            return ggml_new_tensor_2d(ctx, type, *it, *(it + 1));
        }
        if (dims.size() == 3) {
            auto it = dims.begin();
            return ggml_new_tensor_3d(ctx, type, *it, *(it + 1), *(it + 2));
        }
        throw std::runtime_error("unsupported test tensor rank");
    }
};

std::vector<float> read_f32(struct ggml_tensor* t) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(t)));
    std::memcpy(values.data(), t->data, values.size() * sizeof(float));
    return values;
}

std::vector<int32_t> read_i32(struct ggml_tensor* t) {
    std::vector<int32_t> values(static_cast<size_t>(ggml_nelements(t)));
    std::memcpy(values.data(), t->data, values.size() * sizeof(int32_t));
    return values;
}

void expect_near(const std::vector<float>& got, const std::vector<float>& expected, float tolerance = 1e-4f) {
    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], tolerance) << "index " << i;
    }
}

} // namespace

TEST(GGMLOpsParity, MatMulAndLinear) {
    GgmlFixture f;
    // ONNX A [3,2] and B [2,2], stored with reversed axes as [K,M] / [K,N].
    struct ggml_tensor* a = f.f32({2, 3}, {1, 2, 3, 4, 5, 6});
    struct ggml_tensor* b = f.f32({2, 2}, {7, 9, 8, 10});
    struct ggml_tensor* y = ggml_ops::matmul(f.ctx, a, b);
    f.compute(y);
    // ggml [M,N] flat order is m-fastest: [25,57,89, 28,64,100].
    expect_near(read_f32(y), {25, 57, 89, 28, 64, 100});

    struct ggml_tensor* bias = f.f32({2}, {1, 2});
    struct ggml_tensor* z = ggml_ops::linear(f.ctx, a, b, bias);
    f.compute(z);
    expect_near(read_f32(z), {26, 30, 58, 66, 90, 102});
}

TEST(GGMLOpsParity, LayerNorm) {
    GgmlFixture f;
    // ONNX [S=2, D=4] stored as ggml [D=4, S=2] (D fastest).
    struct ggml_tensor* x = f.f32({4, 2}, {1, 2, 3, 4, 2, 4, 6, 8});
    struct ggml_tensor* weight = f.f32({4}, {2, 2, 2, 2});
    struct ggml_tensor* bias = f.f32({4}, {1, 1, 1, 1});
    struct ggml_tensor* y = ggml_ops::layer_norm(f.ctx, x, weight, bias, 1e-5f);
    f.compute(y);
    expect_near(read_f32(y),
                {-1.6832709f, 0.10557634f, 1.8944237f, 3.6832709f, -1.6832790f, 0.10557365f, 1.8944263f, 3.6832790f});
}

TEST(GGMLOpsParity, GeluErf) {
    GgmlFixture f;
    struct ggml_tensor* x = f.f32({5}, {-2, -1, 0, 1, 2});
    struct ggml_tensor* y = ggml_ops::gelu(f.ctx, x);
    f.compute(y);
    expect_near(read_f32(y), {-0.045500264f, -0.15865526f, 0.0f, 0.84134475f, 1.9544997f}, 1e-5f);
}

TEST(GGMLOpsParity, ScaledDotProductAttention) {
    GgmlFixture f;
    // ONNX q/k/v [S=2, D=2], stored as [D=2, S=2].
    struct ggml_tensor* q = f.f32({2, 2}, {1, 0, 0, 1});
    struct ggml_tensor* k = f.f32({2, 2}, {1, 0, 0, 1});
    struct ggml_tensor* v = f.f32({2, 2}, {1, 2, 3, 4});
    struct ggml_tensor* y = ggml_ops::scaled_dot_product_attention(f.ctx, q, k, v);
    f.compute(y);
    // ggml [D,S] flat order is D-fastest: [o00,o10,o01,o11].
    expect_near(read_f32(y), {1.6604769f, 2.6604769f, 2.3395231f, 3.3395231f}, 1e-5f);
}

TEST(GGMLOpsParity, TinyTransformerBlock) {
    GgmlFixture f;
    // x [S=2, D=2] stored [D=2, S=2]; W1 [out=3, in=2] stored [in, out];
    // W2 [out=2, in=3] stored [in, out].
    struct ggml_tensor* x = f.f32({2, 2}, {1, 2, 3, 4});
    struct ggml_tensor* ln_w = f.f32({2}, {1, 1});
    struct ggml_tensor* ln_b = f.f32({2}, {0, 0});
    struct ggml_tensor* w1 = f.f32({2, 3}, {1, 0, 0, 1, 1, 1});
    struct ggml_tensor* b1 = f.f32({3}, {0, 0, 0});
    struct ggml_tensor* w2 = f.f32({3, 2}, {1, 1, 0, 0, 1, 1});
    struct ggml_tensor* b2 = f.f32({2}, {0, 0});
    struct ggml_tensor* h = ggml_ops::layer_norm(f.ctx, x, ln_w, ln_b, 1e-5f);
    h = ggml_ops::linear(f.ctx, h, w1, b1);
    h = ggml_ops::gelu(f.ctx, h);
    h = ggml_ops::linear(f.ctx, h, w2, b2);
    struct ggml_tensor* out = ggml_add(f.ctx, x, h);
    f.compute(out);
    expect_near(read_f32(out), {1.6826661f, 2.8413231f, 3.6826661f, 4.8413231f}, 1e-5f);
}

TEST(GGMLOpsParity, GatherAndTopK) {
    GgmlFixture f;
    // data [N=3, D=2] stored as [D=2, N=3].
    struct ggml_tensor* data = f.f32({2, 3}, {1, 2, 3, 4, 5, 6});
    struct ggml_tensor* indices = f.i32({2}, {2, 0});
    struct ggml_tensor* gathered = ggml_ops::gather_rows(f.ctx, data, indices);
    f.compute(gathered);
    expect_near(read_f32(gathered), {5, 6, 1, 2});

    struct ggml_tensor* logits = f.f32({4}, {1, 4, 2, 3});
    struct ggml_tensor* top = ggml_ops::top_k_indices(f.ctx, logits, 2);
    f.compute(top);
    std::vector<int32_t> got = read_i32(top);
    ASSERT_EQ(got.size(), 2u);
    // ggml explicitly does not guarantee top-k order; compare as a set.
    EXPECT_EQ(got[0] == 1 || got[0] == 3, true);
    EXPECT_EQ(got[1] == 1 || got[1] == 3, true);
    EXPECT_NE(got[0], got[1]);
}

TEST(GGMLOpsParity, RejectsUnsupportedContracts) {
    GgmlFixture f;
    struct ggml_tensor* a = f.f32({2, 2}, {1, 2, 3, 4});
    struct ggml_tensor* b = f.f32({3, 2}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(ggml_ops::matmul(f.ctx, a, b), std::invalid_argument);

    struct ggml_tensor* data = f.f32({2, 2}, {1, 2, 3, 4});
    struct ggml_tensor* i64 = f.f32({2}, {0, 1});
    EXPECT_THROW(ggml_ops::gather_rows(f.ctx, data, i64), std::invalid_argument);
    EXPECT_THROW(ggml_ops::top_k_indices(f.ctx, data, 0), std::invalid_argument);
}
