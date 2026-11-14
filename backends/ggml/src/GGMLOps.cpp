#include "GGMLOps.hpp"

#include <cmath>
#include <stdexcept>

namespace ggml_ops {

struct ggml_tensor* layer_norm(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* weight,
                               struct ggml_tensor* bias, float eps) {
    if (weight == nullptr || bias == nullptr) {
        throw std::invalid_argument("layer_norm requires weight and bias");
    }
    struct ggml_tensor* normalized = ggml_norm(ctx, x, eps);
    struct ggml_tensor* scaled = ggml_mul(ctx, normalized, ggml_repeat(ctx, weight, normalized));
    return ggml_add(ctx, scaled, ggml_repeat(ctx, bias, scaled));
}

struct ggml_tensor* gelu(struct ggml_context* ctx, struct ggml_tensor* x) { return ggml_gelu_erf(ctx, x); }

struct ggml_tensor* matmul(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
    if (a == nullptr || b == nullptr) {
        throw std::invalid_argument("matmul requires both operands");
    }
    if (a->ne[0] != b->ne[0]) {
        throw std::invalid_argument("matmul requires a->ne[0] == b->ne[0]");
    }
    return ggml_mul_mat(ctx, a, b);
}

struct ggml_tensor* linear(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* weight,
                           struct ggml_tensor* bias) {
    struct ggml_tensor* y = matmul(ctx, weight, x);
    if (bias != nullptr) {
        y = ggml_add(ctx, y, ggml_repeat(ctx, bias, y));
    }
    return y;
}

struct ggml_tensor* scaled_dot_product_attention(struct ggml_context* ctx, struct ggml_tensor* q, struct ggml_tensor* k,
                                                 struct ggml_tensor* v) {
    if (q == nullptr || k == nullptr || v == nullptr) {
        throw std::invalid_argument("attention requires q, k and v");
    }
    if (q->ne[0] != k->ne[0] || k->ne[0] != v->ne[0]) {
        throw std::invalid_argument("attention head dimensions must match");
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(q->ne[0]));
    struct ggml_tensor* scores = ggml_mul_mat(ctx, k, q); // [seq_k, seq_q, batch]
    scores = ggml_scale(ctx, scores, scale);
    scores = ggml_soft_max(ctx, scores);
    // V is stored [head_dim, seq_k]; mul_mat contracts on ne[0], so transpose
    // it to [seq_k, head_dim] before multiplying by the [seq_k, seq_q] scores.
    struct ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    return ggml_mul_mat(ctx, v_t, scores); // [head_dim, seq_q, batch]
}

struct ggml_tensor* gather_rows(struct ggml_context* ctx, struct ggml_tensor* data, struct ggml_tensor* indices) {
    if (data == nullptr || indices == nullptr) {
        throw std::invalid_argument("gather_rows requires data and indices");
    }
    if (indices->type != GGML_TYPE_I32) {
        throw std::invalid_argument("gather_rows requires I32 indices (cast ONNX int64 first)");
    }
    return ggml_get_rows(ctx, data, indices);
}

struct ggml_tensor* top_k_indices(struct ggml_context* ctx, struct ggml_tensor* x, int k) {
    if (k <= 0 || k > x->ne[0]) {
        throw std::invalid_argument("top_k_indices requires 0 < k <= ne[0]");
    }
    return ggml_top_k(ctx, x, k);
}

} // namespace ggml_ops
