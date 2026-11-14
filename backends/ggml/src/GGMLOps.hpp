#pragma once

#include <ggml.h>

// Small, reusable GGML op helpers for transformer-style graphs.
//
// Layout convention: ONNX tensors are stored with reversed axes, matching the
// rest of GGMLInfer (NCHW -> [W,H,C,N]). A transformer activation
// [batch, seq, hidden] is therefore ggml [hidden, seq, batch], so the
// normalized/reduced axis is always ne[0].
namespace ggml_ops {

// LayerNorm over ne[0] with per-feature weight/bias. weight may be [D] or
// [D,1,1,1]; bias must be broadcastable to the normalized tensor.
struct ggml_tensor* layer_norm(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* weight,
                               struct ggml_tensor* bias, float eps);

// Exact (erf-based) GELU, matching ONNX's Erf formulation.
struct ggml_tensor* gelu(struct ggml_context* ctx, struct ggml_tensor* x);

// ONNX MatMul A[M,K] * B[K,N] -> [M,N]. Store A as [K,M] and B as [K,N] so
// ggml_mul_mat (which requires a->ne[0] == b->ne[0]) produces [M,N].
struct ggml_tensor* matmul(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b);

// Linear layer: y = x * W^T + b, with W stored as [in_features, out_features].
struct ggml_tensor* linear(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* weight,
                           struct ggml_tensor* bias);

// Single-head scaled dot-product attention:
// q/k/v are [head_dim, seq, batch]; returns [head_dim, seq_q, batch].
struct ggml_tensor* scaled_dot_product_attention(struct ggml_context* ctx, struct ggml_tensor* q, struct ggml_tensor* k,
                                                 struct ggml_tensor* v);

// Row gather: data [n, m], indices I32 [r] -> [n, r]. Callers must cast
// ONNX int64 indices to I32 before calling (ggml_get_rows requires I32).
struct ggml_tensor* gather_rows(struct ggml_context* ctx, struct ggml_tensor* data, struct ggml_tensor* indices);

// Top-k indices along ne[0] (I32 [k, ...]); use gather_rows to materialize the
// corresponding values when the ONNX graph needs both outputs.
struct ggml_tensor* top_k_indices(struct ggml_context* ctx, struct ggml_tensor* x, int k);

} // namespace ggml_ops
