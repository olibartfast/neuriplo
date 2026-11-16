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

// Softmax over the last ONNX dimension, which the GGML layout stores at
// ne[0], so every ne[0] slice is one normalized row. x is a contiguous F32
// tensor [C, ...]; the result has the same shape and each row sums to one
// (ONNX Softmax(axis=-1)).
//
// This is a CPU GGML_OP_CUSTOM kernel rather than a wrapper around
// ggml_soft_max: RF-DETR chains 17 Softmax nodes, so the accumulator is F64
// to keep the ORT-relative drift inside the measured budget, and the row
// geometry is not tied to ggml's softmax fast-path assumptions. Use it where
// the reduced axis is the ONNX last axis; when the reduced axis maps to a
// different GGML dimension, permute/reshape to ne[0] first.
struct ggml_tensor* softmax_rows(struct ggml_context* ctx, struct ggml_tensor* x);

// ONNX TopK indices along ne[0] (I32 [k, ...]), sorted by descending score
// with ties broken by ascending index. Implemented as a CPU GGML_OP_CUSTOM
// kernel because ggml_top_k is unordered and ggml_argsort_top_k does not
// guarantee a deterministic tie-break. Requires a contiguous F32 input. Use
// gather_rows to materialize the corresponding values when the ONNX graph
// needs both outputs.
struct ggml_tensor* top_k_indices(struct ggml_context* ctx, struct ggml_tensor* x, int k);

// ONNX GridSample in the deformable-attention configuration:
// mode=bilinear, padding_mode=zeros, align_corners=0.
//
// input [W,H,C,N] (F32) and grid [2,Wout,Hout,N] (F32, components (x,y) in
// [-1,1]) -> output [Wout,Hout,C,N]. Unnormalized source coordinates are
// ix = ((gx + 1) * W - 1) / 2 and iy = ((gy + 1) * H - 1) / 2; the four
// neighbouring pixels are combined bilinearly, and any corner outside
// [0,W) x [0,H) contributes zero (zeros padding, no edge clamp). Non-finite
// grid values (NaN/Inf, or an overflowed coordinate) yield a zero output
// element. Implemented as a CPU GGML_OP_CUSTOM kernel with src[0]=input and
// src[1]=grid; the backend is CPU-only.
struct ggml_tensor* grid_sample(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* grid);

} // namespace ggml_ops
