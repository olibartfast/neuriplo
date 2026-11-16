#include "GGMLOps.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// ONNX TopK with sorted=1: indices ordered by descending score, ties broken by
// ascending index. The backend is CPU-only, so a CPU GGML_OP_CUSTOM kernel is
// sufficient. k is carried in userdata; the operation has no dynamic state.
void top_k_indices_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* src = dst->src[0];
    const auto k = static_cast<int64_t>(reinterpret_cast<intptr_t>(userdata));
    const int64_t n = src->ne[0];
    const int64_t rows = dst->ne[1] * dst->ne[2] * dst->ne[3];

    const int64_t dr = (rows + nth - 1) / nth;
    const int64_t row_begin = dr * ith;
    const int64_t row_end = std::min(row_begin + dr, rows);

    std::vector<int32_t> order(static_cast<size_t>(n));
    for (int64_t row = row_begin; row < row_end; ++row) {
        const auto* values =
            reinterpret_cast<const float*>(reinterpret_cast<const char*>(src->data) + row * src->nb[1]);
        for (int64_t i = 0; i < n; ++i) {
            order[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        // stable_sort keeps the original (ascending index) order for equal
        // scores, which is the ONNX tie-break for sorted TopK.
        std::stable_sort(order.begin(), order.end(), [values](int32_t a, int32_t b) { return values[a] > values[b]; });
        auto* out = reinterpret_cast<int32_t*>(dst->data) + row * k;
        for (int64_t i = 0; i < k; ++i) {
            out[i] = order[static_cast<size_t>(i)];
        }
    }
}

// ONNX GridSample (bilinear, zeros padding, align_corners=0) as a CPU
// GGML_OP_CUSTOM kernel. dst [Wout,Hout,C,N], dst->src[0] [W,H,C,N],
// dst->src[1] grid [2,Wout,Hout,N]. Work is split over output pixels
// (x_out, y_out, n); every thread owns whole pixels, so the channel
// accumulation buffer is local. Strides are taken from nb[] because the
// inputs may be non-contiguous graph views.
void grid_sample_custom(struct ggml_tensor* dst, int ith, int nth, void* /*userdata*/) {
    const auto* input = dst->src[0];
    const auto* grid = dst->src[1];
    const int64_t W = input->ne[0];
    const int64_t H = input->ne[1];
    const int64_t C = input->ne[2];
    const int64_t Wout = dst->ne[0];
    const int64_t Hout = dst->ne[1];
    const int64_t N = dst->ne[3];

    const int64_t pixels = Wout * Hout * N;
    const int64_t dp = (pixels + nth - 1) / nth;
    const int64_t p_begin = dp * ith;
    const int64_t p_end = std::min(p_begin + dp, pixels);

    std::vector<float> acc(static_cast<size_t>(C), 0.0f);
    for (int64_t p = p_begin; p < p_end; ++p) {
        const int64_t x_out = p % Wout;
        const int64_t y_out = (p / Wout) % Hout;
        const int64_t n = p / (Wout * Hout);

        char* out = static_cast<char*>(dst->data) + x_out * dst->nb[0] + y_out * dst->nb[1] + n * dst->nb[3];
        const char* g =
            static_cast<const char*>(grid->data) + x_out * grid->nb[1] + y_out * grid->nb[2] + n * grid->nb[3];
        const float gx = *reinterpret_cast<const float*>(g);
        const float gy = *reinterpret_cast<const float*>(g + grid->nb[0]);

        // align_corners=0: map [-1,1] onto pixel centres.
        const float ix = ((gx + 1.0f) * static_cast<float>(W) - 1.0f) * 0.5f;
        const float iy = ((gy + 1.0f) * static_cast<float>(H) - 1.0f) * 0.5f;

        std::fill(acc.begin(), acc.end(), 0.0f);
        if (std::isfinite(ix) && std::isfinite(iy)) {
            const float x0f = std::floor(ix);
            const float y0f = std::floor(iy);
            const float fx = ix - x0f;
            const float fy = iy - y0f;
            const int64_t x0 = static_cast<int64_t>(x0f);
            const int64_t y0 = static_cast<int64_t>(y0f);
            for (int cx = 0; cx < 2; ++cx) {
                const int64_t xi = x0 + cx;
                if (xi < 0 || xi >= W) {
                    continue;
                }
                const float wx = cx == 0 ? 1.0f - fx : fx;
                for (int cy = 0; cy < 2; ++cy) {
                    const int64_t yi = y0 + cy;
                    if (yi < 0 || yi >= H) {
                        continue;
                    }
                    const float w = wx * (cy == 0 ? 1.0f - fy : fy);
                    const char* in = static_cast<const char*>(input->data) + xi * input->nb[0] + yi * input->nb[1] +
                                     n * input->nb[3];
                    for (int64_t c = 0; c < C; ++c) {
                        acc[static_cast<size_t>(c)] += w * *reinterpret_cast<const float*>(in + c * input->nb[2]);
                    }
                }
            }
        }
        for (int64_t c = 0; c < C; ++c) {
            *reinterpret_cast<float*>(out + c * dst->nb[2]) = acc[static_cast<size_t>(c)];
        }
    }
}

// ONNX Softmax(axis=-1) as a CPU GGML_OP_CUSTOM kernel. The last ONNX axis is
// ne[0], so one row is a contiguous ne[0] slice and there are
// ne[1]*ne[2]*ne[3] rows. Work is split over rows; the accumulator is F64 to
// hold the RF-DETR chained-softmax budget. A non-finite row maximum (all
// +inf/-inf) would make exp(x - max) produce inf-inf NaNs, so those rows get
// uniform mass over the peak entries instead.
void softmax_rows_custom(struct ggml_tensor* dst, int ith, int nth, void* /*userdata*/) {
    const auto* src = dst->src[0];
    const int64_t n = src->ne[0];
    const int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];

    const int64_t dr = (rows + nth - 1) / nth;
    const int64_t row_begin = dr * ith;
    const int64_t row_end = std::min(row_begin + dr, rows);

    for (int64_t row = row_begin; row < row_end; ++row) {
        const auto* in = reinterpret_cast<const float*>(reinterpret_cast<const char*>(src->data) + row * src->nb[1]);
        auto* out = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + row * dst->nb[1]);

        double max = -std::numeric_limits<double>::infinity();
        for (int64_t i = 0; i < n; ++i) {
            max = std::max(max, static_cast<double>(in[i]));
        }
        if (!std::isfinite(max)) {
            int64_t peak = 0;
            for (int64_t i = 0; i < n; ++i) {
                if (static_cast<double>(in[i]) == max) {
                    ++peak;
                }
            }
            const float uniform = peak > 0 ? 1.0f / static_cast<float>(peak) : 0.0f;
            for (int64_t i = 0; i < n; ++i) {
                out[i] = static_cast<double>(in[i]) == max ? uniform : 0.0f;
            }
            continue;
        }

        double sum = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const float e = static_cast<float>(std::exp(static_cast<double>(in[i]) - max));
            out[i] = e;
            sum += e;
        }
        const float denom = static_cast<float>(sum);
        for (int64_t i = 0; i < n; ++i) {
            out[i] /= denom;
        }
    }
}

bool checked_mul(int64_t a, int64_t b, int64_t& out) {
    if (a < 0 || b < 0) {
        return false;
    }
    if (a != 0 && b > INT64_MAX / a) {
        return false;
    }
    out = a * b;
    return true;
}

} // namespace

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

struct ggml_tensor* softmax_rows(struct ggml_context* ctx, struct ggml_tensor* x) {
    if (x == nullptr) {
        throw std::invalid_argument("softmax_rows requires an input tensor");
    }
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) {
        throw std::invalid_argument("softmax_rows requires a contiguous F32 input");
    }
    if (x->ne[0] <= 0) {
        throw std::invalid_argument("softmax_rows requires a positive row length");
    }
    int64_t count = 0;
    if (!checked_mul(x->ne[0], x->ne[1], count) || !checked_mul(count, x->ne[2], count) ||
        !checked_mul(count, x->ne[3], count)) {
        throw std::invalid_argument("softmax_rows element count overflows int64");
    }
    struct ggml_tensor* args[1] = {x};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3], args, 1, softmax_rows_custom,
                          /*n_tasks=*/1, nullptr);
}

struct ggml_tensor* top_k_indices(struct ggml_context* ctx, struct ggml_tensor* x, int k) {
    if (x == nullptr) {
        throw std::invalid_argument("top_k_indices requires an input tensor");
    }
    if (k <= 0 || k > x->ne[0]) {
        throw std::invalid_argument("top_k_indices requires 0 < k <= ne[0]");
    }
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) {
        throw std::invalid_argument("top_k_indices requires a contiguous F32 input");
    }

    struct ggml_tensor* args[1] = {x};
    return ggml_custom_4d(ctx, GGML_TYPE_I32, k, x->ne[1], x->ne[2], x->ne[3], args, 1, top_k_indices_custom,
                          /*n_tasks=*/1, reinterpret_cast<void*>(static_cast<intptr_t>(k)));
}

struct ggml_tensor* grid_sample(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* grid) {
    if (input == nullptr || grid == nullptr) {
        throw std::invalid_argument("grid_sample requires input and grid");
    }
    if (input->type != GGML_TYPE_F32 || grid->type != GGML_TYPE_F32) {
        throw std::invalid_argument("grid_sample requires F32 input and grid");
    }
    // grid [2,Wout,Hout,N]: ne[0] holds the (x,y) pair, ne[1]/ne[2] the output
    // extent and ne[3] the batch that must match the input.
    if (grid->ne[0] != 2) {
        throw std::invalid_argument("grid_sample requires grid->ne[0] == 2 (x,y)");
    }
    if (grid->ne[3] != input->ne[3]) {
        throw std::invalid_argument("grid_sample batch mismatch between grid and input");
    }
    const int64_t W = input->ne[0];
    const int64_t H = input->ne[1];
    const int64_t C = input->ne[2];
    const int64_t N = input->ne[3];
    const int64_t Wout = grid->ne[1];
    const int64_t Hout = grid->ne[2];
    if (W <= 0 || H <= 0 || C <= 0 || N <= 0) {
        throw std::invalid_argument("grid_sample requires positive input extents");
    }
    if (Wout <= 0 || Hout <= 0) {
        throw std::invalid_argument("grid_sample requires positive output extents");
    }
    // Reject element counts that do not fit in int64 rather than overflowing
    // the tensor-size arithmetic (same guard style as load_gguf_tensors).
    int64_t input_count = 0;
    int64_t grid_count = 0;
    int64_t output_count = 0;
    if (!checked_mul(W, H, input_count) || !checked_mul(input_count, C, input_count) ||
        !checked_mul(input_count, N, input_count) || !checked_mul(Wout, Hout, output_count) ||
        !checked_mul(output_count, C, output_count) || !checked_mul(output_count, N, output_count) ||
        !checked_mul(Wout, Hout, grid_count) || !checked_mul(grid_count, N, grid_count) ||
        !checked_mul(grid_count, 2, grid_count)) {
        throw std::invalid_argument("grid_sample tensor sizes overflow int64");
    }

    struct ggml_tensor* args[2] = {input, grid};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, Wout, Hout, C, N, args, 2, grid_sample_custom, /*n_tasks=*/1, nullptr);
}

} // namespace ggml_ops
