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

// Bilinear Resize with half-pixel centers and clamp-to-edge (matches the
// RF-DETR-seg mask-branch Resize: linear mode, half_pixel, 36->108;
// verified against ORT: clamped matches to 6.6e-07, zeros does not).
// input [W,H,C,N] -> [oW,oH,C,N]; F32, contiguous.
void bilinear_resize_custom(struct ggml_tensor* dst, int ith, int nth, void* /*userdata*/) {
    const auto* src = dst->src[0];
    const int64_t W = src->ne[0];
    const int64_t H = src->ne[1];
    const int64_t C = src->ne[2];
    const int64_t N = src->ne[3];
    const int64_t oW = dst->ne[0];
    const int64_t oH = dst->ne[1];
    const double sx = static_cast<double>(W) / static_cast<double>(oW);
    const double sy = static_cast<double>(H) / static_cast<double>(oH);
    const int64_t rows = C * N;
    const int64_t dr = (rows + nth - 1) / nth;
    const int64_t row_begin = dr * ith;
    const int64_t row_end = std::min(row_begin + dr, rows);
    for (int64_t row = row_begin; row < row_end; ++row) {
        const int64_t c = row % C;
        const int64_t n = row / C;
        const float* in = reinterpret_cast<const float*>(reinterpret_cast<const char*>(src->data) +
                                                         (c * H + n * C * H) * W * sizeof(float));
        float* out =
            reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + (c * oH + n * C * oH) * oW * sizeof(float));
        for (int64_t y = 0; y < oH; ++y) {
            const double iy = (static_cast<double>(y) + 0.5) * sy - 0.5;
            const int64_t y0 = static_cast<int64_t>(std::floor(iy));
            const double fy = iy - static_cast<double>(y0);
            for (int64_t x = 0; x < oW; ++x) {
                const double ix = (static_cast<double>(x) + 0.5) * sx - 0.5;
                const int64_t x0 = static_cast<int64_t>(std::floor(ix));
                const double fx = ix - static_cast<double>(x0);
                double acc = 0.0;
                for (int ky = 0; ky < 2; ++ky) {
                    int64_t yy = y0 + ky;
                    yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
                    const double wy = ky == 0 ? 1.0 - fy : fy;
                    for (int kx = 0; kx < 2; ++kx) {
                        int64_t xx = x0 + kx;
                        xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
                        const double wx = kx == 0 ? 1.0 - fx : fx;
                        acc += static_cast<double>(in[yy * W + xx]) * wx * wy;
                    }
                }
                out[y * oW + x] = static_cast<float>(acc);
            }
        }
    }
}

// RF-DETR decoder sine position embedding (see header for the layout
// contract). One thread over queries is enough: 300x512 sin/cos pairs.
void rfdetr_sine_embed_custom(struct ggml_tensor* dst, int ith, int nth, void* /*userdata*/) {
    const auto* src = dst->src[0];
    constexpr int kCoords = 4;
    constexpr int kQuarters = 4;
    constexpr int kPairs = 64;
    constexpr int kOut = 512;
    constexpr int kOrder[4] = {1, 0, 2, 3};
    constexpr double kPi2 = 6.2831854820251465;
    const int64_t nq = src->ne[1];
    const float* in = reinterpret_cast<const float*>(src->data);
    float* out = reinterpret_cast<float*>(dst->data);
    const int64_t dq = (nq + nth - 1) / nth;
    const int64_t q_begin = dq * ith;
    const int64_t q_end = std::min(q_begin + dq, nq);
    for (int64_t q = q_begin; q < q_end; ++q) {
        for (int qp = 0; qp < kQuarters; ++qp) {
            const double c = static_cast<double>(in[q * kCoords + kOrder[qp]]);
            for (int i = 0; i < kPairs; ++i) {
                const double scale = std::pow(10000.0, i / 64.0);
                out[q * kOut + qp * 128 + 2 * i] = static_cast<float>(std::sin(c * kPi2 / scale));
                out[q * kOut + qp * 128 + 2 * i + 1] = static_cast<float>(std::cos(c * kPi2 / scale));
            }
        }
    }
}

struct ggml_tensor* rfdetr_sine_embed(struct ggml_context* ctx, struct ggml_tensor* coords) {
    if (coords == nullptr) {
        throw std::invalid_argument("rfdetr_sine_embed requires an input tensor");
    }
    if (coords->type != GGML_TYPE_F32 || !ggml_is_contiguous(coords)) {
        throw std::invalid_argument("rfdetr_sine_embed requires a contiguous F32 input");
    }
    if (coords->ne[0] != 4 || coords->ne[1] <= 0 || coords->ne[2] != 1 || coords->ne[3] != 1) {
        throw std::invalid_argument("rfdetr_sine_embed requires [4,Q,1,1] coords");
    }
    struct ggml_tensor* args[1] = {coords};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, 512, coords->ne[1], 1, 1, args, 1, rfdetr_sine_embed_custom,
                          /*n_tasks=*/1, nullptr);
}

struct ggml_tensor* bilinear_resize(struct ggml_context* ctx, struct ggml_tensor* x, int64_t out_w, int64_t out_h) {
    if (x == nullptr) {
        throw std::invalid_argument("bilinear_resize requires an input tensor");
    }
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) {
        throw std::invalid_argument("bilinear_resize requires a contiguous F32 input");
    }
    if (x->ne[0] <= 0 || x->ne[1] <= 0 || out_w <= 0 || out_h <= 0) {
        throw std::invalid_argument("bilinear_resize requires positive extents");
    }
    int64_t input_count = 0;
    int64_t output_count = 0;
    if (!checked_mul(x->ne[0], x->ne[1], input_count) || !checked_mul(input_count, x->ne[2], input_count) ||
        !checked_mul(input_count, x->ne[3], input_count) || !checked_mul(out_w, out_h, output_count) ||
        !checked_mul(output_count, x->ne[2], output_count) || !checked_mul(output_count, x->ne[3], output_count)) {
        throw std::invalid_argument("bilinear_resize tensor sizes overflow int64");
    }
    struct ggml_tensor* args[1] = {x};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, out_w, out_h, x->ne[2], x->ne[3], args, 1, bilinear_resize_custom,
                          /*n_tasks=*/1, nullptr);
}

// Depthwise 3x3 stride-1 pad-1 convolution (matches the RF-DETR-seg mask
// branch dwconv: groups=C, ONNX pads [1,1,1,1] zeros). ggml_conv_2d uses a
// fast algorithm with ~1e-3 absolute error at this signal magnitude, which the
// mask branch amplifies coherently through the 1x1 and Einsum 256-sums, so a
// direct exact kernel is used instead: F32 accumulation in fixed ky,kx order.
// input [W,H,C,N] -> [W,H,C,N]; weight [9,C,1,1] with k[c*9+ky*3+kx];
// bias [C]; all F32, contiguous.
void depthwise_conv3x3_custom(struct ggml_tensor* dst, int ith, int nth, void* /*userdata*/) {
    const auto* src = dst->src[0];
    const auto* wt = dst->src[1];
    const auto* bs = dst->src[2];
    const int64_t W = src->ne[0];
    const int64_t H = src->ne[1];
    const int64_t C = src->ne[2];
    const int64_t N = src->ne[3];
    const int64_t rows = C * N;
    const int64_t dr = (rows + nth - 1) / nth;
    const int64_t row_begin = dr * ith;
    const int64_t row_end = std::min(row_begin + dr, rows);
    for (int64_t row = row_begin; row < row_end; ++row) {
        const int64_t c = row % C;
        const int64_t n = row / C;
        const float* in = reinterpret_cast<const float*>(reinterpret_cast<const char*>(src->data) +
                                                         (c * H + n * C * H) * W * sizeof(float));
        const float* k =
            reinterpret_cast<const float*>(reinterpret_cast<const char*>(wt->data) + c * 9 * sizeof(float));
        const float b = reinterpret_cast<const float*>(bs->data)[c];
        float* out =
            reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + (c * H + n * C * H) * W * sizeof(float));
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                double acc = b;
                for (int64_t ky = 0; ky < 3; ++ky) {
                    const int64_t yy = y + ky - 1;
                    if (yy < 0 || yy >= H) {
                        continue;
                    }
                    for (int64_t kx = 0; kx < 3; ++kx) {
                        const int64_t xx = x + kx - 1;
                        if (xx < 0 || xx >= W) {
                            continue;
                        }
                        acc += static_cast<double>(in[yy * W + xx]) * static_cast<double>(k[ky * 3 + kx]);
                    }
                }
                out[y * W + x] = static_cast<float>(acc);
            }
        }
    }
}

struct ggml_tensor* depthwise_conv3x3(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* weight,
                                      struct ggml_tensor* bias) {
    if (x == nullptr || weight == nullptr || bias == nullptr) {
        throw std::invalid_argument("depthwise_conv3x3 requires input, weight and bias tensors");
    }
    if (x->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32) {
        throw std::invalid_argument("depthwise_conv3x3 requires F32 tensors");
    }
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(weight) || !ggml_is_contiguous(bias)) {
        throw std::invalid_argument("depthwise_conv3x3 requires contiguous tensors");
    }
    if (x->ne[0] <= 0 || x->ne[1] <= 0 || x->ne[2] <= 0 || x->ne[3] <= 0) {
        throw std::invalid_argument("depthwise_conv3x3 requires positive input extents");
    }
    if (weight->ne[0] != 9 || weight->ne[1] != x->ne[2] || weight->ne[2] != 1 || weight->ne[3] != 1) {
        throw std::invalid_argument("depthwise_conv3x3 requires weight [9,C,1,1] matching input channels");
    }
    if (bias->ne[0] != x->ne[2] || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
        throw std::invalid_argument("depthwise_conv3x3 requires bias [C,1,1,1]");
    }
    int64_t count = 0;
    if (!checked_mul(x->ne[0], x->ne[1], count) || !checked_mul(count, x->ne[2], count) ||
        !checked_mul(count, x->ne[3], count)) {
        throw std::invalid_argument("depthwise_conv3x3 tensor sizes overflow int64");
    }
    struct ggml_tensor* args[3] = {x, weight, bias};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3], args, 3, depthwise_conv3x3_custom,
                          /*n_tasks=*/1, nullptr);
}

} // namespace ggml_ops
