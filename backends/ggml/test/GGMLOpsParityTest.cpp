#include "GGMLOps.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
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
        if (dims.size() == 4) {
            auto it = dims.begin();
            return ggml_new_tensor_4d(ctx, type, *it, *(it + 1), *(it + 2), *(it + 3));
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

// Loads a raw little-endian F32 dump (the ORT reference format used by the
// RF-DETR GridSample probe). Returns false when the file is absent so the
// test can skip instead of failing on a machine without probe artifacts.
bool load_f32_dump(const std::string& path, std::vector<float>& values) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return false;
    }
    const auto size = static_cast<size_t>(in.tellg());
    if (size % sizeof(float) != 0) {
        return false;
    }
    values.resize(size / sizeof(float));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(size));
    return static_cast<size_t>(in.gcount()) == size;
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
    // ONNX TopK sorted=1: descending score, so 4 (index 1) then 3 (index 3).
    EXPECT_EQ(got, (std::vector<int32_t>{1, 3}));
}

TEST(GGMLOpsParity, TopKStableTieBreak) {
    GgmlFixture f;
    // Equal scores at indices 0 and 2 must keep ascending index order.
    struct ggml_tensor* scores = f.f32({4}, {5, 3, 5, 1});
    struct ggml_tensor* top = ggml_ops::top_k_indices(f.ctx, scores, 3);
    f.compute(top);
    EXPECT_EQ(read_i32(top), (std::vector<int32_t>{0, 2, 1}));
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

    struct ggml_tensor* grid4 = f.f32({2, 2, 2, 1}, std::vector<float>(8, 0.0f));
    struct ggml_tensor* bad_grid = f.f32({1, 2, 2, 1}, std::vector<float>(4, 0.0f));
    struct ggml_tensor* bad_batch = f.f32({2, 2, 2, 2}, std::vector<float>(16, 0.0f));
    struct ggml_tensor* i32_input = f.i32({2, 2, 2, 1}, std::vector<int32_t>(8, 0));
    EXPECT_THROW(ggml_ops::grid_sample(f.ctx, data, bad_grid), std::invalid_argument);
    EXPECT_THROW(ggml_ops::grid_sample(f.ctx, data, bad_batch), std::invalid_argument);
    EXPECT_THROW(ggml_ops::grid_sample(f.ctx, i32_input, grid4), std::invalid_argument);
    EXPECT_THROW(ggml_ops::softmax_rows(f.ctx, i32_input), std::invalid_argument);
}

// RF-DETR deformable-attention GridSample against the ORT reference dumps:
// input Reshape_3 (N=16,C=16,H=40,W=40) -> ggml [W,H,C,N] = [40,40,16,16],
// grid Reshape_4 (N=16,Hout=300,Wout=2,2) -> ggml [2,Wout,Hout,N] =
// [2,2,300,16], output (N=16,C=16,Hout=300,Wout=2) -> ggml [Wout,Hout,C,N] =
// [2,300,16,16]. The raw ONNX row-major dumps are already in ggml flat order
// because the axis reversal only renames, not reorders, the contiguous layout.
TEST(GGMLOpsParity, GridSampleRfdetr) {
    const std::string prefix = "/tmp/opencode/rfdetr_gs_";
    std::vector<std::string> missing;
    for (int layer = 0; layer < 3; ++layer) {
        for (const char* tag : {"reshape3", "reshape4", "gridsample"}) {
            const std::string path = prefix + "L" + std::to_string(layer) + "_" + tag + ".bin";
            std::ifstream probe(path, std::ios::binary);
            if (!probe) {
                missing.push_back(path);
            }
        }
    }
    if (!missing.empty()) {
        GTEST_SKIP() << "RF-DETR GridSample dumps not found (" << missing.front()
                     << "); regenerate with /tmp/opencode/rfdetr_gs_dump.py";
    }

    for (int layer = 0; layer < 3; ++layer) {
        const std::string base = prefix + "L" + std::to_string(layer) + "_";
        std::vector<float> input_data;
        std::vector<float> grid_data;
        std::vector<float> expected;
        ASSERT_TRUE(load_f32_dump(base + "reshape3.bin", input_data));
        ASSERT_TRUE(load_f32_dump(base + "reshape4.bin", grid_data));
        ASSERT_TRUE(load_f32_dump(base + "gridsample.bin", expected));
        ASSERT_EQ(input_data.size(), static_cast<size_t>(40 * 40 * 16 * 16));
        ASSERT_EQ(grid_data.size(), static_cast<size_t>(2 * 2 * 300 * 16));
        ASSERT_EQ(expected.size(), static_cast<size_t>(2 * 300 * 16 * 16));

        GgmlFixture f;
        struct ggml_tensor* input = f.f32({40, 40, 16, 16}, input_data);
        struct ggml_tensor* grid = f.f32({2, 2, 300, 16}, grid_data);
        struct ggml_tensor* out = ggml_ops::grid_sample(f.ctx, input, grid);
        ASSERT_NE(out, nullptr);
        EXPECT_EQ(out->ne[0], 2);
        EXPECT_EQ(out->ne[1], 300);
        EXPECT_EQ(out->ne[2], 16);
        EXPECT_EQ(out->ne[3], 16);
        f.compute(out);

        const std::vector<float> got = read_f32(out);
        ASSERT_EQ(got.size(), expected.size());
        double max_diff = 0.0;
        double sum_diff = 0.0;
        size_t mismatches = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(expected[i]));
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
            if (diff > 1e-5) {
                ++mismatches;
            }
        }
        const double mean_diff = sum_diff / static_cast<double>(got.size());
        // std::to_string would print six fixed decimals and round these values
        // to "0.000000"; record them with enough precision to see ULP drift.
        std::ostringstream diff;
        diff << std::scientific << std::setprecision(6);
        diff << max_diff;
        RecordProperty("L" + std::to_string(layer) + "_max_abs_diff", diff.str());
        diff.str("");
        diff << mean_diff;
        RecordProperty("L" + std::to_string(layer) + "_mean_abs_diff", diff.str());
        EXPECT_LE(max_diff, 1e-5) << "layer " << layer << " max_abs_diff " << max_diff << " mean_abs_diff " << mean_diff
                                  << " elements over tolerance " << mismatches;
    }
}

// Minimal .npy v1.0 loader for C-order little-endian F32 dumps (the ORT
// reference format under /tmp/opencode/dec0/). Returns false when the file is
// absent or not a plain C-order F32 array so the test can skip.
bool load_npy_f32(const std::string& path, std::vector<float>& values) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0 || magic[6] != 1 || magic[7] != 0) {
        return false;
    }
    uint16_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), 2);
    if (!in) {
        return false;
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    if (!in || header.find("'descr': '<f4'") == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos) {
        return false;
    }
    const auto payload_pos = in.tellg();
    in.seekg(0, std::ios::end);
    const auto end_pos = in.tellg();
    const size_t payload = static_cast<size_t>(end_pos - payload_pos);
    if (payload % sizeof(float) != 0) {
        return false;
    }
    in.seekg(payload_pos);
    values.resize(payload / sizeof(float));
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(payload));
    return static_cast<bool>(in);
}

struct SoftmaxParityResult {
    bool skipped = false;
    bool ok = true;
    std::string error;
    double max_diff = 0.0;
    double mean_diff = 0.0;
    double rel = 0.0;
    size_t got_argmax = 0;
    size_t expected_argmax = 0;
};

SoftmaxParityResult run_softmax_parity(const std::vector<int64_t>& ggml_dims, const std::string& input_path,
                                       const std::string& expected_path) {
    std::vector<float> input_data;
    std::vector<float> expected;
    SoftmaxParityResult result;
    if (!load_npy_f32(input_path, input_data) || !load_npy_f32(expected_path, expected)) {
        result.skipped = true;
        return result;
    }
    if (input_data.size() != expected.size()) {
        result.ok = false;
        result.error = "input/expected size mismatch";
        return result;
    }

    GgmlFixture f;
    struct ggml_tensor* input = f.f32({ggml_dims[0], ggml_dims[1], ggml_dims[2], ggml_dims[3]}, input_data);
    struct ggml_tensor* out = ggml_ops::softmax_rows(f.ctx, input);
    if (out == nullptr) {
        result.ok = false;
        result.error = "softmax_rows returned nullptr";
        return result;
    }
    f.compute(out);

    const std::vector<float> got = read_f32(out);
    if (got.size() != expected.size()) {
        result.ok = false;
        result.error = "output/expected size mismatch";
        return result;
    }
    double max_diff = 0.0;
    double sum_diff = 0.0;
    size_t got_argmax = 0;
    size_t expected_argmax = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(expected[i]));
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        if (got[i] > got[got_argmax]) {
            got_argmax = i;
        }
        if (expected[i] > expected[expected_argmax]) {
            expected_argmax = i;
        }
    }
    const double mean_diff = sum_diff / static_cast<double>(got.size());
    double max_abs_expected = 0.0;
    for (const float v : expected) {
        max_abs_expected = std::max(max_abs_expected, static_cast<double>(std::fabs(v)));
    }
    result.max_diff = max_diff;
    result.mean_diff = mean_diff;
    result.rel = max_abs_expected > 0.0 ? max_diff / max_abs_expected : max_diff;
    result.got_argmax = got_argmax;
    result.expected_argmax = expected_argmax;
    return result;
}

void check_softmax_parity(const std::string& tag, const SoftmaxParityResult& result) {
    ASSERT_TRUE(result.ok) << tag << " " << result.error;
    EXPECT_EQ(result.got_argmax, result.expected_argmax) << tag << " max-cell mismatch";
    EXPECT_TRUE(result.max_diff <= 1e-5 || result.rel <= 2e-5)
        << tag << " max_abs_diff " << result.max_diff << " rel " << result.rel << " mean_abs_diff " << result.mean_diff;
}

// RF-DETR decoder layer-0 Softmax rows vs ORT: the ONNX last axis maps to
// ggml ne[0], so raw row-major dumps are already in ggml flat order.
// Self-attn (8,300,300) -> ggml [300,300,8]. Cross-attn weights (1,300,32)
// are a pure view of Reshape_1 (1,300,16,2) (same bytes); the Softmax node
// reduces that last axis, so ggml dims are [2,16,300,1] and the flat bytes
// compare directly against expected (1,300,16,2).
TEST(GGMLOpsParity, SoftmaxRowsRfdetr) {
    const std::string base = "/tmp/opencode/dec0/";
    const SoftmaxParityResult self_attn =
        run_softmax_parity({300, 300, 8, 1}, base + "ref__layers.0__self_attn__MatMul_3_output_0.npy",
                           base + "ref__layers.0__self_attn__Softmax_output_0.npy");
    const SoftmaxParityResult cross_attn =
        run_softmax_parity({2, 16, 300, 1}, base + "ref__layers.0__cross_attn__attention_weights__Add_output_0.npy",
                           base + "ref__layers.0__cross_attn__Softmax_output_0.npy");
    if (self_attn.skipped || cross_attn.skipped) {
        GTEST_SKIP() << "softmax refs not found under " << base << "; regenerate layer-0 dec0 dumps";
    }
    check_softmax_parity("self_attn", self_attn);
    check_softmax_parity("cross_attn", cross_attn);
    for (const auto& entry : {std::pair<std::string, SoftmaxParityResult>{"self_attn", self_attn},
                              std::pair<std::string, SoftmaxParityResult>{"cross_attn", cross_attn}}) {
        std::ostringstream note;
        note << std::scientific << std::setprecision(6);
        note << entry.second.max_diff;
        RecordProperty(entry.first + "_max_abs_diff", note.str());
        note.str("");
        note << entry.second.rel;
        RecordProperty(entry.first + "_rel_diff", note.str());
    }
}
