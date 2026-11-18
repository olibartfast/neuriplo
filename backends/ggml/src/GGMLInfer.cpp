#include "GGMLInfer.hpp"

#include "GGMLOps.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Real-loader weight layout (GGUF, strict-GGUF-only):
//   GGUF v3, F32 tensors only, ggml-order dims, per-tensor
//   neuriplo.rank.<name> KV entries carrying the ONNX rank.
//   Written by scripts/convert_onnx_to_gguf.py (all float initializers,
//   names verbatim). No NGGL and no ONNX path at runtime.
// The topology is built from exact tensor names: the file named
// resnet18-v1-7.onnx carries resnetv15_* names (basic blocks, 512-d FC),
// the file named resnet50-v1-12.onnx carries resnetv17_* names (bottleneck
// blocks, 2048-d FC). Stem (shared): conv0/bn0. ResNet-18 stages: stage1
// conv0..3/bn0..3, stages 2..4 conv0..4/bn0..4 (conv2/bn2 is the 1x1
// stride-2 downsample of the first block). ResNet-50 stages [3,4,6,3]
// bottleneck blocks: block b uses convs (3b,3b+1,3b+2) for b==0 else
// (3b+1,3b+2,3b+3); conv3/bn3 is the 1x1 downsample of the stage entry
// (stride 1 on stage 1, 2 below). FC <prefix>dense0_{weight,bias}.
// Graph for the fixed Nx3x224x224 F32 -> 1000 path: input NCHW blob
// [N,3,224,224] (row-major, which coincides element-for-element with ggml
// [W,H,C,N]) -> conv7x7/s2/p3 64ch -> folded bias add -> relu ->
// maxpool k3/s2/p1 -> stage blocks (basic: conv3x3/s/p1 -> bias -> relu ->
// conv3x3/s1/p1 -> bias -> skip add -> relu; bottleneck: 1x1 reduce/s ->
// bias -> relu -> 3x3/s1/p1 -> bias -> relu -> 1x1 expand -> bias ->
// skip add -> relu; entry skips are 1x1/s/p0 -> bias, no relu)
// -> global avgpool k7 -> reshape K -> FC[1000,K] verbatim orientation +
// bias -> logits [1000,N] (K=512 ResNet-18, K=2048 ResNet-50). BN is
// folded exactly at load (D-7); no BN op in-graph.
namespace {

constexpr size_t kInputChannels = 3;

// RF-DETR backbone per-block MatMul weight names (DINOv2-S, 11 blocks),
// transcribed from rfdetr.onnx (see /tmp/opencode/rfdetr_tables.inc).
// File dims are ONNX row-major [K,N]; the loader transposes the bytes into
// ggml ne=[K,N] (same coincidence the probe uses).
constexpr const char* kRfdetrWQ[11] = {"onnx::MatMul_3712", "onnx::MatMul_3737", "onnx::MatMul_3766",
                                       "onnx::MatMul_3795", "onnx::MatMul_3820", "onnx::MatMul_3849",
                                       "onnx::MatMul_3878", "onnx::MatMul_3903", "onnx::MatMul_3932",
                                       "onnx::MatMul_3961", "onnx::MatMul_3986"};
constexpr const char* kRfdetrWK[11] = {"onnx::MatMul_3713", "onnx::MatMul_3738", "onnx::MatMul_3767",
                                       "onnx::MatMul_3796", "onnx::MatMul_3821", "onnx::MatMul_3850",
                                       "onnx::MatMul_3879", "onnx::MatMul_3904", "onnx::MatMul_3933",
                                       "onnx::MatMul_3962", "onnx::MatMul_3987"};
constexpr const char* kRfdetrWV[11] = {"onnx::MatMul_3719", "onnx::MatMul_3744", "onnx::MatMul_3773",
                                       "onnx::MatMul_3802", "onnx::MatMul_3827", "onnx::MatMul_3856",
                                       "onnx::MatMul_3885", "onnx::MatMul_3910", "onnx::MatMul_3939",
                                       "onnx::MatMul_3968", "onnx::MatMul_3993"};
constexpr const char* kRfdetrWO[11] = {"onnx::MatMul_3734", "onnx::MatMul_3759", "onnx::MatMul_3788",
                                       "onnx::MatMul_3817", "onnx::MatMul_3842", "onnx::MatMul_3871",
                                       "onnx::MatMul_3900", "onnx::MatMul_3925", "onnx::MatMul_3954",
                                       "onnx::MatMul_3983", "onnx::MatMul_4008"};
constexpr const char* kRfdetrWF1[11] = {"onnx::MatMul_3735", "onnx::MatMul_3760", "onnx::MatMul_3793",
                                        "onnx::MatMul_3818", "onnx::MatMul_3843", "onnx::MatMul_3876",
                                        "onnx::MatMul_3901", "onnx::MatMul_3926", "onnx::MatMul_3959",
                                        "onnx::MatMul_3984", "onnx::MatMul_4009"};
constexpr const char* kRfdetrWF2[11] = {"onnx::MatMul_3736", "onnx::MatMul_3761", "onnx::MatMul_3794",
                                        "onnx::MatMul_3819", "onnx::MatMul_3844", "onnx::MatMul_3877",
                                        "onnx::MatMul_3902", "onnx::MatMul_3927", "onnx::MatMul_3960",
                                        "onnx::MatMul_3985", "onnx::MatMul_4010"};
// RF-DETR-seg backbone per-block MatMul weight names (DINOv2-S, 12 blocks,
// global attention in blocks 3/6/9), transcribed from rfdetr_seg.onnx.
constexpr const char* kSegWQ[12] = {"onnx::MatMul_4276", "onnx::MatMul_4301", "onnx::MatMul_4326", "onnx::MatMul_4355",
                                    "onnx::MatMul_4384", "onnx::MatMul_4409", "onnx::MatMul_4438", "onnx::MatMul_4467",
                                    "onnx::MatMul_4492", "onnx::MatMul_4521", "onnx::MatMul_4550", "onnx::MatMul_4575"};
constexpr const char* kSegWK[12] = {"onnx::MatMul_4277", "onnx::MatMul_4302", "onnx::MatMul_4327", "onnx::MatMul_4356",
                                    "onnx::MatMul_4385", "onnx::MatMul_4410", "onnx::MatMul_4439", "onnx::MatMul_4468",
                                    "onnx::MatMul_4493", "onnx::MatMul_4522", "onnx::MatMul_4551", "onnx::MatMul_4576"};
constexpr const char* kSegWV[12] = {"onnx::MatMul_4283", "onnx::MatMul_4308", "onnx::MatMul_4333", "onnx::MatMul_4362",
                                    "onnx::MatMul_4391", "onnx::MatMul_4416", "onnx::MatMul_4445", "onnx::MatMul_4474",
                                    "onnx::MatMul_4499", "onnx::MatMul_4528", "onnx::MatMul_4557", "onnx::MatMul_4582"};
constexpr const char* kSegWO[12] = {"onnx::MatMul_4298", "onnx::MatMul_4323", "onnx::MatMul_4348", "onnx::MatMul_4377",
                                    "onnx::MatMul_4406", "onnx::MatMul_4431", "onnx::MatMul_4460", "onnx::MatMul_4489",
                                    "onnx::MatMul_4514", "onnx::MatMul_4543", "onnx::MatMul_4572", "onnx::MatMul_4597"};
constexpr const char* kSegWF1[12] = {"onnx::MatMul_4299", "onnx::MatMul_4324", "onnx::MatMul_4349",
                                     "onnx::MatMul_4382", "onnx::MatMul_4407", "onnx::MatMul_4432",
                                     "onnx::MatMul_4465", "onnx::MatMul_4490", "onnx::MatMul_4515",
                                     "onnx::MatMul_4548", "onnx::MatMul_4573", "onnx::MatMul_4598"};
constexpr const char* kSegWF2[12] = {"onnx::MatMul_4300", "onnx::MatMul_4325", "onnx::MatMul_4350",
                                     "onnx::MatMul_4383", "onnx::MatMul_4408", "onnx::MatMul_4433",
                                     "onnx::MatMul_4466", "onnx::MatMul_4491", "onnx::MatMul_4516",
                                     "onnx::MatMul_4549", "onnx::MatMul_4574", "onnx::MatMul_4599"};
// RF-DETR-seg decoder per-layer weights (4 layers). Self q/k/v are anonymous
// MatMuls with anonymous Add biases; out_proj is a Gemm with a named
// weight/bias pair (same Y=X@W orientation as the MatMul helper expects).
struct RfdetrDecLayer {
    const char* q;
    const char* qb;
    const char* k;
    const char* kb;
    const char* v;
    const char* vb;
    const char* out;
    const char* vv;
    const char* so;
    const char* aw;
    const char* oo;
    const char* f1;
    const char* f2;
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): fixed 3-row table
constexpr RfdetrDecLayer kDetDec[3] = {
    {"onnx::MatMul_4272", "onnx::Add_4267", "onnx::MatMul_4273", "onnx::Add_4269", "onnx::MatMul_4274",
     "onnx::Add_4271", "onnx::MatMul_4285", "onnx::MatMul_4286", "onnx::MatMul_4287", "onnx::MatMul_4293",
     "onnx::MatMul_4323", "onnx::MatMul_4324", "onnx::MatMul_4325"},
    {"onnx::MatMul_4346", "onnx::Add_4341", "onnx::MatMul_4347", "onnx::Add_4343", "onnx::MatMul_4348",
     "onnx::Add_4345", "onnx::MatMul_4359", "onnx::MatMul_4360", "onnx::MatMul_4361", "onnx::MatMul_4367",
     "onnx::MatMul_4374", "onnx::MatMul_4375", "onnx::MatMul_4376"},
    {"onnx::MatMul_4397", "onnx::Add_4392", "onnx::MatMul_4398", "onnx::Add_4394", "onnx::MatMul_4399",
     "onnx::Add_4396", "onnx::MatMul_4410", "onnx::MatMul_4411", "onnx::MatMul_4412", "onnx::MatMul_4418",
     "onnx::MatMul_4425", "onnx::MatMul_4426", "onnx::MatMul_4427"},
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): fixed 4-row table
constexpr RfdetrDecLayer kSegDec[4] = {
    {"onnx::MatMul_4827", "onnx::Add_4822", "onnx::MatMul_4828", "onnx::Add_4824", "onnx::MatMul_4829",
     "onnx::Add_4826", "transformer.decoder.layers.0.self_attn.out_proj.weight", "onnx::MatMul_4834",
     "onnx::MatMul_4835", "onnx::MatMul_4841", "onnx::MatMul_4876", "onnx::MatMul_4877", "onnx::MatMul_4878"},
    {"onnx::MatMul_4899", "onnx::Add_4894", "onnx::MatMul_4900", "onnx::Add_4896", "onnx::MatMul_4901",
     "onnx::Add_4898", "transformer.decoder.layers.1.self_attn.out_proj.weight", "onnx::MatMul_4905",
     "onnx::MatMul_4906", "onnx::MatMul_4912", "onnx::MatMul_4919", "onnx::MatMul_4920", "onnx::MatMul_4921"},
    {"onnx::MatMul_4942", "onnx::Add_4937", "onnx::MatMul_4943", "onnx::Add_4939", "onnx::MatMul_4944",
     "onnx::Add_4941", "transformer.decoder.layers.2.self_attn.out_proj.weight", "onnx::MatMul_4948",
     "onnx::MatMul_4949", "onnx::MatMul_4955", "onnx::MatMul_4962", "onnx::MatMul_4963", "onnx::MatMul_4964"},
    {"onnx::MatMul_4985", "onnx::Add_4980", "onnx::MatMul_4986", "onnx::Add_4982", "onnx::MatMul_4987",
     "onnx::Add_4984", "transformer.decoder.layers.3.self_attn.out_proj.weight", "onnx::MatMul_4991",
     "onnx::MatMul_4992", "onnx::MatMul_4998", "onnx::MatMul_5005", "onnx::MatMul_5006", "onnx::MatMul_5007"},
};
constexpr size_t kInputHeight = 224;
constexpr size_t kInputWidth = 224;
constexpr int64_t kInputElements = 150528; // 3*224*224
constexpr int64_t kNumClasses = 1000;
// D-7: exact frozen-inference BN fold; epsilon verified uniform (1e-5) in
// resnet18-v1-7.onnx (all 20 BatchNormalization nodes) and in
// resnet50-v1-12.onnx (all 53 nodes; file value 9.9999997e-06, the float32
// spelling of 1e-5).
constexpr double kBatchNormEps = 1e-5;

struct LoadedTensor {
    std::vector<uint32_t> dims;
    std::vector<float> data;
};

[[noreturn]] void fail_model(const std::string& model_path, const std::string& reason) {
    throw std::runtime_error("Invalid GGML model file '" + model_path + "': " + reason);
}

// Folded conv constant pair: weight in ggml conv layout [KW,KH,IC,OC],
// bias shaped [1,1,OC,1] so ggml_repeat broadcasts it per-channel over a
// [W,H,OC,N] activation (verified: repeat indexes per-dimension modulo).
struct FoldedConv {
    struct ggml_tensor* weight = nullptr;
    struct ggml_tensor* bias = nullptr;
};

std::string describe_shapes(const std::map<std::string, LoadedTensor>& tensors) {
    std::string seen;
    seen.reserve(tensors.size() * 48);
    for (const auto& kv : tensors) {
        seen += " '" + kv.first + "'[";
        for (size_t i = 0; i < kv.second.dims.size(); ++i) {
            if (i != 0) {
                seen += "x";
            }
            seen += std::to_string(kv.second.dims[i]);
        }
        seen += "]";
    }
    return seen;
}

const LoadedTensor& require_tensor(const std::map<std::string, LoadedTensor>& tensors, const std::string& name,
                                   const std::string& model_path, const std::string& seen) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        fail_model(model_path, "missing required tensor '" + name + "'; tensors:" + seen);
    }
    return it->second;
}

void require_dims(const LoadedTensor& t, const std::string& name, const std::string& model_path,
                  const std::vector<uint32_t>& expected, const std::string& seen = "") {
    if (t.dims != expected) {
        auto fmt = [](const std::vector<uint32_t>& d) {
            std::string s;
            for (size_t k = 0; k < d.size(); ++k) {
                if (k != 0) {
                    s += "x";
                }
                s += std::to_string(d[k]);
            }
            return s;
        };
        fail_model(model_path, "tensor '" + name + "' has dims [" + fmt(t.dims) + "], expected [" + fmt(expected) +
                                   "]" + (seen.empty() ? "" : "; tensors:" + seen));
    }
}

// Fold BatchNormalization into the preceding conv (D-7: frozen inference,
// per-channel s = gamma/sqrt(var+eps), beta = param_b - mean*s) and permute
// the kernel ONNX row-major [O,I,H,W] -> ggml [KW,KH,IC,OC], i.e. file
// element ((o*I+i)*H+ky)*W+kx lands at ggml (kx,ky,i,o) (D-8). A conv own
// bias (present on the resnet50-v1-12 1x1 convs, absent on resnet18) is
// subsumed exactly: y = s*(conv(x) + cb - mean) + beta, so the folded bias
// is beta[o] + s[o]*(cb[o] - mean[o]); a missing bias counts as zeros.
std::pair<std::vector<float>, std::vector<float>>
fold_bn_and_permute_kernel(const std::map<std::string, LoadedTensor>& tensors, const std::string& conv_name,
                           const std::string& bn_prefix, const std::string& model_path, const std::string& seen) {
    const LoadedTensor& w = require_tensor(tensors, conv_name, model_path, seen);
    if (w.dims.size() != 4) {
        fail_model(model_path,
                   "tensor '" + conv_name + "' is not a 4D conv kernel (rank " + std::to_string(w.dims.size()) + ")");
    }
    const uint32_t O = w.dims[0];
    const uint32_t I = w.dims[1];
    const uint32_t H = w.dims[2];
    const uint32_t W = w.dims[3];

    const std::string kWeightSuffix = "_weight";
    if (conv_name.size() <= kWeightSuffix.size() ||
        conv_name.compare(conv_name.size() - kWeightSuffix.size(), kWeightSuffix.size(), kWeightSuffix) != 0) {
        fail_model(model_path, "tensor '" + conv_name + "' does not end in '_weight'");
    }
    const std::string own_bias = conv_name.substr(0, conv_name.size() - kWeightSuffix.size()) + "_bias";
    std::vector<float> conv_bias;
    const auto own_it = tensors.find(own_bias);
    if (own_it != tensors.end()) {
        if (own_it->second.dims.size() != 1 || own_it->second.dims[0] != O) {
            fail_model(model_path, "tensor '" + own_bias + "' has wrong shape (need 1D [" + std::to_string(O) +
                                       "]) for BN fold of '" + conv_name + "'");
        }
        conv_bias = own_it->second.data;
    }

    const LoadedTensor& gamma = require_tensor(tensors, bn_prefix + "gamma", model_path, seen);
    const LoadedTensor& beta = require_tensor(tensors, bn_prefix + "beta", model_path, seen);
    const LoadedTensor& mean = require_tensor(tensors, bn_prefix + "running_mean", model_path, seen);
    const LoadedTensor& var = require_tensor(tensors, bn_prefix + "running_var", model_path, seen);
    for (const auto* bn : {&gamma, &beta, &mean, &var}) {
        if (bn->dims.size() != 1 || bn->dims[0] != O) {
            fail_model(model_path,
                       "BN param for '" + conv_name + "' has wrong shape (need 1D [" + std::to_string(O) + "])");
        }
    }

    std::vector<float> permuted(static_cast<size_t>(O) * I * H * W);
    std::vector<float> folded_bias(O);
    for (uint32_t o = 0; o < O; ++o) {
        const double s =
            static_cast<double>(gamma.data[o]) / std::sqrt(static_cast<double>(var.data[o]) + kBatchNormEps);
        const double cb = conv_bias.empty() ? 0.0 : static_cast<double>(conv_bias[o]);
        folded_bias[o] =
            static_cast<float>(static_cast<double>(beta.data[o]) + s * (cb - static_cast<double>(mean.data[o])));
        for (uint32_t i = 0; i < I; ++i) {
            for (uint32_t ky = 0; ky < H; ++ky) {
                for (uint32_t kx = 0; kx < W; ++kx) {
                    const float wv = w.data[(static_cast<size_t>(o) * I + i) * H * W + ky * W + kx];
                    permuted[(static_cast<size_t>(o) * I + i) * H * W + ky * W + kx] =
                        static_cast<float>(static_cast<double>(wv) * s);
                }
            }
        }
    }
    return {std::move(permuted), std::move(folded_bias)};
}

// conv/s/p (dilation 1) + folded-bias add via ggml_repeat to full shape +
// ggml_add (no broadcast-add assumption) [+ relu].
struct ggml_tensor* conv_bias_add(struct ggml_context* ctx, struct ggml_tensor* x, const FoldedConv& c, int stride,
                                  int pad, bool relu) {
    struct ggml_tensor* h = ggml_conv_2d(ctx, c.weight, x, stride, stride, pad, pad, 1, 1);
    h = ggml_add(ctx, h, ggml_repeat(ctx, c.bias, h));
    if (relu) {
        h = ggml_relu(ctx, h);
    }
    return h;
}

// Basic residual block: conv3x3/s/p1 -> bias -> relu -> conv3x3/s1/p1 ->
// bias -> skip add -> relu. Downsample path (first block of layers 2..4) is
// conv1x1/s/p0 -> bias with no relu.
struct ggml_tensor* basic_block(struct ggml_context* ctx, struct ggml_tensor* x, const FoldedConv& c1,
                                const FoldedConv& c2, const FoldedConv* downsample, int stride) {
    struct ggml_tensor* h = conv_bias_add(ctx, x, c1, stride, 1, true);
    h = conv_bias_add(ctx, h, c2, 1, 1, false);
    struct ggml_tensor* skip = (downsample != nullptr) ? conv_bias_add(ctx, x, *downsample, stride, 0, false) : x;
    return ggml_relu(ctx, ggml_add(ctx, h, skip));
}

// Bottleneck residual block (ResNet-50): 1x1 reduce (carries the stride on
// stage entries) -> bias -> relu -> 3x3/s1/p1 -> bias -> relu -> 1x1 expand
// -> bias -> skip add -> relu. The downsample path (every stage entry:
// channel and/or spatial change) is 1x1/s/p0 -> bias with no relu.
struct ggml_tensor* bottleneck_block(struct ggml_context* ctx, struct ggml_tensor* x, const FoldedConv& reduce,
                                     const FoldedConv& mid, const FoldedConv& expand, const FoldedConv* downsample,
                                     int stride) {
    struct ggml_tensor* h = conv_bias_add(ctx, x, reduce, stride, 0, true);
    h = conv_bias_add(ctx, h, mid, 1, 1, true);
    h = conv_bias_add(ctx, h, expand, 1, 0, false);
    struct ggml_tensor* skip = (downsample != nullptr) ? conv_bias_add(ctx, x, *downsample, stride, 0, false) : x;
    return ggml_relu(ctx, ggml_add(ctx, h, skip));
}

// ---- YOLOv8 helpers (fixed 640 geometry; NCHW stored ggml-order) ----

// Split a channel dim in half. Input must be contiguous (conv/silu outputs
// are); the two views share storage, no copy.
inline std::pair<struct ggml_tensor*, struct ggml_tensor*> split_half(struct ggml_context* ctx, struct ggml_tensor* x) {
    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];
    const int64_t C = x->ne[2];
    const int64_t N = x->ne[3];
    // nb strides of a contiguous tensor, in bytes.
    const size_t nb1 = static_cast<size_t>(W) * sizeof(float);
    const size_t nb2 = nb1 * static_cast<size_t>(H);
    const size_t nb3 = nb2 * static_cast<size_t>(C);
    struct ggml_tensor* a = ggml_view_4d(ctx, x, W, H, C / 2, N, nb1, nb2, nb3, 0);
    struct ggml_tensor* b = ggml_view_4d(ctx, x, W, H, C / 2, N, nb1, nb2, nb3, (nb3 / 2));
    return {a, b};
}

// Channel concat of 2..4 inputs (ggml dim 2 == NCHW dim 1).
inline struct ggml_tensor* concat_ch(struct ggml_context* ctx, const std::vector<struct ggml_tensor*>& xs) {
    struct ggml_tensor* out = xs[0];
    for (size_t i = 1; i < xs.size(); ++i) {
        out = ggml_concat(ctx, out, xs[i], 2);
    }
    return ggml_cont(ctx, out);
}

void load_gguf_tensors(const std::string& model_path, std::map<std::string, LoadedTensor>& tensors) {
    // GGUF weights container written by scripts/convert_onnx_to_gguf.py:
    // F32 tensors only, ggml-order dims, per-tensor neuriplo.rank.<name>
    // KV entries carrying the ONNX rank. Parsed directly (not via the gguf
    // C API) because tensor names can exceed that API's 64-char limit
    // (e.g. RF-DETR's nested encoder names).
    std::ifstream blob(model_path, std::ios::binary);
    if (!blob) {
        throw std::runtime_error("Cannot open model file: " + model_path);
    }
    blob.seekg(0, std::ios::end);
    const auto file_size = static_cast<size_t>(blob.tellg());
    blob.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(file_size);
    blob.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(file_size));
    if (static_cast<size_t>(blob.gcount()) != file_size) {
        fail_model(model_path, "cannot read GGUF file");
    }
    const auto u32_at = [&](size_t off, uint32_t& out) {
        if (off > bytes.size() || bytes.size() - off < 4) {
            fail_model(model_path, "truncated GGUF header");
        }
        std::memcpy(&out, bytes.data() + off, 4);
    };
    const auto u64_at = [&](size_t off, uint64_t& out) {
        if (off > bytes.size() || bytes.size() - off < 8) {
            fail_model(model_path, "truncated GGUF header");
        }
        std::memcpy(&out, bytes.data() + off, 8);
    };
    const auto str_at = [&](size_t& off) {
        uint64_t len = 0;
        u64_at(off, len);
        off += 8;
        if (len > (1 << 20) || off > bytes.size() || len > bytes.size() - off) {
            fail_model(model_path, "truncated GGUF string");
        }
        std::string s(reinterpret_cast<const char*>(bytes.data() + off), static_cast<size_t>(len));
        off += static_cast<size_t>(len);
        return s;
    };
    if (bytes.size() < 24 || std::memcmp(bytes.data(), "GGUF", 4) != 0) {
        fail_model(model_path, "bad GGUF magic");
    }
    uint32_t version = 0;
    u32_at(4, version);
    if (version != 3) {
        fail_model(model_path, "unsupported GGUF version " + std::to_string(version) + ", expected 3");
    }
    uint64_t n_tensors = 0, n_kv = 0;
    u64_at(8, n_tensors);
    u64_at(16, n_kv);
    if (n_tensors == 0 || n_tensors > 100000) {
        fail_model(model_path, "implausible GGUF tensor count " + std::to_string(n_tensors));
    }
    // Each KV consumes at least a 12-byte key length/type prefix; reject
    // counts that cannot fit before entering the metadata loop.
    if (n_kv > bytes.size() / 12) {
        fail_model(model_path, "implausible GGUF KV count " + std::to_string(n_kv));
    }
    size_t off = 24;
    std::map<std::string, int32_t> ranks;
    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = str_at(off);
        uint32_t vtype = 0;
        u32_at(off, vtype);
        off += 4;
        if (vtype == 8) { // string
            (void)str_at(off);
        } else if (vtype == 5) { // int32
            uint32_t raw = 0;
            u32_at(off, raw);
            off += 4;
            constexpr char kRankPrefix[] = "neuriplo.rank.";
            if (key.compare(0, sizeof(kRankPrefix) - 1, kRankPrefix) == 0) {
                int32_t rank = 0;
                std::memcpy(&rank, &raw, 4);
                const std::string rank_name = key.substr(sizeof(kRankPrefix) - 1);
                if (rank_name.empty()) {
                    fail_model(model_path, "empty tensor name in rank entry");
                }
                if (!ranks.emplace(rank_name, rank).second) {
                    fail_model(model_path, "duplicate rank entry for tensor '" + rank_name + "'");
                }
            }
        } else {
            fail_model(model_path, "unsupported GGUF KV type for key '" + key + "'");
        }
    }
    struct Info {
        std::string name;
        int32_t rank = 0;
        std::vector<uint64_t> ne;
        uint64_t offset = 0;
    };
    std::vector<Info> infos;
    infos.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Info in;
        in.name = str_at(off);
        if (in.name.empty()) {
            fail_model(model_path, "GGUF tensor has no name");
        }
        uint32_t nd = 0, ttype = 0;
        u32_at(off, nd);
        off += 4;
        if (nd < 1 || nd > 4) {
            fail_model(model_path, "tensor '" + in.name + "' has invalid ndims");
        }
        in.ne.resize(nd);
        for (uint32_t d = 0; d < nd; ++d) {
            u64_at(off, in.ne[d]);
            off += 8;
            // Zero-size initializers are legal in ONNX and are carried by the
            // converter as empty tensors; the topology ignores them.
            if (in.ne[d] > (1 << 24)) {
                fail_model(model_path, "tensor '" + in.name + "' has invalid dim");
            }
        }
        u32_at(off, ttype);
        off += 4;
        if (ttype != static_cast<uint32_t>(GGML_TYPE_F32)) {
            fail_model(model_path, "tensor '" + in.name + "' is not F32");
        }
        uint64_t toffset = 0;
        u64_at(off, toffset);
        off += 8;
        const auto rank_it = ranks.find(in.name);
        if (rank_it == ranks.end()) {
            fail_model(model_path, "tensor '" + in.name + "' has no 'neuriplo.rank." + in.name + "' entry");
        }
        in.rank = rank_it->second;
        if (in.rank < 1 || in.rank > 4 || static_cast<uint32_t>(in.rank) != nd) {
            fail_model(model_path, "tensor '" + in.name + "' has invalid rank " + std::to_string(in.rank));
        }
        in.offset = toffset;
        infos.push_back(std::move(in));
    }
    if (off > std::numeric_limits<size_t>::max() - 31) {
        fail_model(model_path, "GGUF metadata offset overflow");
    }
    const size_t data_base = (off + 31) / 32 * 32;
    if (data_base > bytes.size()) {
        fail_model(model_path, "tensor data section starts past end of file");
    }
    // Every tensor consumes at least one float per element; a larger element
    // count cannot be backed by this file and must not drive an allocation.
    const size_t max_elements = bytes.size() / sizeof(float);
    for (const auto& in : infos) {
        // ne[] is ggml order; dims[] is ONNX order (reversed).
        std::vector<uint32_t> dims;
        dims.reserve(static_cast<size_t>(in.rank));
        uint64_t num_elements = 1;
        for (int32_t d = in.rank - 1; d >= 0; --d) {
            dims.push_back(static_cast<uint32_t>(in.ne[d]));
            if (in.ne[d] == 0) {
                num_elements = 0;
            } else if (num_elements != 0) {
                if (num_elements > static_cast<uint64_t>(max_elements) / in.ne[d]) {
                    fail_model(model_path, "tensor '" + in.name + "' element count too large");
                }
                num_elements *= in.ne[d];
            }
        }
        if (num_elements > max_elements) {
            fail_model(model_path, "tensor '" + in.name + "' element count too large");
        }
        const size_t element_count = static_cast<size_t>(num_elements);
        const size_t byte_size = element_count * sizeof(float);
        if (in.offset > bytes.size() - data_base) {
            fail_model(model_path, "tensor '" + in.name + "' extends past end of file");
        }
        const size_t start = data_base + static_cast<size_t>(in.offset);
        if (byte_size > bytes.size() - start) {
            fail_model(model_path, "tensor '" + in.name + "' extends past end of file");
        }
        std::vector<float> data(element_count);
        if (byte_size != 0) {
            std::memcpy(data.data(), bytes.data() + start, byte_size);
        }
        if (tensors.count(in.name) != 0) {
            fail_model(model_path, "duplicate tensor '" + in.name + "'");
        }
        tensors.emplace(in.name, LoadedTensor{std::move(dims), std::move(data)});
    }
}

} // namespace

GGMLInfer::GGMLInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                     const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}, ctx_(nullptr), backend_(nullptr),
      buffer_(nullptr), graph_(nullptr), feat_tensor_(nullptr), model_loaded_(false) {
    try {
        LOG(INFO) << "Running using GGML runtime: " << model_path;

        // Initialize GGML
        ggml_init_params params = {.mem_size = 1024 * 1024 * 1024, // 1GB
                                   .mem_buffer = nullptr,
                                   .no_alloc = true};

        ctx_ = ggml_init(params);
        if (!ctx_) {
            throw std::runtime_error("Failed to initialize GGML context");
        }

        // Setup backend (CPU or GPU)
        setup_backend(use_gpu);

        // Load model
        load_model(model_path);

        // Setup input/output tensors
        setup_input_output_tensors(input_sizes);

        model_loaded_ = true;

        state_ = BackendState::Ready;

    } catch (const std::exception& e) {
        if (buffer_) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        if (backend_) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        state_ = BackendState::Failed;
        throw;
    }
}

GGMLInfer::~GGMLInfer() {
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
    }
    if (backend_) {
        ggml_backend_free(backend_);
    }
    if (ctx_) {
        ggml_free(ctx_);
    }
}

void GGMLInfer::setup_backend(bool use_gpu) {
    if (use_gpu) {
        // CPU-only build: fail fast instead of silently running on CPU.
        throw std::runtime_error("GGML backend requested use_gpu=true, but this build is CPU-only; "
                                 "retry with use_gpu=false");
    }
    backend_ = ggml_backend_cpu_init();
    LOG(INFO) << "Using CPU backend";

    if (!backend_) {
        throw std::runtime_error("Failed to initialize GGML backend");
    }
}

void GGMLInfer::load_model(const std::string& model_path) {
    // Check if file exists
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_path);
    }
    std::string extension = model_path.size() >= 5 ? model_path.substr(model_path.size() - 5) : std::string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    if (extension != ".gguf") {
        fail_model(model_path, "only .gguf weights are supported; convert first: "
                               "python3 scripts/convert_onnx_to_gguf.py --input <model.onnx> --output <model.gguf>");
    }

    LOG(INFO) << "Loading GGML model from: " << model_path;

    std::map<std::string, LoadedTensor> tensors;
    load_gguf_tensors(model_path, tensors);

    // --- ResNet topology from exact tensor names (D-6) ---
    // Dispatch ResNet-18 vs ResNet-50 by FC input dim (512 vs 2048) plus
    // bottleneck tensor presence (enforced by the stage requires below):
    // exact per-file classifier names first (D-1), dims-only structural
    // fallback where names genuinely vary. Anything unrecognized fails loud
    // with the path and the shapes seen (R-5).
    const std::string shapes_seen = describe_shapes(tensors);

    // YOLOv8n family: dot-separated ultralytics tensor names anchored at the
    // stem conv. Detection returns [1,84,8400]; segmentation adds the cv4
    // mask-coefficient head and the proto branch, returning
    // [1,116,8400] and [1,32,160,160].
    const bool has_seg_proto = tensors.count("model.22.proto.cv1.conv.weight") != 0;
    const bool has_seg_cv4 = tensors.count("model.22.cv4.0.0.conv.weight") != 0;
    // YOLO11 family: DFL moved model.22 -> model.23, new C2PSA attention at
    // model.10 (qkv/PE/proj/FFN). Detect before the v8 stem check below.
    // ONNX tensor names: model.10.m.0.attn.qkv.conv.weight (single m.0)
    const bool is_yolo11 =
        tensors.count("model.23.dfl.conv.weight") != 0 && tensors.count("model.10.m.0.attn.qkv.conv.weight") != 0;
    is_yolo11_ = is_yolo11;
    // YOLO11-seg detection
    const bool has_yolo11_seg_proto = tensors.count("model.23.proto.cv1.conv.weight") != 0;
    const bool has_yolo11_seg_cv4 = tensors.count("model.23.cv4.0.0.conv.weight") != 0;
    const bool is_yolo11_seg = has_yolo11_seg_proto && has_yolo11_seg_cv4;
    if (has_yolo11_seg_proto != has_yolo11_seg_cv4) {
        fail_model(model_path, "incomplete YOLO11-seg head (proto and cv4 tensors must both be present)");
    }
    if (has_seg_proto != has_seg_cv4) {
        fail_model(model_path, "incomplete YOLOv8n-seg head (proto and cv4 tensors must both be present)");
    }
    const bool is_yolo = tensors.count("model.0.conv.weight") != 0;
    is_yolo_ = is_yolo && !is_yolo11; // YOLOv8 only, not YOLO11
    is_yolo11_ = is_yolo11;
    is_yolo_seg_ = has_seg_proto || is_yolo11_seg;
    if (is_yolo_seg_ && !is_yolo && !is_yolo11) {
        fail_model(model_path, "YOLO-seg head tensors found without YOLO stem");
    }
    // Check YOLO11 first (more specific), then YOLOv8
    if (is_yolo11) {
        input_height_ = 640;
        input_width_ = 640;
        output_shapes_ = is_yolo11_seg ? std::vector<std::vector<int64_t>>{{1, 116, 8400}, {1, 32, 160, 160}}
                                       : std::vector<std::vector<int64_t>>{{1, 84, 8400}};
        if (batch_size_ != 1) {
            fail_model(model_path, "GGML YOLO11n path is fixed batch-1 (box math constants are single-batch)");
        }
    } else if (is_yolo) {
        input_height_ = 640;
        input_width_ = 640;
        output_shapes_ = is_yolo_seg_ ? std::vector<std::vector<int64_t>>{{1, 116, 8400}, {1, 32, 160, 160}}
                                      : std::vector<std::vector<int64_t>>{{1, 84, 8400}};
        if (batch_size_ != 1) {
            fail_model(model_path, "GGML YOLOv8n path is fixed batch-1 (box math constants are single-batch)");
        }
    }

    // RF-DETR family (detection first, seg later): hand-written GGUF
    // topology, no importer. Dispatch on the 91-class head: class_embed.bias
    // [91] plus a 2D [256,91] classifier weight (encoder or head rule, both
    // RF-DETR-only). S-1 intermediate contract: the loader exposes the
    // backbone+projector map [1,256,40,40]; the full detection graph lands in
    // a later slice and moves this sentinel.
    bool is_rfdetr = false;
    bool is_rfdetr_seg = false;
    {
        const auto seg_head = tensors.find("segmentation_head.bias");
        if (seg_head != tensors.end()) {
            is_rfdetr_seg = true;
        }
    }
    {
        const auto head = tensors.find("class_embed.bias");
        if (head != tensors.end() && head->second.dims == std::vector<uint32_t>{91}) {
            for (const auto& kv : tensors) {
                if (kv.second.dims == std::vector<uint32_t>{256, 91}) {
                    is_rfdetr = true;
                    break;
                }
            }
        }
    }
    if (is_rfdetr_seg) {
        if (is_yolo) {
            fail_model(model_path, "model has both YOLOv8n stem and RF-DETR-seg head; tensors:" + shapes_seen);
        }
        input_height_ = 432;
        input_width_ = 432;
        // App-facing outputs lead with dets/masks/labels for the stock
        // RfDetrSegmentationPostprocessor (boxes=0, masks=1, labels=2);
        // intermediates follow for parity tests.
        output_shapes_ = {{1, 200, 4},   {1, 200, 108, 108}, {1, 200, 91},  {1, 256, 36, 36}, {1, 200, 4},
                          {1, 200, 256}, {1, 200, 256},      {1, 200, 256}, {1, 200, 256}};
        if (batch_size_ != 1) {
            fail_model(model_path, "GGML RF-DETR-seg path is fixed batch-1 (window packing is single-batch)");
        }
    }
    if (is_rfdetr && !is_rfdetr_seg) {
        if (is_yolo) {
            fail_model(model_path, "model has both YOLOv8n stem and RF-DETR head; tensors:" + shapes_seen);
        }
        input_height_ = 560;
        input_width_ = 560;
        output_shapes_ = {{1, 300, 4},   {1, 300, 91},  {1, 256, 40, 40}, {1, 300, 4},
                          {1, 300, 256}, {1, 300, 256}, {1, 300, 256}};
        if (batch_size_ != 1) {
            fail_model(model_path, "GGML RF-DETR path is fixed batch-1 (window packing is single-batch)");
        }
    }

    std::string prefix;
    std::string fc_w_name;
    std::string fc_b_name;
    uint32_t fc_in_dim = 0;
    bool is_bottleneck = false;
    if (!is_yolo && !is_rfdetr && !is_rfdetr_seg) {
        struct VariantSpec {
            const char* prefix;
            const char* fc_weight;
            const char* fc_bias;
        };
        // resnet18-v1-7.onnx carries resnetv15_* (512-d FC);
        // resnet50-v1-12.onnx carries resnetv17_* (2048-d FC).
        constexpr VariantSpec kVariants[] = {
            {"resnetv15_", "resnetv15_dense0_weight", "resnetv15_dense0_bias"},
            {"resnetv17_", "resnetv17_dense0_weight", "resnetv17_dense0_bias"},
        };
        for (const auto& v : kVariants) {
            if (tensors.count(v.fc_weight) != 0 && tensors.count(v.fc_bias) != 0) {
                prefix = v.prefix;
                fc_w_name = v.fc_weight;
                fc_b_name = v.fc_bias;
                break;
            }
        }
        if (fc_w_name.empty()) {
            // Dims-only fallback: one 2D [1000,K]/[K,1000] classifier weight
            // decides; the stage prefix is stripped from its name so the stage
            // requires below resolve under the file's own naming.
            std::string w_name;
            for (const auto& kv : tensors) {
                const auto& d = kv.second.dims;
                if (d.size() == 2 && ((d[0] == 1000 && d[1] != 1000) || (d[1] == 1000 && d[0] != 1000))) {
                    if (!w_name.empty()) {
                        fail_model(model_path,
                                   "ambiguous classifier weights (need one 2D [1000,K]/[K,1000]); tensors:" +
                                       shapes_seen);
                    }
                    w_name = kv.first;
                }
            }
            if (w_name.empty()) {
                fail_model(model_path,
                           "no recognized ResNet-18/50 classifier pair (need [1000,512]/[1000,2048] weight + "
                           "[1000] bias); tensors:" +
                               shapes_seen);
            }
            constexpr char kFcSuffix[] = "dense0_weight";
            constexpr size_t kFcSuffixLen = sizeof(kFcSuffix) - 1;
            if (w_name.size() < kFcSuffixLen ||
                w_name.compare(w_name.size() - kFcSuffixLen, kFcSuffixLen, kFcSuffix, kFcSuffixLen) != 0) {
                fail_model(model_path,
                           "classifier weight '" + w_name + "' has no 'dense0_weight' suffix; tensors:" + shapes_seen);
            }
            prefix = w_name.substr(0, w_name.size() - kFcSuffixLen);
            fc_w_name = w_name;
            fc_b_name = prefix + "dense0_bias";
            require_tensor(tensors, fc_b_name, model_path, shapes_seen);
        }
        const LoadedTensor& fc_weight_probe = require_tensor(tensors, fc_w_name, model_path, shapes_seen);
        if (fc_weight_probe.dims.size() != 2 ||
            !((fc_weight_probe.dims[0] == 1000 &&
               (fc_weight_probe.dims[1] == 512 || fc_weight_probe.dims[1] == 2048)) ||
              (fc_weight_probe.dims[1] == 1000 &&
               (fc_weight_probe.dims[0] == 512 || fc_weight_probe.dims[0] == 2048)))) {
            fail_model(model_path, "classifier weight '" + fc_w_name +
                                       "' has unexpected dims (need [1000,512]/[1000,2048] "
                                       "or transposed); tensors:" +
                                       shapes_seen);
        }
        fc_in_dim = (fc_weight_probe.dims[0] == 1000) ? fc_weight_probe.dims[1] : fc_weight_probe.dims[0];
        // FC input dim decides the block topology; the stage requires below are
        // the bottleneck tensor-presence check (a 2048-d file without bottleneck
        // stage tensors fails on the first missing/wrong-shaped conv).
        is_bottleneck = (fc_in_dim == 2048);
    } // end if (!is_yolo): ResNet dispatch

    // Constant fills staged here: tensor storage is only valid after the
    // backend buffer is allocated, so weights are memcpy'd post-alloc.
    struct PendingFill {
        struct ggml_tensor* tensor = nullptr;
        std::vector<float> data;
    };
    std::vector<PendingFill> pending;
    // ConvTranspose's CPU kernel supports F16 weights only; the kernel is
    // filled after the backend buffer is allocated, like the F32 constants.
    struct PendingFillF16 {
        struct ggml_tensor* tensor = nullptr;
        std::vector<ggml_fp16_t> data;
    };
    std::vector<PendingFillF16> pending_f16;
    // Static I32 index vectors (RF-DETR window packing for get_rows) staged
    // the same way: storage is only valid after the backend buffer exists.
    struct PendingFillI32 {
        struct ggml_tensor* tensor = nullptr;
        std::vector<int32_t> data;
    };
    std::vector<PendingFillI32> pending_i32;
    auto new_const_4d = [&](int64_t n0, int64_t n1, int64_t n2, int64_t n3,
                            std::vector<float> data) -> struct ggml_tensor* {
        struct ggml_tensor* t = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, n0, n1, n2, n3);
        pending.push_back(PendingFill{t, std::move(data)});
        return t;
    };
    auto new_const_4d_f16 = [&](int64_t n0, int64_t n1, int64_t n2, int64_t n3,
                                std::vector<float> data) -> struct ggml_tensor* {
        struct ggml_tensor* t = ggml_new_tensor_4d(ctx_, GGML_TYPE_F16, n0, n1, n2, n3);
        std::vector<ggml_fp16_t> half(data.size());
        ggml_fp32_to_fp16_row(data.data(), half.data(), static_cast<int64_t>(data.size()));
        pending_f16.push_back(PendingFillF16{t, std::move(half)});
        return t;
    };

    auto folded_conv = [&](const std::string& conv_name, const std::string& bn_prefix, uint32_t O, uint32_t I,
                           uint32_t H, uint32_t W) -> FoldedConv {
        require_dims(require_tensor(tensors, conv_name, model_path, shapes_seen), conv_name, model_path, {O, I, H, W},
                     shapes_seen);
        auto folded = fold_bn_and_permute_kernel(tensors, conv_name, bn_prefix, model_path, shapes_seen);
        FoldedConv c;
        c.weight = new_const_4d(W, H, I, O, std::move(folded.first));
        c.bias = new_const_4d(1, 1, O, 1, std::move(folded.second));
        return c;
    };

    const int64_t N = static_cast<int64_t>(batch_size_);
    // feat_tensor_ is the graph input [W,H,C,N], fed (NCHW-coincident copy)
    // each inference. Geometry is per-model (members set at dispatch).
    feat_tensor_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, input_width_, input_height_, kInputChannels, N);

    struct ggml_tensor* x = feat_tensor_;
    struct ggml_tensor* logits = nullptr;
    struct ggml_tensor* seg_proto = nullptr;
    // RF-DETR intermediate outputs (encoder selection, decoder layers) ride
    // alongside the primary logits until the full graph lands.
    struct ggml_tensor* aux_out = nullptr;
    struct ggml_tensor* dec0_out = nullptr;
    struct ggml_tensor* dec1_out = nullptr;
    struct ggml_tensor* dec2_out = nullptr;
    struct ggml_tensor* dets_out = nullptr;
    struct ggml_tensor* hlabels = nullptr;
    struct ggml_tensor* dec3_out = nullptr;
    struct ggml_tensor* masks_out = nullptr;

    if (!is_yolo && !is_rfdetr && !is_rfdetr_seg) {

        // Stem: conv7x7/s2/p3 64ch -> folded bias -> relu -> maxpool k3/s2/p1.
        // 224 -> conv -> 112 -> pool -> 56. Shared by both variants.
        {
            const FoldedConv stem = folded_conv(prefix + "conv0_weight", prefix + "batchnorm0_", 64, 3, 7, 7);
            x = conv_bias_add(ctx_, x, stem, 2, 3, true);
            x = ggml_pool_2d(ctx_, x, GGML_OP_POOL_MAX, 3, 3, 2, 2, 1.0f, 1.0f);
        }

        if (!is_bottleneck) {
            // ResNet-18: layer1 2x basic(64, s1, no downsample), 56x56x64.
            {
                const FoldedConv c0 =
                    folded_conv(prefix + "stage1_conv0_weight", prefix + "stage1_batchnorm0_", 64, 64, 3, 3);
                const FoldedConv c1 =
                    folded_conv(prefix + "stage1_conv1_weight", prefix + "stage1_batchnorm1_", 64, 64, 3, 3);
                x = basic_block(ctx_, x, c0, c1, nullptr, 1);
                const FoldedConv c2 =
                    folded_conv(prefix + "stage1_conv2_weight", prefix + "stage1_batchnorm2_", 64, 64, 3, 3);
                const FoldedConv c3 =
                    folded_conv(prefix + "stage1_conv3_weight", prefix + "stage1_batchnorm3_", 64, 64, 3, 3);
                x = basic_block(ctx_, x, c2, c3, nullptr, 1);
            }

            // Layers 2..4: basic(C, s2 + 1x1 downsample conv2/bn2) + basic(C, s1).
            // 56 -> 28 (128ch) -> 14 (256ch) -> 7 (512ch).
            for (int stage = 2; stage <= 4; ++stage) {
                const uint32_t C = stage == 2 ? 128 : (stage == 3 ? 256 : 512);
                const uint32_t Cin = C / 2;
                const std::string pre = prefix + "stage" + std::to_string(stage) + "_";
                const FoldedConv c0 = folded_conv(pre + "conv0_weight", pre + "batchnorm0_", C, Cin, 3, 3);
                const FoldedConv c1 = folded_conv(pre + "conv1_weight", pre + "batchnorm1_", C, C, 3, 3);
                const FoldedConv ds = folded_conv(pre + "conv2_weight", pre + "batchnorm2_", C, Cin, 1, 1);
                x = basic_block(ctx_, x, c0, c1, &ds, 2);
                const FoldedConv c3 = folded_conv(pre + "conv3_weight", pre + "batchnorm3_", C, C, 3, 3);
                const FoldedConv c4 = folded_conv(pre + "conv4_weight", pre + "batchnorm4_", C, C, 3, 3);
                x = basic_block(ctx_, x, c3, c4, nullptr, 1);
            }
        } else {
            // ResNet-50: stages [3,4,6,3] bottleneck blocks, widths 64/128/256/512
            // expanding 4x. Every stage entry (block 0) carries the stride on the
            // 1x1 reduce (1 on stage 1, 2 below) and a 1x1 downsample on the skip;
            // later blocks are stride-1 with identity skips. Per-block conv
            // indices: block 0 -> (0,1,2), block b>=1 -> (3b+1,3b+2,3b+3);
            // the downsample is always conv3/bn3.
            constexpr uint32_t kBlocks[4] = {3, 4, 6, 3};
            constexpr uint32_t kWidth[4] = {64, 128, 256, 512};
            uint32_t in_ch = 64; // stem output channels
            for (int s = 0; s < 4; ++s) {
                const uint32_t C = kWidth[s];
                const int entry_stride = (s == 0) ? 1 : 2;
                const std::string pre = prefix + "stage" + std::to_string(s + 1) + "_";
                for (uint32_t b = 0; b < kBlocks[s]; ++b) {
                    const uint32_t base = (b == 0) ? 0 : 3 * b + 1;
                    const uint32_t in = (b == 0) ? in_ch : 4 * C;
                    const int stride = (b == 0) ? entry_stride : 1;
                    const FoldedConv reduce = folded_conv(pre + "conv" + std::to_string(base) + "_weight",
                                                          pre + "batchnorm" + std::to_string(base) + "_", C, in, 1, 1);
                    const FoldedConv mid = folded_conv(pre + "conv" + std::to_string(base + 1) + "_weight",
                                                       pre + "batchnorm" + std::to_string(base + 1) + "_", C, C, 3, 3);
                    const FoldedConv expand =
                        folded_conv(pre + "conv" + std::to_string(base + 2) + "_weight",
                                    pre + "batchnorm" + std::to_string(base + 2) + "_", 4 * C, C, 1, 1);
                    if (b == 0) {
                        const FoldedConv ds =
                            folded_conv(pre + "conv3_weight", pre + "batchnorm3_", 4 * C, in_ch, 1, 1);
                        x = bottleneck_block(ctx_, x, reduce, mid, expand, &ds, stride);
                    } else {
                        x = bottleneck_block(ctx_, x, reduce, mid, expand, nullptr, stride);
                    }
                }
                in_ch = 4 * C;
            }
        }

        // Head: global avgpool k7 (7x7 -> 1x1) -> reshape K -> FC.
        const LoadedTensor& fc_weight = require_tensor(tensors, fc_w_name, model_path, shapes_seen);
        const LoadedTensor& fc_bias = require_tensor(tensors, fc_b_name, model_path, shapes_seen);
        // Normalize the weight to ggml ne order [K, 1000] (ONNX Gemm transB=1:
        // file holds row-major [1000,K], i.e. W[n][k] at n*K+k). ggml is
        // column-major and ggml_mul_mat(a,b) needs a->ne[0] == b->ne[0],
        // yielding [a->ne[1], b->ne[1]] with dst[n,m] = sum_k a[k,n]*b[k,m]:
        // row-major [1000,K] bytes already coincide with ne=[K,1000]
        // element (k,n) at k+n*K, so they copy verbatim. A file already in
        // [K,1000] row-major order is transposed into place instead.
        std::vector<float> w_normalized(static_cast<size_t>(fc_in_dim) * 1000);
        if (fc_weight.dims == std::vector<uint32_t>{1000, fc_in_dim}) {
            w_normalized = fc_weight.data;
        } else if (fc_weight.dims == std::vector<uint32_t>{fc_in_dim, 1000}) {
            for (uint32_t k = 0; k < fc_in_dim; ++k) {
                for (int64_t n = 0; n < 1000; ++n) {
                    w_normalized[static_cast<size_t>(k) + static_cast<size_t>(n) * fc_in_dim] =
                        fc_weight.data[static_cast<size_t>(k) * 1000 + n];
                }
            }
        } else {
            fail_model(model_path,
                       "tensor '" + fc_w_name +
                           "' has unexpected dims (need [1000,K] or [K,1000], K=512/2048); tensors:" + shapes_seen);
        }
        require_dims(fc_bias, fc_b_name, model_path, {1000}, shapes_seen);

        struct ggml_tensor* fc_w = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, fc_in_dim, 1000);
        pending.push_back(PendingFill{fc_w, std::move(w_normalized)});
        struct ggml_tensor* fc_b = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, kNumClasses);
        pending.push_back(PendingFill{fc_b, fc_bias.data});

        struct ggml_tensor* gap = ggml_pool_2d(ctx_, x, GGML_OP_POOL_AVG, 7, 7, 1, 1, 0.0f, 0.0f);
        struct ggml_tensor* feat = ggml_reshape_2d(ctx_, gap, fc_in_dim, N);
        struct ggml_tensor* mm = ggml_mul_mat(ctx_, fc_w, feat);
        logits = ggml_add(ctx_, mm, ggml_repeat(ctx_, fc_b, mm));
    } else if (is_yolo) {
        // ---- YOLOv8n: fixed 1x3x640x640 in, [1,84,8400] raw out ----
        // Helpers share new_const_4d/pending above. Conventions: NCHW blobs map
        // to ggml [W,H,C,N] (reversed axes, coincident storage); kernels
        // [O,I,H,W] -> [KW,KH,IC,OC] verbatim (D-8 coincidence).
        auto ykernel = [&](const std::string& base, uint32_t O, uint32_t I, uint32_t H,
                           uint32_t W) -> struct ggml_tensor* {
            const LoadedTensor& w = require_tensor(tensors, base + ".weight", model_path, shapes_seen);
            require_dims(w, base + ".weight", model_path, {O, I, H, W}, shapes_seen);
            return new_const_4d(W, H, I, O, w.data);
        };
        auto ybias = [&](const std::string& base, uint32_t O) -> struct ggml_tensor* {
            const LoadedTensor& b = require_tensor(tensors, base + ".bias", model_path, shapes_seen);
            require_dims(b, base + ".bias", model_path, {O}, shapes_seen);
            return new_const_4d(1, 1, O, 1, b.data);
        };
        auto yact = [&](struct ggml_tensor* xx, struct ggml_tensor* w, struct ggml_tensor* b, int s,
                        int p) -> struct ggml_tensor* {
            struct ggml_tensor* h = ggml_conv_2d(ctx_, w, xx, s, s, p, p, 1, 1);
            h = ggml_add(ctx_, h, ggml_repeat(ctx_, b, h));
            return ggml_mul(ctx_, h, ggml_sigmoid(ctx_, h));
        };
        auto yconv = [&](struct ggml_tensor* xx, struct ggml_tensor* w, struct ggml_tensor* b, int s,
                         int p) -> struct ggml_tensor* {
            struct ggml_tensor* h = ggml_conv_2d(ctx_, w, xx, s, s, p, p, 1, 1);
            return ggml_add(ctx_, h, ggml_repeat(ctx_, b, h));
        };
        // C2f block (nodes 6-20 pattern): cv1 1x1 -> split -> n bottlenecks
        // (3x3+silu, 3x3+silu [, +residual]) chained -> concat all -> cv2 1x1.
        auto c2f = [&](struct ggml_tensor* xx, const std::string& pre, uint32_t C, uint32_t Cin, int n,
                       bool residual) -> struct ggml_tensor* {
            struct ggml_tensor* h =
                yact(xx, ykernel(pre + ".cv1.conv", C, Cin, 1, 1), ybias(pre + ".cv1.conv", C), 1, 0);
            h = ggml_cont(ctx_, h);
            auto halves = split_half(ctx_, h);
            std::vector<struct ggml_tensor*> parts = {halves.first, halves.second};
            struct ggml_tensor* cur = halves.second;
            for (int i = 0; i < n; ++i) {
                const std::string bpre = pre + ".m." + std::to_string(i) + ".";
                struct ggml_tensor* b =
                    yact(cur, ykernel(bpre + "cv1.conv", C / 2, C / 2, 3, 3), ybias(bpre + "cv1.conv", C / 2), 1, 1);
                b = yact(b, ykernel(bpre + "cv2.conv", C / 2, C / 2, 3, 3), ybias(bpre + "cv2.conv", C / 2), 1, 1);
                if (residual) {
                    b = ggml_add(ctx_, cur, b);
                }
                parts.push_back(b);
                cur = b;
            }
            struct ggml_tensor* cat = concat_ch(ctx_, parts);
            const uint32_t cat_ch = C / 2 * static_cast<uint32_t>(n + 2);
            return yact(cat, ykernel(pre + ".cv2.conv", C, cat_ch, 1, 1), ybias(pre + ".cv2.conv", C), 1, 0);
        };
        // Box-tail constants (NCHW -> ggml reversed, coincident storage):
        // [1,2,8400] anchors -> [8400,1,2,1]; [1,8400] strides -> [8400,1,1,1].
        // Returned at natural shape (same-shape elementwise ops need no repeat).
        auto yconst = [&](const std::string& name, std::initializer_list<uint32_t> dims) -> struct ggml_tensor* {
            const LoadedTensor& t = require_tensor(tensors, name, model_path, shapes_seen);
            if (t.dims != std::vector<uint32_t>(dims)) {
                fail_model(model_path, "tensor '" + name + "' has unexpected dims; tensors:" + shapes_seen);
            }
            if (t.dims == std::vector<uint32_t>{1, 2, 8400}) {
                return new_const_4d(8400, 1, 2, 1, t.data);
            }
            if (t.dims == std::vector<uint32_t>{1, 8400}) {
                return new_const_4d(8400, 1, 1, 1, t.data);
            }
            fail_model(model_path, "tensor '" + name + "' has unsupported const shape; tensors:" + shapes_seen);
            return nullptr;
        };

        if (is_yolo11_) {
            // ---- YOLO11n: fixed 1x3x640x640 in, [1,84,8400] raw out ----
            auto c3k2_simple = [&](struct ggml_tensor* xx, const std::string& pre,
                                   struct ggml_tensor** dbg_cv1 = nullptr, struct ggml_tensor** dbg_m0_cv1 = nullptr,
                                   struct ggml_tensor** dbg_m0_cv2 = nullptr, struct ggml_tensor** dbg_m0_add = nullptr,
                                   struct ggml_tensor** dbg_cv2 = nullptr) -> struct ggml_tensor* {
                const LoadedTensor& cv1w = require_tensor(tensors, pre + ".cv1.conv.weight", model_path, shapes_seen);
                const uint32_t cv1_out = cv1w.dims[0];
                const uint32_t Cin = cv1w.dims[1];
                struct ggml_tensor* h =
                    yact(xx, ykernel(pre + ".cv1.conv", cv1_out, Cin, 1, 1), ybias(pre + ".cv1.conv", cv1_out), 1, 0);
                if (dbg_cv1)
                    *dbg_cv1 = h;
                h = ggml_cont(ctx_, h);
                auto halves = split_half(ctx_, h);
                // hid = cv1_out / 4 for all YOLO11 C3k2_simple blocks (32->8, 64->16, 128->32)
                const uint32_t hid = cv1_out / 4;
                const LoadedTensor& b1w =
                    require_tensor(tensors, pre + ".m.0.cv1.conv.weight", model_path, shapes_seen);
                const LoadedTensor& b2w =
                    require_tensor(tensors, pre + ".m.0.cv2.conv.weight", model_path, shapes_seen);
                const uint32_t bout = b2w.dims[0];
                struct ggml_tensor* b = yact(halves.second, ykernel(pre + ".m.0.cv1.conv", hid, cv1_out / 2, 3, 3),
                                             ybias(pre + ".m.0.cv1.conv", hid), 1, 1);
                if (dbg_m0_cv1)
                    *dbg_m0_cv1 = b;
                b = yact(b, ykernel(pre + ".m.0.cv2.conv", bout, hid, 3, 3), ybias(pre + ".m.0.cv2.conv", bout), 1, 1);
                if (dbg_m0_cv2)
                    *dbg_m0_cv2 = b;
                b = ggml_add(ctx_, halves.second, b);
                if (dbg_m0_add)
                    *dbg_m0_add = b;
                struct ggml_tensor* cat = concat_ch(ctx_, {halves.first, halves.second, b});
                const LoadedTensor& cv2w = require_tensor(tensors, pre + ".cv2.conv.weight", model_path, shapes_seen);
                const uint32_t cv2_out = cv2w.dims[0];
                struct ggml_tensor* out = yact(cat, ykernel(pre + ".cv2.conv", cv2_out, cv1_out + bout, 1, 1),
                                               ybias(pre + ".cv2.conv", cv2_out), 1, 0);
                if (dbg_cv2)
                    *dbg_cv2 = out;
                return out;
            };
            auto c3k2_c3k = [&](struct ggml_tensor* xx, const std::string& pre) -> struct ggml_tensor* {
                const LoadedTensor& cv1w = require_tensor(tensors, pre + ".cv1.conv.weight", model_path, shapes_seen);
                const uint32_t cv1_out = cv1w.dims[0];
                const uint32_t Cin = cv1w.dims[1];
                struct ggml_tensor* h =
                    yact(xx, ykernel(pre + ".cv1.conv", cv1_out, Cin, 1, 1), ybias(pre + ".cv1.conv", cv1_out), 1, 0);
                h = ggml_cont(ctx_, h);
                auto halves = split_half(ctx_, h);
                const uint32_t half = cv1_out / 2;
                const LoadedTensor& ic1w =
                    require_tensor(tensors, pre + ".m.0.cv1.conv.weight", model_path, shapes_seen);
                const uint32_t ic = ic1w.dims[0];
                struct ggml_tensor* a = yact(halves.second, ykernel(pre + ".m.0.cv1.conv", ic, half, 1, 1),
                                             ybias(pre + ".m.0.cv1.conv", ic), 1, 0);
                struct ggml_tensor* bside = yact(halves.second, ykernel(pre + ".m.0.cv2.conv", ic, half, 1, 1),
                                                 ybias(pre + ".m.0.cv2.conv", ic), 1, 0);
                const LoadedTensor& k1w =
                    require_tensor(tensors, pre + ".m.0.m.0.cv1.conv.weight", model_path, shapes_seen);
                const uint32_t kc = k1w.dims[0];
                struct ggml_tensor* cur = yact(a, ykernel(pre + ".m.0.m.0.cv1.conv", kc, ic, 3, 3),
                                               ybias(pre + ".m.0.m.0.cv1.conv", kc), 1, 1);
                cur = yact(cur, ykernel(pre + ".m.0.m.0.cv2.conv", kc, kc, 3, 3), ybias(pre + ".m.0.m.0.cv2.conv", kc),
                           1, 1);
                cur = ggml_add(ctx_, a, cur);
                struct ggml_tensor* cur2 = yact(cur, ykernel(pre + ".m.0.m.1.cv1.conv", kc, kc, 3, 3),
                                                ybias(pre + ".m.0.m.1.cv1.conv", kc), 1, 1);
                cur2 = yact(cur2, ykernel(pre + ".m.0.m.1.cv2.conv", kc, kc, 3, 3),
                            ybias(pre + ".m.0.m.1.cv2.conv", kc), 1, 1);
                cur2 = ggml_add(ctx_, cur, cur2);
                struct ggml_tensor* inner = concat_ch(ctx_, {cur2, bside});
                const LoadedTensor& cv3w =
                    require_tensor(tensors, pre + ".m.0.cv3.conv.weight", model_path, shapes_seen);
                const uint32_t cv3_out = cv3w.dims[0];
                struct ggml_tensor* c3 = yact(inner, ykernel(pre + ".m.0.cv3.conv", cv3_out, 2 * ic, 1, 1),
                                              ybias(pre + ".m.0.cv3.conv", cv3_out), 1, 0);
                struct ggml_tensor* cat = concat_ch(ctx_, {halves.first, halves.second, c3});
                const LoadedTensor& cv2w = require_tensor(tensors, pre + ".cv2.conv.weight", model_path, shapes_seen);
                const uint32_t cv2_out = cv2w.dims[0];
                return yact(cat, ykernel(pre + ".cv2.conv", cv2_out, cv1_out + cv3_out, 1, 1),
                            ybias(pre + ".cv2.conv", cv2_out), 1, 0);
            };
            auto ydw_silu = [&](struct ggml_tensor* xx, const std::string& wname, const std::string& bname,
                                uint32_t C) -> struct ggml_tensor* {
                const LoadedTensor& w = require_tensor(tensors, wname, model_path, shapes_seen);
                require_dims(w, wname, model_path, {C, 1, 3, 3}, shapes_seen);
                struct ggml_tensor* kw = new_const_4d(9, C, 1, 1, w.data);
                const LoadedTensor& b = require_tensor(tensors, bname, model_path, shapes_seen);
                require_dims(b, bname, model_path, {C}, shapes_seen);
                struct ggml_tensor* kb = new_const_4d(C, 1, 1, 1, b.data);
                struct ggml_tensor* h = ggml_ops::depthwise_conv3x3(ctx_, ggml_cont(ctx_, xx), kw, kb);
                h = ggml_cont(ctx_, h);
                return ggml_mul(ctx_, h, ggml_sigmoid(ctx_, h));
            };
            struct ggml_tensor* y11 = yact(x, ykernel("model.0.conv", 16, 3, 3, 3), ybias("model.0.conv", 16), 2, 1);
            if (is_yolo11_)
                debug_m0_ = y11;
            y11 = yact(y11, ykernel("model.1.conv", 32, 16, 3, 3), ybias("model.1.conv", 32), 2, 1);
            if (is_yolo11_)
                debug_m1_ = y11;
            y11 = c3k2_simple(y11, "model.2");
            if (is_yolo11_)
                debug_m2_ = y11;

            y11 = yact(y11, ykernel("model.3.conv", 64, 64, 3, 3), ybias("model.3.conv", 64), 2, 1);
            if (is_yolo11_)
                debug_m3_ = y11;
            struct ggml_tensor* m4 = c3k2_simple(y11, "model.4", &debug_m4_cv1_, &debug_m4_m0_cv1_, &debug_m4_m0_cv2_,
                                                 &debug_m4_m0_add_, &debug_m4_cv2_);

            y11 = yact(m4, ykernel("model.5.conv", 128, 128, 3, 3), ybias("model.5.conv", 128), 2, 1);
            struct ggml_tensor* m6 = c3k2_c3k(y11, "model.6");

            y11 = yact(m6, ykernel("model.7.conv", 256, 128, 3, 3), ybias("model.7.conv", 256), 2, 1);
            y11 = c3k2_c3k(y11, "model.8");

            {
                struct ggml_tensor* h =
                    yact(y11, ykernel("model.9.cv1.conv", 128, 256, 1, 1), ybias("model.9.cv1.conv", 128), 1, 0);
                struct ggml_tensor* p1 = ggml_pool_2d(ctx_, h, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2.0f, 2.0f);
                struct ggml_tensor* p2 = ggml_pool_2d(ctx_, p1, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2.0f, 2.0f);
                struct ggml_tensor* p3 = ggml_pool_2d(ctx_, p2, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2.0f, 2.0f);
                y11 = yact(concat_ch(ctx_, {h, p1, p2, p3}), ykernel("model.9.cv2.conv", 256, 512, 1, 1),
                           ybias("model.9.cv2.conv", 256), 1, 0);
            }
            {
                struct ggml_tensor* h =
                    yact(y11, ykernel("model.10.cv1.conv", 256, 256, 1, 1), ybias("model.10.cv1.conv", 256), 1, 0);
                h = ggml_cont(ctx_, h);
                auto halves10 = split_half(ctx_, h);
                struct ggml_tensor* qkv = yconv(halves10.second, ykernel("model.10.m.0.attn.qkv.conv", 256, 128, 1, 1),
                                                ybias("model.10.m.0.attn.qkv.conv", 256), 1, 0);
                struct ggml_tensor* t = ggml_reshape_4d(ctx_, qkv, 400, 128, 2, 1);
                const size_t nb1 = static_cast<size_t>(400) * sizeof(float);
                const size_t nb2 = nb1 * static_cast<size_t>(128);
                const float kScale = 0.17677669f;
                struct ggml_tensor* outs[2];
                struct ggml_tensor* vts[2];
                for (int hh = 0; hh < 2; ++hh) {
                    const size_t hoff = static_cast<size_t>(hh) * nb2;
                    struct ggml_tensor* q0 = ggml_view_2d(ctx_, t, 400, 32, nb1, hoff + static_cast<size_t>(0) * nb1);
                    struct ggml_tensor* k0 = ggml_view_2d(ctx_, t, 400, 32, nb1, hoff + static_cast<size_t>(32) * nb1);
                    struct ggml_tensor* vt = ggml_view_2d(ctx_, t, 400, 64, nb1, hoff + static_cast<size_t>(64) * nb1);
                    struct ggml_tensor* q = ggml_cont(ctx_, ggml_permute(ctx_, q0, 1, 0, 2, 3));
                    struct ggml_tensor* k = ggml_cont(ctx_, ggml_permute(ctx_, k0, 1, 0, 2, 3));
                    struct ggml_tensor* s = ggml_scale(ctx_, ggml_mul_mat(ctx_, k, q), kScale);
                    // F64-accumulator rowwise softmax (same kernel RF-DETR uses for its
                    // 17 softmax nodes): keeps the 400-wide attention drift in budget.
                    struct ggml_tensor* pr = ggml_ops::softmax_rows(ctx_, s);
                    outs[hh] = ggml_mul_mat(ctx_, pr, vt);
                    vts[hh] = vt;
                }
                struct ggml_tensor* att = ggml_cont(ctx_, ggml_concat(ctx_, outs[0], outs[1], 1));
                struct ggml_tensor* vall = ggml_cont(ctx_, ggml_concat(ctx_, vts[0], vts[1], 1));
                struct ggml_tensor* att_map = ggml_reshape_4d(ctx_, att, 20, 20, 128, 1);
                struct ggml_tensor* v_map = ggml_reshape_4d(ctx_, vall, 20, 20, 128, 1);
                struct ggml_tensor* pe = [&]() -> struct ggml_tensor* {
                    const LoadedTensor& w =
                        require_tensor(tensors, "model.10.m.0.attn.pe.conv.weight", model_path, shapes_seen);
                    require_dims(w, "model.10.m.0.attn.pe.conv.weight", model_path, {128, 1, 3, 3}, shapes_seen);
                    struct ggml_tensor* kw = new_const_4d(9, 128, 1, 1, w.data);
                    const LoadedTensor& b =
                        require_tensor(tensors, "model.10.m.0.attn.pe.conv.bias", model_path, shapes_seen);
                    require_dims(b, "model.10.m.0.attn.pe.conv.bias", model_path, {128}, shapes_seen);
                    struct ggml_tensor* kb = new_const_4d(128, 1, 1, 1, b.data);
                    return ggml_cont(ctx_, ggml_ops::depthwise_conv3x3(ctx_, ggml_cont(ctx_, v_map), kw, kb));
                }();
                struct ggml_tensor* amix = ggml_add(ctx_, att_map, pe);
                struct ggml_tensor* proj = yconv(amix, ykernel("model.10.m.0.attn.proj.conv", 128, 128, 1, 1),
                                                 ybias("model.10.m.0.attn.proj.conv", 128), 1, 0);
                struct ggml_tensor* ares = ggml_add(ctx_, halves10.second, proj);
                struct ggml_tensor* ff = yact(ares, ykernel("model.10.m.0.ffn.0.conv", 256, 128, 1, 1),
                                              ybias("model.10.m.0.ffn.0.conv", 256), 1, 0);
                ff = yconv(ff, ykernel("model.10.m.0.ffn.1.conv", 128, 256, 1, 1),
                           ybias("model.10.m.0.ffn.1.conv", 128), 1, 0);
                struct ggml_tensor* bout = ggml_add(ctx_, ares, ff);
                y11 = yact(concat_ch(ctx_, {halves10.first, bout}), ykernel("model.10.cv2.conv", 256, 256, 1, 1),
                           ybias("model.10.cv2.conv", 256), 1, 0);
            }
            struct ggml_tensor* p5 = y11;
            if (is_yolo11_)
                debug_p5_ = p5;
            struct ggml_tensor* up1 = ggml_upscale(ctx_, p5, 2, GGML_SCALE_MODE_NEAREST);
            if (is_yolo11_)
                debug_up1_ = up1;
            struct ggml_tensor* n13 = c3k2_simple(concat_ch(ctx_, {up1, m6}), "model.13");
            if (is_yolo11_)
                debug_n13_ = n13;
            struct ggml_tensor* up2 = ggml_upscale(ctx_, n13, 2, GGML_SCALE_MODE_NEAREST);
            if (is_yolo11_)
                debug_up2_ = up2;
            struct ggml_tensor* p3f = c3k2_simple(concat_ch(ctx_, {up2, m4}), "model.16");
            struct ggml_tensor* d5 =
                yact(p3f, ykernel("model.17.conv", 64, 64, 3, 3), ybias("model.17.conv", 64), 2, 1);
            if (is_yolo11_)
                debug_d5_ = d5;
            struct ggml_tensor* p4f = c3k2_simple(concat_ch(ctx_, {d5, n13}), "model.19");
            struct ggml_tensor* d6 =
                yact(p4f, ykernel("model.20.conv", 128, 128, 3, 3), ybias("model.20.conv", 128), 2, 1);
            if (is_yolo11_)
                debug_d6_ = d6;
            struct ggml_tensor* p5f = c3k2_c3k(concat_ch(ctx_, {d6, p5}), "model.22");
            struct ggml_tensor* box_parts[3];
            struct ggml_tensor* cls_parts[3];
            struct ggml_tensor* feats[3] = {p3f, p4f, p5f};
            if (is_yolo11_) {
                debug_feat_[0] = p3f;
                debug_feat_[1] = p4f;
                debug_feat_[2] = p5f;
                debug_m4_ = m4;
                debug_m6_ = m6;
            }
            for (int s = 0; s < 3; ++s) {
                const std::string pre = "model.23.cv2." + std::to_string(s);
                struct ggml_tensor* hb = yact(feats[s], ykernel(pre + ".0.conv", 64, feats[s]->ne[2], 3, 3),
                                              ybias(pre + ".0.conv", 64), 1, 1);
                hb = yact(hb, ykernel(pre + ".1.conv", 64, 64, 3, 3), ybias(pre + ".1.conv", 64), 1, 1);
                hb = yconv(hb, ykernel(pre + ".2", 64, 64, 1, 1), ybias(pre + ".2", 64), 1, 0);
                box_parts[s] = ggml_reshape_4d(ctx_, hb, hb->ne[0] * hb->ne[1], 1, 64, 1);
                const std::string pc = "model.23.cv3." + std::to_string(s);
                const uint32_t pc_in = static_cast<uint32_t>(feats[s]->ne[2]);
                struct ggml_tensor* hc = ydw_silu(feats[s], pc + ".0.0.conv.weight", pc + ".0.0.conv.bias", pc_in);
                hc = yact(hc, ykernel(pc + ".0.1.conv", 80, pc_in, 1, 1), ybias(pc + ".0.1.conv", 80), 1, 0);
                hc = ydw_silu(hc, pc + ".1.0.conv.weight", pc + ".1.0.conv.bias", 80);
                hc = yact(hc, ykernel(pc + ".1.1.conv", 80, 80, 1, 1), ybias(pc + ".1.1.conv", 80), 1, 0);
                hc = yconv(hc, ykernel(pc + ".2", 80, 80, 1, 1), ybias(pc + ".2", 80), 1, 0);
                cls_parts[s] = ggml_reshape_4d(ctx_, hc, hc->ne[0] * hc->ne[1], 1, 80, 1);
            }
            struct ggml_tensor* reg =
                ggml_concat(ctx_, ggml_concat(ctx_, box_parts[0], box_parts[1], 0), box_parts[2], 0);
            debug_reg_ = reg;
            struct ggml_tensor* clsc =
                ggml_concat(ctx_, ggml_concat(ctx_, cls_parts[0], cls_parts[1], 0), cls_parts[2], 0);
            struct ggml_tensor* dfl_rs = ggml_reshape_4d(ctx_, reg, 8400, 1, 16, 4);
            struct ggml_tensor* dfl_in = ggml_cont(ctx_, ggml_permute(ctx_, dfl_rs, 1, 3, 0, 2));
            struct ggml_tensor* dsm = ggml_soft_max(ctx_, dfl_in);
            struct ggml_tensor* dflat = ggml_reshape_2d(ctx_, dsm, 16, 33600);
            const LoadedTensor& dw = require_tensor(tensors, "model.23.dfl.conv.weight", model_path, shapes_seen);
            require_dims(dw, "model.23.dfl.conv.weight", model_path, {1, 16, 1, 1}, shapes_seen);
            struct ggml_tensor* dwq = new_const_4d(16, 1, 1, 1, dw.data);
            struct ggml_tensor* dm = ggml_mul_mat(ctx_, dwq, dflat);
            debug_dm_ = dm;
            struct ggml_tensor* d4 = ggml_reshape_4d(ctx_, dm, 8400, 1, 4, 1);
            debug_d4_ = d4;
            const size_t row4 = static_cast<size_t>(8400) * sizeof(float);
            struct ggml_tensor* xy = ggml_view_4d(ctx_, d4, 8400, 1, 2, 1, row4, row4, row4, 0);
            struct ggml_tensor* wh = ggml_view_4d(ctx_, d4, 8400, 1, 2, 1, row4, row4, row4, 2 * row4);
            // Converter provides Constant_12 (A), Constant_10 (B). Stride (Constant_15) missing - construct it.
            // Stride per anchor: 6400 anchors at stride 8 (P3), 1600 at stride 16 (P4), 400 at stride 32 (P5)
            struct ggml_tensor* Avec_flat = yconst("/model.23/Constant_12_output_0", {1, 8400});
            struct ggml_tensor* Avec = ggml_concat(ctx_, Avec_flat, Avec_flat, 2); // [8400,1,2,1]
            struct ggml_tensor* Bvec = yconst("/model.23/Constant_10_output_0", {1, 2, 8400});
            struct ggml_tensor* Bvec_reshaped = ggml_reshape_4d(ctx_, Bvec, 8400, 1, 2, 1);
            struct ggml_tensor* sub0 = ggml_sub(ctx_, Avec, xy);
            struct ggml_tensor* add1 = ggml_add(ctx_, Bvec_reshaped, wh);
            struct ggml_tensor* add2 = ggml_add(ctx_, sub0, add1);
            struct ggml_tensor* sub1 = ggml_sub(ctx_, add1, sub0);
            struct ggml_tensor* div1 = ggml_scale(ctx_, add2, 0.5f);
            struct ggml_tensor* cat2 = ggml_concat(ctx_, div1, sub1, 2);
            // Hardcode stride tensor: 6400x8, 1600x16, 400x32
            std::vector<float> stride_data(8400);
            std::fill(stride_data.begin(), stride_data.begin() + 6400, 8.0f);
            std::fill(stride_data.begin() + 6400, stride_data.begin() + 8000, 16.0f);
            std::fill(stride_data.begin() + 8000, stride_data.end(), 32.0f);
            struct ggml_tensor* stride_tensor = new_const_4d(8400, 1, 1, 1, stride_data);
            struct ggml_tensor* mul2 = ggml_mul(ctx_, cat2, ggml_repeat(ctx_, stride_tensor, cat2));
            struct ggml_tensor* ones = new_const_4d(8400, 1, 80, 1, std::vector<float>(8400 * 80, 1.0f));
            struct ggml_tensor* sig = ggml_div(ctx_, ones, ggml_add(ctx_, ones, ggml_exp(ctx_, ggml_neg(ctx_, clsc))));
            logits = ggml_concat(ctx_, mul2, sig, 2);
            if (is_yolo_seg_) {
                // ---- YOLO11-seg: mask coefficients (cv4) + prototype masks (proto) ----
                // Mask coefficients: cv4 branch per scale (C->32, 32->32, 1x1 32->32, no activation),
                // concatenated along the flat spatial dim and appended after box(4)+class(80) -> 116 channels.
                struct ggml_tensor* mask_parts[3];
                for (int s = 0; s < 3; ++s) {
                    const std::string pm = "model.23.cv4." + std::to_string(s);
                    struct ggml_tensor* hm = yact(feats[s], ykernel(pm + ".0.conv", 32, feats[s]->ne[2], 3, 3),
                                                  ybias(pm + ".0.conv", 32), 1, 1);
                    hm = yact(hm, ykernel(pm + ".1.conv", 32, 32, 3, 3), ybias(pm + ".1.conv", 32), 1, 1);
                    hm = yconv(hm, ykernel(pm + ".2", 32, 32, 1, 1), ybias(pm + ".2", 32), 1, 0);
                    mask_parts[s] = ggml_reshape_4d(ctx_, hm, hm->ne[0] * hm->ne[1], 1, 32, 1);
                }
                struct ggml_tensor* mask_coeff =
                    ggml_concat(ctx_, ggml_concat(ctx_, mask_parts[0], mask_parts[1], 0), mask_parts[2], 0);
                logits = ggml_concat(ctx_, logits, mask_coeff, 2); // [8400,1,116,1]

                // Prototype masks: P3 -> cv1(SiLU) -> 2x up ConvTranspose ->
                // cv2(SiLU) -> cv3(SiLU). The ONNX ConvTranspose weight is
                // [Cin,Cout,kH,kW]; ggml's p0 kernel takes [kW,kH,Cout,Cin]
                // and requires F16 weights on the CPU backend.
                const LoadedTensor& up_w =
                    require_tensor(tensors, "model.23.proto.upsample.weight", model_path, shapes_seen);
                require_dims(up_w, "model.23.proto.upsample.weight", model_path, {64, 64, 2, 2}, shapes_seen);
                // ONNX [Cin,Cout,kH,kW] row-major and ggml [kW,kH,Cout,Cin]
                // column-major have the same flat order for this 64/64/2/2
                // kernel (the same reversed-dims coincidence used by the conv
                // weights above), so the copy is intentional, not a no-op bug.
                struct ggml_tensor* up_weight = new_const_4d_f16(2, 2, 64, 64, up_w.data);
                struct ggml_tensor* proto = yact(feats[0], ykernel("model.23.proto.cv1.conv", 64, 64, 3, 3),
                                                 ybias("model.23.proto.cv1.conv", 64), 1, 1);
                proto = ggml_conv_transpose_2d_p0(ctx_, up_weight, proto, 2);
                proto = ggml_add(ctx_, proto, ggml_repeat(ctx_, ybias("model.23.proto.upsample", 64), proto));
                proto = yact(proto, ykernel("model.23.proto.cv2.conv", 64, 64, 3, 3),
                             ybias("model.23.proto.cv2.conv", 64), 1, 1);
                seg_proto = yact(proto, ykernel("model.23.proto.cv3.conv", 32, 64, 1, 1),
                                 ybias("model.23.proto.cv3.conv", 32), 1, 0); // [160,160,32,1]
            }
        } else {
            // --- backbone ---
            struct ggml_tensor* y =
                yact(x, ykernel("model.0.conv", 16, 3, 3, 3), ybias("model.0.conv", 16), 2, 1);      // 320
            y = yact(y, ykernel("model.1.conv", 32, 16, 3, 3), ybias("model.1.conv", 32), 2, 1);     // 160
            y = c2f(y, "model.2", 32, 32, 1, true);                                                  // 160
            y = yact(y, ykernel("model.3.conv", 64, 32, 3, 3), ybias("model.3.conv", 64), 2, 1);     // 80
            struct ggml_tensor* m4 = c2f(y, "model.4", 64, 64, 2, true);                             // 80
            y = yact(m4, ykernel("model.5.conv", 128, 64, 3, 3), ybias("model.5.conv", 128), 2, 1);  // 40
            struct ggml_tensor* m6 = c2f(y, "model.6", 128, 128, 2, true);                           // 40
            y = yact(m6, ykernel("model.7.conv", 256, 128, 3, 3), ybias("model.7.conv", 256), 2, 1); // 20
            y = c2f(y, "model.8", 256, 256, 1, true);                                                // 20
            // SPPF: cv1 1x1 128ch, 3x chained maxpool k5 s1 p2, concat, cv2 1x1 256ch.
            {
                struct ggml_tensor* h =
                    yact(y, ykernel("model.9.cv1.conv", 128, 256, 1, 1), ybias("model.9.cv1.conv", 128), 1, 0);
                struct ggml_tensor* p1 = ggml_pool_2d(ctx_, h, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2.0f, 2.0f);
                struct ggml_tensor* p2 = ggml_pool_2d(ctx_, p1, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2.0f, 2.0f);
                struct ggml_tensor* p3 = ggml_pool_2d(ctx_, p2, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2.0f, 2.0f);
                y = yact(concat_ch(ctx_, {h, p1, p2, p3}), ykernel("model.9.cv2.conv", 256, 512, 1, 1),
                         ybias("model.9.cv2.conv", 256), 1, 0);
            }
            struct ggml_tensor* p5 = y; // 20x20x256
            // --- neck ---
            struct ggml_tensor* up1 = ggml_upscale(ctx_, p5, 2, GGML_SCALE_MODE_NEAREST);              // 40
            struct ggml_tensor* n12 = c2f(concat_ch(ctx_, {up1, m6}), "model.12", 128, 384, 1, false); // 40
            struct ggml_tensor* up2 = ggml_upscale(ctx_, n12, 2, GGML_SCALE_MODE_NEAREST);             // 80
            struct ggml_tensor* p3f = c2f(concat_ch(ctx_, {up2, m4}), "model.15", 64, 192, 1, false);  // 80, P3
            struct ggml_tensor* d5 =
                yact(p3f, ykernel("model.16.conv", 64, 64, 3, 3), ybias("model.16.conv", 64), 2, 1);   // 40
            struct ggml_tensor* p4f = c2f(concat_ch(ctx_, {d5, n12}), "model.18", 128, 192, 1, false); // 40, P4
            struct ggml_tensor* d6 =
                yact(p4f, ykernel("model.19.conv", 128, 128, 3, 3), ybias("model.19.conv", 128), 2, 1); // 20
            struct ggml_tensor* p5f = c2f(concat_ch(ctx_, {d6, p5}), "model.21", 256, 384, 1, false);   // 20, P5
            // --- decoupled head per scale (P3/P4/P5): box branch 64ch, cls branch 80ch ---
            struct ggml_tensor* box_parts[3];
            struct ggml_tensor* cls_parts[3];
            struct ggml_tensor* feats[3] = {p3f, p4f, p5f};
            for (int s = 0; s < 3; ++s) {
                const std::string pre = "model.22.cv2." + std::to_string(s);
                struct ggml_tensor* hb = yact(feats[s], ykernel(pre + ".0.conv", 64, feats[s]->ne[2], 3, 3),
                                              ybias(pre + ".0.conv", 64), 1, 1);
                hb = yact(hb, ykernel(pre + ".1.conv", 64, 64, 3, 3), ybias(pre + ".1.conv", 64), 1, 1);
                hb = yconv(hb, ykernel(pre + ".2", 64, 64, 1, 1), ybias(pre + ".2", 64), 1, 0);
                box_parts[s] = ggml_reshape_4d(ctx_, hb, hb->ne[0] * hb->ne[1], 1, 64, 1);
                const std::string pc = "model.22.cv3." + std::to_string(s);
                struct ggml_tensor* hc =
                    yact(feats[s], ykernel(pc + ".0.conv", 80, feats[s]->ne[2], 3, 3), ybias(pc + ".0.conv", 80), 1, 1);
                hc = yact(hc, ykernel(pc + ".1.conv", 80, 80, 3, 3), ybias(pc + ".1.conv", 80), 1, 1);
                hc = yconv(hc, ykernel(pc + ".2", 80, 80, 1, 1), ybias(pc + ".2", 80), 1, 0);
                cls_parts[s] = ggml_reshape_4d(ctx_, hc, hc->ne[0] * hc->ne[1], 1, 80, 1);
            }
            // Concat along the flat spatial dim (NCHW axis=-1 == ggml dim 0).
            struct ggml_tensor* reg =
                ggml_concat(ctx_, ggml_concat(ctx_, box_parts[0], box_parts[1], 0), box_parts[2], 0); // [8400,1,64,1]
            struct ggml_tensor* clsc =
                ggml_concat(ctx_, ggml_concat(ctx_, cls_parts[0], cls_parts[1], 0), cls_parts[2], 0); // [8400,1,80,1]
            // DFL: split the 64 reg channels into 4 coords x 16 bins. ggml reshape
            // splits column-major, so [8400,1,64,1] -> [8400,1,16,4] yields
            // g = k + 16*c, exactly the row-major [1,4,16,8400] split. Permute
            // with axes (1,3,0,2) — ggml semantics result[axis_k] = input[k] —
            // giving [16,8400,4,1] with the 16 bins at ne[0] for softmax. Then
            // cont (the permuted view is strided) and softmax over the bins.
            struct ggml_tensor* dfl_rs = ggml_reshape_4d(ctx_, reg, 8400, 1, 16, 4);
            struct ggml_tensor* dfl_in = ggml_cont(ctx_, ggml_permute(ctx_, dfl_rs, 1, 3, 0, 2));
            struct ggml_tensor* dsm = ggml_soft_max(ctx_, dfl_in);
            struct ggml_tensor* dflat = ggml_reshape_2d(ctx_, dsm, 16, 33600);
            const LoadedTensor& dw = require_tensor(tensors, "model.22.dfl.conv.weight", model_path, shapes_seen);
            require_dims(dw, "model.22.dfl.conv.weight", model_path, {1, 16, 1, 1}, shapes_seen);
            struct ggml_tensor* dwq = new_const_4d(16, 1, 1, 1, dw.data);
            struct ggml_tensor* dm = ggml_mul_mat(ctx_, dwq, dflat); // [1,33600]
            struct ggml_tensor* d4 = ggml_reshape_4d(ctx_, dm, 8400, 1, 4, 1);
            // Box decode: xy = A - slice0, wh = B + slice1, center/corner combine,
            // /2, concat, *strides. The seg export numbers the same constants
            // differently from the detection export, so select by presence.
            const bool has_det_consts = tensors.count("/model.22/Constant_12_output_0") != 0;
            const std::string a_name =
                has_det_consts ? "/model.22/Constant_12_output_0" : "/model.22/Constant_15_output_0";
            const std::string b_name =
                has_det_consts ? "/model.22/Constant_13_output_0" : "/model.22/Constant_16_output_0";
            const std::string stride_name =
                has_det_consts ? "/model.22/Constant_15_output_0" : "/model.22/Constant_18_output_0";
            const size_t row4 = static_cast<size_t>(8400) * sizeof(float);
            struct ggml_tensor* xy = ggml_view_4d(ctx_, d4, 8400, 1, 2, 1, row4, row4, row4, 0);
            struct ggml_tensor* wh = ggml_view_4d(ctx_, d4, 8400, 1, 2, 1, row4, row4, row4, 2 * row4);
            struct ggml_tensor* A = yconst(a_name, {1, 2, 8400});
            struct ggml_tensor* B = yconst(b_name, {1, 2, 8400});
            struct ggml_tensor* sub0 = ggml_sub(ctx_, A, xy);
            struct ggml_tensor* add1 = ggml_add(ctx_, B, wh);
            struct ggml_tensor* add2 = ggml_add(ctx_, sub0, add1);
            struct ggml_tensor* sub1 = ggml_sub(ctx_, add1, sub0);
            struct ggml_tensor* div1 = ggml_scale(ctx_, add2, 0.5f);
            struct ggml_tensor* cat2 = ggml_concat(ctx_, div1, sub1, 2);
            struct ggml_tensor* mul2 = ggml_mul(ctx_, cat2, ggml_repeat(ctx_, yconst(stride_name, {1, 8400}), cat2));
            // Cls sigmoid: 1 / (1 + exp(-x)).
            struct ggml_tensor* ones = new_const_4d(8400, 1, 80, 1, std::vector<float>(8400 * 80, 1.0f));
            struct ggml_tensor* sig = ggml_div(ctx_, ones, ggml_add(ctx_, ones, ggml_exp(ctx_, ggml_neg(ctx_, clsc))));
            logits = ggml_concat(ctx_, mul2, sig, 2); // [8400,1,84,1] == NCHW [1,84,8400]

            if (is_yolo_seg_) {
                // Mask coefficients: cv4 branch per scale (C->32, 32->32, 1x1
                // 32->32, no activation), concatenated along the flat spatial
                // dim and appended after box(4)+class(80) -> 116 channels.
                struct ggml_tensor* mask_parts[3];
                for (int s = 0; s < 3; ++s) {
                    const std::string pm = "model.22.cv4." + std::to_string(s);
                    struct ggml_tensor* hm = yact(feats[s], ykernel(pm + ".0.conv", 32, feats[s]->ne[2], 3, 3),
                                                  ybias(pm + ".0.conv", 32), 1, 1);
                    hm = yact(hm, ykernel(pm + ".1.conv", 32, 32, 3, 3), ybias(pm + ".1.conv", 32), 1, 1);
                    hm = yconv(hm, ykernel(pm + ".2", 32, 32, 1, 1), ybias(pm + ".2", 32), 1, 0);
                    mask_parts[s] = ggml_reshape_4d(ctx_, hm, hm->ne[0] * hm->ne[1], 1, 32, 1);
                }
                struct ggml_tensor* mask_coeff =
                    ggml_concat(ctx_, ggml_concat(ctx_, mask_parts[0], mask_parts[1], 0), mask_parts[2], 0);
                logits = ggml_concat(ctx_, logits, mask_coeff, 2); // [8400,1,116,1]

                // Prototype masks: P3 -> cv1(SiLU) -> 2x up ConvTranspose ->
                // cv2(SiLU) -> cv3(SiLU). The ONNX ConvTranspose weight is
                // [Cin,Cout,kH,kW]; ggml's p0 kernel takes [kW,kH,Cout,Cin]
                // and requires F16 weights on the CPU backend.
                const LoadedTensor& up_w =
                    require_tensor(tensors, "model.22.proto.upsample.weight", model_path, shapes_seen);
                require_dims(up_w, "model.22.proto.upsample.weight", model_path, {64, 64, 2, 2}, shapes_seen);
                // ONNX [Cin,Cout,kH,kW] row-major and ggml [kW,kH,Cout,Cin]
                // column-major have the same flat order for this 64/64/2/2
                // kernel (the same reversed-dims coincidence used by the conv
                // weights above), so the copy is intentional, not a no-op bug.
                struct ggml_tensor* up_weight = new_const_4d_f16(2, 2, 64, 64, up_w.data);
                struct ggml_tensor* proto = yact(feats[0], ykernel("model.22.proto.cv1.conv", 64, 64, 3, 3),
                                                 ybias("model.22.proto.cv1.conv", 64), 1, 1);
                proto = ggml_conv_transpose_2d_p0(ctx_, up_weight, proto, 2);
                proto = ggml_add(ctx_, proto, ggml_repeat(ctx_, ybias("model.22.proto.upsample", 64), proto));
                proto = yact(proto, ykernel("model.22.proto.cv2.conv", 64, 64, 3, 3),
                             ybias("model.22.proto.cv2.conv", 64), 1, 1);
                seg_proto = yact(proto, ykernel("model.22.proto.cv3.conv", 32, 64, 1, 1),
                                 ybias("model.22.proto.cv3.conv", 32), 1, 0); // [160,160,32,1]
            }
        } // end else (YOLOv8n path)
    } else {
        // RF-DETR model variant: detection (560px input, 11 backbone blocks,
        // 300 queries, 3 decoder layers) vs segmentation (432px, 12 blocks,
        // 200 queries, 4 decoder layers). is_rfdetr_seg is set at dispatch.
        const bool seg = is_rfdetr_seg;
        const int RC_NB = seg ? 12 : 11;
        const int RC_T = seg ? 325 : 101;
        const int RC_W = seg ? 4 : 16;
        const int RC_WN = seg ? 2 : 4;
        const int RC_TOK = seg ? 324 : 100;
        const int RC_WS = seg ? 18 : 10;
        const int RC_MS = seg ? 36 : 40;
        const int RC_NT = seg ? 1296 : 1600;
        const int RC_TOK1 = seg ? 1297 : 1601;
        const int RC_NQ = seg ? 200 : 300;
        const int RC_NL = seg ? 4 : 3;
        const int RC_PATCH = seg ? 12 : 14;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): tap layers
        const int RC_TAPS[4] = {seg ? 2 : 1, seg ? 5 : 4, seg ? 8 : 7, seg ? 11 : 10};
        const char* const* RC_WQ = seg ? kSegWQ : kRfdetrWQ;
        const char* const* RC_WK = seg ? kSegWK : kRfdetrWK;
        const char* const* RC_WV = seg ? kSegWV : kRfdetrWV;
        const char* const* RC_WO = seg ? kSegWO : kRfdetrWO;
        const char* const* RC_WF1 = seg ? kSegWF1 : kRfdetrWF1;
        const char* const* RC_WF2 = seg ? kSegWF2 : kRfdetrWF2;

        const int RC_G0 = seg ? 3 : 2;
        const int RC_G1 = seg ? 6 : 5;
        const int RC_G2 = seg ? 9 : 8;
        // ---- RF-DETR S-1: backbone (DINOv2-S ViT) + projector ----
        // Intermediate contract: the loader exposes the projector map
        // [1,256,40,40] (ggml [40,40,256,1]); the full detection graph lands
        // in a later slice and moves this sentinel. Layout coincidence (D-8):
        // NCHW row-major bytes equal ggml [W,H,C,N] flat order, so 4D consts
        // copy verbatim; 2D ONNX MatMul weights [K,N] are transposed into
        // ggml ne=[K,N] (same recipe the S-1 probe uses).
        auto rweight_2d = [&](const std::string& name, uint32_t rows, uint32_t cols) -> struct ggml_tensor* {
            const LoadedTensor& w = require_tensor(tensors, name, model_path, shapes_seen);
            require_dims(w, name, model_path, {rows, cols}, shapes_seen);
            std::vector<float> tw(static_cast<size_t>(rows) * cols);
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t c = 0; c < cols; ++c) {
                    tw[static_cast<size_t>(c) * rows + r] = w.data[static_cast<size_t>(r) * cols + c];
                }
            }
            return new_const_4d(rows, cols, 1, 1, std::move(tw));
        };
        auto rlinear = [&](struct ggml_tensor* xx, const std::string& wname,
                           const std::string& bname) -> struct ggml_tensor* {
            const LoadedTensor& w = require_tensor(tensors, wname, model_path, shapes_seen);
            if (w.dims.size() != 2 || w.dims[0] != static_cast<uint32_t>(xx->ne[0])) {
                fail_model(model_path, "RF-DETR linear weight '" + wname +
                                           "' needs [K,N] with K==input ne[0]; tensors:" + shapes_seen);
            }
            struct ggml_tensor* y = ggml_mul_mat(ctx_, rweight_2d(wname, w.dims[0], w.dims[1]), xx);
            const LoadedTensor& b = require_tensor(tensors, bname, model_path, shapes_seen);
            require_dims(b, bname, model_path, {w.dims[1]}, shapes_seen);
            struct ggml_tensor* bb = new_const_4d(w.dims[1], 1, 1, 1, b.data);
            return ggml_add(ctx_, y, ggml_repeat(ctx_, bb, y));
        };
        // Gemm(transB=1) linear: ONNX computes Y = X @ W^T + b (seg decoder
        // self-attn out_proj). rweight_2d stores transpose(W) bytes, which is
        // exactly row-major(W^T); feeding W^T through rweight_2d would give
        // transpose(W^T) = verbatim W bytes. So the correct ggml constant here
        // is the verbatim initializer bytes (no transpose).
        auto rlinear_gemm_tb = [&](struct ggml_tensor* xx, const std::string& wname,
                                   const std::string& bname) -> struct ggml_tensor* {
            const LoadedTensor& w = require_tensor(tensors, wname, model_path, shapes_seen);
            if (w.dims.size() != 2 || w.dims[0] != static_cast<uint32_t>(xx->ne[0])) {
                fail_model(model_path, "RF-DETR Gemm(transB) weight '" + wname +
                                           "' needs [K,N] with K==input ne[0]; tensors:" + shapes_seen);
            }
            struct ggml_tensor* y = ggml_mul_mat(ctx_, new_const_4d(w.dims[0], w.dims[1], 1, 1, w.data), xx);
            const LoadedTensor& b = require_tensor(tensors, bname, model_path, shapes_seen);
            require_dims(b, bname, model_path, {w.dims[1]}, shapes_seen);
            struct ggml_tensor* bb = new_const_4d(w.dims[1], 1, 1, 1, b.data);
            return ggml_add(ctx_, y, ggml_repeat(ctx_, bb, y));
        };
        auto rln = [&](struct ggml_tensor* xx, const std::string& pre, float eps) -> struct ggml_tensor* {
            const int64_t d = xx->ne[0];
            struct ggml_tensor* n = ggml_norm(ctx_, xx, eps);
            const LoadedTensor& gw = require_tensor(tensors, pre + ".weight", model_path, shapes_seen);
            require_dims(gw, pre + ".weight", model_path, {static_cast<uint32_t>(d)}, shapes_seen);
            const LoadedTensor& gb = require_tensor(tensors, pre + ".bias", model_path, shapes_seen);
            require_dims(gb, pre + ".bias", model_path, {static_cast<uint32_t>(d)}, shapes_seen);
            struct ggml_tensor* w = new_const_4d(d, 1, 1, 1, gw.data);
            struct ggml_tensor* b = new_const_4d(d, 1, 1, 1, gb.data);
            return ggml_add(ctx_, ggml_mul(ctx_, n, ggml_repeat(ctx_, w, n)), ggml_repeat(ctx_, b, n));
        };
        // LayerNorm over the channel dim of a conv map [W,H,C,N].
        auto rchan_ln = [&](struct ggml_tensor* xx, const std::string& mul, const std::string& add,
                            float eps) -> struct ggml_tensor* {
            const int64_t c = xx->ne[2];
            struct ggml_tensor* p = ggml_cont(ctx_, ggml_permute(ctx_, xx, 1, 2, 0, 3));
            struct ggml_tensor* n = ggml_norm(ctx_, p, eps);
            const LoadedTensor& gm = require_tensor(tensors, mul, model_path, shapes_seen);
            // Affine pairs are [C,1,1] in the detection export and rank-1 [C]
            // in the seg export; both are flat [C] in memory.
            if (gm.dims != std::vector<uint32_t>{static_cast<uint32_t>(c), 1, 1} &&
                gm.dims != std::vector<uint32_t>{static_cast<uint32_t>(c)}) {
                fail_model(model_path, "channel-LN weight '" + mul + "' needs [C,1,1] or [C]; tensors:" + shapes_seen);
            }
            const LoadedTensor& ga = require_tensor(tensors, add, model_path, shapes_seen);
            if (ga.dims != std::vector<uint32_t>{static_cast<uint32_t>(c), 1, 1} &&
                ga.dims != std::vector<uint32_t>{static_cast<uint32_t>(c)}) {
                fail_model(model_path, "channel-LN bias '" + add + "' needs [C,1,1] or [C]; tensors:" + shapes_seen);
            }
            struct ggml_tensor* w = new_const_4d(c, 1, 1, 1, gm.data);
            struct ggml_tensor* b = new_const_4d(c, 1, 1, 1, ga.data);
            struct ggml_tensor* y = ggml_add(ctx_, ggml_mul(ctx_, n, ggml_repeat(ctx_, w, n)), ggml_repeat(ctx_, b, n));
            return ggml_cont(ctx_, ggml_permute(ctx_, y, 2, 0, 1, 3));
        };
        auto rconv = [&](struct ggml_tensor* xx, const std::string& wname, uint32_t O, uint32_t I, int K, int s,
                         int p) -> struct ggml_tensor* {
            const LoadedTensor& w = require_tensor(tensors, wname, model_path, shapes_seen);
            require_dims(w, wname, model_path, {O, I, static_cast<uint32_t>(K), static_cast<uint32_t>(K)}, shapes_seen);
            struct ggml_tensor* k = new_const_4d(K, K, I, O, w.data);
            return ggml_conv_2d(ctx_, k, xx, s, s, p, p, 1, 1);
        };
        auto rsilu = [&](struct ggml_tensor* xx) -> struct ggml_tensor* {
            return ggml_mul(ctx_, xx, ggml_sigmoid(ctx_, xx));
        };
        // Projector convs carry no bias (their "bn" is the channel LayerNorm
        // below); the stem bias is added by the caller.
        auto rbn_conv = [&](struct ggml_tensor* xx, const std::string& wname, const std::string& mul,
                            const std::string& add, uint32_t O, uint32_t I, int K, int s,
                            int p) -> struct ggml_tensor* {
            struct ggml_tensor* h = rconv(xx, wname, O, I, K, s, p);
            h = rchan_ln(h, mul, add, 1e-6f);
            return rsilu(h);
        };
        // [384,T,W] windowed tokens -> [MS,MS,384,1] spatial map (drop cls).
        auto rblock_to_map = [&](struct ggml_tensor* t) -> struct ggml_tensor* {
            std::vector<struct ggml_tensor*> rows;
            for (int yw = 0; yw < RC_WN; ++yw) {
                std::vector<struct ggml_tensor*> cols;
                for (int xw = 0; xw < RC_WN; ++xw) {
                    const int w = yw * RC_WN + xw;
                    const size_t off = (static_cast<size_t>(384) * RC_T * w + 384) * sizeof(float);
                    struct ggml_tensor* tok = ggml_view_2d(ctx_, t, 384, RC_TOK, 384 * sizeof(float), off);
                    tok = ggml_cont(ctx_, tok);
                    struct ggml_tensor* map = ggml_reshape_3d(ctx_, tok, 384, RC_WS, RC_WS);
                    map = ggml_cont(ctx_, ggml_permute(ctx_, map, 2, 0, 1, 3));
                    cols.push_back(map);
                }
                struct ggml_tensor* row = cols[0];
                for (int xw = 1; xw < RC_WN; ++xw) {
                    row = ggml_concat(ctx_, row, cols[static_cast<size_t>(xw)], 0);
                }
                rows.push_back(ggml_cont(ctx_, row));
            }
            struct ggml_tensor* out = rows[0];
            for (int yw = 1; yw < RC_WN; ++yw) {
                out = ggml_concat(ctx_, out, rows[static_cast<size_t>(yw)], 1);
            }
            return ggml_cont(ctx_, out);
        };
        // Identity matrix for the V^T eye trick (avoids transpose+cont,
        // broken for non-square in this ggml version).
        std::vector<float> eye(384 * 384, 0.0f);
        for (int i = 0; i < 384; ++i) {
            eye[static_cast<size_t>(i) * 384 + i] = 1.0f;
        }
        struct ggml_tensor* eye384 = new_const_4d(384, 384, 1, 1, std::move(eye));
        auto rattn = [&](struct ggml_tensor* xx, int b) -> struct ggml_tensor* {
            constexpr int D = 384, H = 6, Dh = 64;
            const int T = RC_T, W = RC_W;
            const std::string P = "backbone.0.encoder.encoder.encoder.layer." + std::to_string(b) + ".";
            struct ggml_tensor* h = rln(xx, P + "norm1", 1e-6f);
            struct ggml_tensor* q = rlinear(h, RC_WQ[b], P + "attention.attention.query.bias");
            struct ggml_tensor* k = rlinear(h, RC_WK[b], P + "attention.attention.key.bias");
            struct ggml_tensor* v = rlinear(h, RC_WV[b], P + "attention.attention.value.bias");
            const bool global_attn = (b == RC_G0 || b == RC_G1 || b == RC_G2);
            struct ggml_tensor* actx = nullptr;
            if (global_attn) {
                const int TG = T * W;
                struct ggml_tensor* qf = ggml_reshape_3d(ctx_, q, D, TG, 1);
                struct ggml_tensor* kf = ggml_reshape_3d(ctx_, k, D, TG, 1);
                struct ggml_tensor* vf = ggml_reshape_3d(ctx_, v, D, TG, 1);
                struct ggml_tensor* vflat = ggml_reshape_2d(ctx_, vf, D, TG);
                struct ggml_tensor* vt = ggml_mul_mat(ctx_, vflat, eye384); // [TG,D]
                std::vector<struct ggml_tensor*> heads;
                for (int hh = 0; hh < H; ++hh) {
                    struct ggml_tensor* qh =
                        ggml_view_2d(ctx_, qf, Dh, TG, D * sizeof(float), static_cast<size_t>(hh) * Dh * sizeof(float));
                    struct ggml_tensor* kh =
                        ggml_view_2d(ctx_, kf, Dh, TG, D * sizeof(float), static_cast<size_t>(hh) * Dh * sizeof(float));
                    struct ggml_tensor* s = ggml_scale(ctx_, ggml_mul_mat(ctx_, kh, qh), 0.125f);
                    struct ggml_tensor* pr = ggml_soft_max(ctx_, s);
                    struct ggml_tensor* vht = ggml_view_2d(ctx_, vt, TG, Dh, static_cast<size_t>(TG) * sizeof(float),
                                                           static_cast<size_t>(hh) * Dh * TG * sizeof(float));
                    heads.push_back(ggml_mul_mat(ctx_, vht, pr));
                }
                struct ggml_tensor* c = heads[0];
                for (int hh = 1; hh < H; ++hh) {
                    c = ggml_concat(ctx_, c, heads[hh], 0);
                }
                c = ggml_cont(ctx_, c);
                actx = ggml_reshape_3d(ctx_, c, D, T, W);
            } else {
                struct ggml_tensor* vflat = ggml_reshape_2d(ctx_, v, 384, T * W);
                struct ggml_tensor* vt = ggml_mul_mat(ctx_, vflat, eye384);
                std::vector<struct ggml_tensor*> win_out;
                for (int w = 0; w < W; ++w) {
                    std::vector<struct ggml_tensor*> heads;
                    for (int hh = 0; hh < H; ++hh) {
                        const size_t off =
                            (static_cast<size_t>(w) * D * T + static_cast<size_t>(hh) * Dh) * sizeof(float);
                        struct ggml_tensor* qh = ggml_view_2d(ctx_, q, Dh, T, D * sizeof(float), off);
                        struct ggml_tensor* kh = ggml_view_2d(ctx_, k, Dh, T, D * sizeof(float), off);
                        struct ggml_tensor* s = ggml_scale(ctx_, ggml_mul_mat(ctx_, kh, qh), 0.125f);
                        struct ggml_tensor* pr = ggml_soft_max(ctx_, s);
                        const size_t off2 =
                            (static_cast<size_t>(w) * T + static_cast<size_t>(hh) * Dh * (T * W)) * sizeof(float);
                        struct ggml_tensor* vht =
                            ggml_view_2d(ctx_, vt, T, Dh, static_cast<size_t>(T * W) * sizeof(float), off2);
                        struct ggml_tensor* ct = ggml_mul_mat(ctx_, vht, pr);
                        heads.push_back(ct);
                    }
                    struct ggml_tensor* c = heads[0];
                    for (int hh = 1; hh < H; ++hh) {
                        c = ggml_concat(ctx_, c, heads[hh], 0);
                    }
                    c = ggml_cont(ctx_, c);
                    win_out.push_back(ggml_reshape_4d(ctx_, c, D, T, 1, 1));
                }
                actx = win_out[0];
                for (int w = 1; w < W; ++w) {
                    actx = ggml_concat(ctx_, actx, win_out[w], 2);
                }
                actx = ggml_cont(ctx_, actx);
            }
            struct ggml_tensor* o = rlinear(actx, RC_WO[b], P + "attention.output.dense.bias");
            const LoadedTensor& gl1t = require_tensor(tensors, P + "layer_scale1.lambda1", model_path, shapes_seen);
            require_dims(gl1t, P + "layer_scale1.lambda1", model_path, {384}, shapes_seen);
            struct ggml_tensor* gl1 = new_const_4d(D, 1, 1, 1, gl1t.data);
            o = ggml_add(ctx_, xx, ggml_mul(ctx_, o, ggml_repeat(ctx_, gl1, o)));
            struct ggml_tensor* h2 = rln(o, P + "norm2", 1e-6f);
            struct ggml_tensor* f1 = rlinear(h2, RC_WF1[b], P + "mlp.fc1.bias");
            f1 = ggml_gelu_erf(ctx_, f1);
            struct ggml_tensor* f2 = rlinear(f1, RC_WF2[b], P + "mlp.fc2.bias");
            const LoadedTensor& gl2t = require_tensor(tensors, P + "layer_scale2.lambda1", model_path, shapes_seen);
            require_dims(gl2t, P + "layer_scale2.lambda1", model_path, {384}, shapes_seen);
            struct ggml_tensor* gl2 = new_const_4d(D, 1, 1, 1, gl2t.data);
            return ggml_add(ctx_, o, ggml_mul(ctx_, f2, ggml_repeat(ctx_, gl2, f2)));
        };
        auto rprojector = [&](struct ggml_tensor* f1, struct ggml_tensor* f4, struct ggml_tensor* f7,
                              struct ggml_tensor* f10) -> struct ggml_tensor* {
            const std::string P = "backbone.0.projector.stages.0.0.";
            const std::string cv1m = seg ? P + "cv1.bn.weight" : "onnx::Mul_4092";
            const std::string cv1a = seg ? P + "cv1.bn.bias" : "onnx::Add_4094";
            const std::string cv2m = seg ? P + "cv2.bn.weight" : "onnx::Mul_4120";
            const std::string cv2a = seg ? P + "cv2.bn.bias" : "onnx::Add_4122";
            const std::string finm = seg ? "backbone.0.projector.stages.0.1.weight" : "onnx::Mul_4124";
            const std::string fina = seg ? "backbone.0.projector.stages.0.1.bias" : "onnx::Add_4126";
            const char* mm1[3] = {nullptr, nullptr, nullptr};
            const char* aa1[3] = {nullptr, nullptr, nullptr};
            const char* mm2[3] = {nullptr, nullptr, nullptr};
            const char* aa2[3] = {nullptr, nullptr, nullptr};
            static const char* det_mm1[3] = {"onnx::Mul_4096", "onnx::Mul_4104", "onnx::Mul_4112"};
            static const char* det_aa1[3] = {"onnx::Add_4098", "onnx::Add_4106", "onnx::Add_4114"};
            static const char* det_mm2[3] = {"onnx::Mul_4100", "onnx::Mul_4108", "onnx::Mul_4116"};
            static const char* det_aa2[3] = {"onnx::Add_4102", "onnx::Add_4110", "onnx::Add_4118"};
            std::string seg_mm1[3], seg_aa1[3], seg_mm2[3], seg_aa2[3];
            for (int i = 0; i < 3; ++i) {
                const std::string M = P + "m." + std::to_string(i) + ".";
                seg_mm1[i] = M + "cv1.bn.weight";
                seg_aa1[i] = M + "cv1.bn.bias";
                seg_mm2[i] = M + "cv2.bn.weight";
                seg_aa2[i] = M + "cv2.bn.bias";
                mm1[i] = seg ? seg_mm1[i].c_str() : det_mm1[i];
                aa1[i] = seg ? seg_aa1[i].c_str() : det_aa1[i];
                mm2[i] = seg ? seg_mm2[i].c_str() : det_mm2[i];
                aa2[i] = seg ? seg_aa2[i].c_str() : det_aa2[i];
            }
            struct ggml_tensor* cat = ggml_concat(ctx_, f1, f4, 2);
            cat = ggml_concat(ctx_, cat, f7, 2);
            cat = ggml_cont(ctx_, ggml_concat(ctx_, cat, f10, 2));
            struct ggml_tensor* h = rbn_conv(cat, P + "cv1.conv.weight", cv1m, cv1a, 256, 1536, 1, 1, 0);
            h = ggml_cont(ctx_, h);
            const size_t bnb1 = static_cast<size_t>(RC_MS) * sizeof(float);
            const size_t bnb2 = bnb1 * static_cast<size_t>(RC_MS);
            const size_t bnb3 = bnb2 * 256;
            struct ggml_tensor* a = ggml_cont(ctx_, ggml_view_4d(ctx_, h, RC_MS, RC_MS, 128, 1, bnb1, bnb2, bnb3, 0));
            struct ggml_tensor* b =
                ggml_cont(ctx_, ggml_view_4d(ctx_, h, RC_MS, RC_MS, 128, 1, bnb1, bnb2, bnb3, bnb2 * 128));
            std::vector<struct ggml_tensor*> parts = {a, b};
            struct ggml_tensor* cur = b;
            for (int i = 0; i < 3; ++i) {
                const std::string M = P + "m." + std::to_string(i) + ".";
                cur = rbn_conv(cur, M + "cv1.conv.weight", mm1[i], aa1[i], 128, 128, 3, 1, 1);
                cur = rbn_conv(cur, M + "cv2.conv.weight", mm2[i], aa2[i], 128, 128, 3, 1, 1);
                parts.push_back(cur);
            }
            struct ggml_tensor* merged = parts[0];
            for (size_t i = 1; i < parts.size(); ++i) {
                merged = ggml_concat(ctx_, merged, parts[i], 2);
            }
            merged = ggml_cont(ctx_, merged);
            struct ggml_tensor* out = rbn_conv(merged, P + "cv2.conv.weight", cv2m, cv2a, 256, 640, 1, 1, 0);
            return rchan_ln(out, finm, fina, 1e-6f);
        };
        // Stem: 14x14/s14 conv + bias (no activation; cls/pos join below).
        const LoadedTensor& stem_wt =
            require_tensor(tensors, "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight",
                           model_path, shapes_seen);
        require_dims(stem_wt, "patch projection weight", model_path,
                     {384, 3, static_cast<uint32_t>(RC_PATCH), static_cast<uint32_t>(RC_PATCH)}, shapes_seen);
        struct ggml_tensor* stem = ggml_conv_2d(ctx_, new_const_4d(RC_PATCH, RC_PATCH, 3, 384, stem_wt.data), x,
                                                RC_PATCH, RC_PATCH, 0, 0, 1, 1);
        const LoadedTensor& stem_bt = require_tensor(
            tensors, "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.bias", model_path, shapes_seen);
        require_dims(stem_bt, "patch projection bias", model_path, {384}, shapes_seen);
        struct ggml_tensor* stem_b = new_const_4d(1, 1, 384, 1, stem_bt.data);
        stem = ggml_add(ctx_, stem, ggml_repeat(ctx_, stem_b, stem));
        // Tokens [cls; 1600 patches] + position embeddings, packed into 16
        // windows of 101 via a static index vector (get_rows on dim 0).
        struct ggml_tensor* stem_t = ggml_reshape_4d(ctx_, stem, RC_NT, 384, 1, 1);
        const LoadedTensor& cls_t =
            require_tensor(tensors, "backbone.0.encoder.encoder.embeddings.cls_token", model_path, shapes_seen);
        require_dims(cls_t, "cls_token", model_path, {1, 1, 384}, shapes_seen);
        struct ggml_tensor* cls = new_const_4d(1, 384, 1, 1, cls_t.data);
        struct ggml_tensor* tok = ggml_concat(ctx_, cls, stem_t, 0);                      // [1601,384]
        struct ggml_tensor* tok_t = ggml_cont(ctx_, ggml_permute(ctx_, tok, 1, 0, 2, 3)); // [384,1601]
        const LoadedTensor& pos_t = require_tensor(tensors, "backbone.0.encoder.encoder.embeddings.position_embeddings",
                                                   model_path, shapes_seen);
        require_dims(pos_t, "position_embeddings", model_path, {1, static_cast<uint32_t>(RC_TOK1), 384}, shapes_seen);
        struct ggml_tensor* pos = new_const_4d(384, RC_TOK1, 1, 1, pos_t.data);
        struct ggml_tensor* toks = ggml_add(ctx_, tok_t, pos); // [384,1601]
        std::vector<int32_t> widx(static_cast<size_t>(RC_W) * RC_T);
        for (int w = 0; w < RC_W; ++w) {
            const int wx = w % RC_WN, wy = w / RC_WN;
            widx[static_cast<size_t>(w) * RC_T] = 0;
            for (int pp = 1; pp <= RC_TOK; ++pp) {
                const int dx = (pp - 1) % RC_WS, dy = (pp - 1) / RC_WS;
                widx[static_cast<size_t>(w) * RC_T + pp] = 1 + (wy * RC_WS + dy) * RC_MS + (wx * RC_WS + dx);
            }
        }
        struct ggml_tensor* widx_t = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, RC_W * RC_T);
        pending_i32.push_back(PendingFillI32{widx_t, std::move(widx)});
        // get_rows gathers dim-1 rows: out[c,e] = toks[c, widx[e]].
        struct ggml_tensor* gathered = ggml_get_rows(ctx_, toks, widx_t); // [384,1616]
        struct ggml_tensor* xt = ggml_reshape_4d(ctx_, gathered, 384, RC_T, RC_W, 1);
        struct ggml_tensor* b0 = xt;
        std::vector<struct ggml_tensor*> feats(static_cast<size_t>(RC_NB), nullptr);
        for (int b = 0; b < RC_NB; ++b) {
            b0 = rattn(b0, b);
            feats[static_cast<size_t>(b)] = b0;
        }
        const std::string rln_pre = "backbone.0.encoder.encoder.layernorm";
        auto rfeat_map = [&](int b) -> struct ggml_tensor* {
            return rblock_to_map(rln(feats[static_cast<size_t>(b)], rln_pre, 1e-6f));
        };
        logits = rprojector(rfeat_map(RC_TAPS[0]), rfeat_map(RC_TAPS[1]), rfeat_map(RC_TAPS[2]), rfeat_map(RC_TAPS[3]));
        // ---- Slice 2a: neck encoder tower + top-300 query selection ----
        // Tokens [256,1600] from the projector map; encoder Linear+LayerNorm;
        // class/bbox heads; default-box decode; top-k rows via
        // ggml_ops::top_k_indices (never argsort); dim-1 get_rows gather.
        // Exposed as the second output [1,300,4] (ggml [4,300,1,1]).
        struct ggml_tensor* enc_tok = ggml_cont(ctx_, ggml_permute(ctx_, logits, 1, 2, 0, 3)); // [256,40,40]
        const char* RC_ENC0 = seg ? "onnx::MatMul_4704" : "onnx::MatMul_4155";
        const char* RC_ENC1 = seg ? "onnx::MatMul_4705" : "onnx::MatMul_4156";
        const char* RC_ENC2 = seg ? "onnx::MatMul_4706" : "onnx::MatMul_4157";
        const char* RC_ENC3 = seg ? "onnx::MatMul_4707" : "onnx::MatMul_4158";
        const char* RC_ENC4 = seg ? "onnx::MatMul_4708" : "onnx::MatMul_4159";
        enc_tok = ggml_reshape_3d(ctx_, enc_tok, 256, RC_NT, 1);
        struct ggml_tensor* enc = rlinear(enc_tok, RC_ENC0, "transformer.enc_output.0.bias");
        enc = rln(enc, "transformer.enc_output_norm.0", 1e-5f);
        struct ggml_tensor* enc_class = rlinear(enc, RC_ENC1, "transformer.enc_out_class_embed.0.bias");
        struct ggml_tensor* eb = rlinear(enc, RC_ENC2, "transformer.enc_out_bbox_embed.0.layers.0.bias");
        eb = ggml_relu(ctx_, eb);
        eb = rlinear(eb, RC_ENC3, "transformer.enc_out_bbox_embed.0.layers.1.bias");
        eb = ggml_relu(ctx_, eb);
        eb = rlinear(eb, RC_ENC4, "transformer.enc_out_bbox_embed.0.layers.2.bias");
        // Default boxes [cx,cy,0.05,0.05] on the model grid, host-computed.
        std::vector<float> def(static_cast<size_t>(4) * RC_NT);
        for (int y = 0; y < RC_MS; ++y) {
            for (int xx = 0; xx < RC_MS; ++xx) {
                const size_t i = static_cast<size_t>(xx) + static_cast<size_t>(RC_MS) * y;
                def[0 + 4 * i] = (xx + 0.5f) / static_cast<float>(RC_MS);
                def[1 + 4 * i] = (y + 0.5f) / static_cast<float>(RC_MS);
                def[2 + 4 * i] = 0.05f;
                def[3 + 4 * i] = 0.05f;
            }
        }
        struct ggml_tensor* defb = new_const_4d(4, RC_NT, 1, 1, std::move(def));
        const size_t bb4 = 4 * sizeof(float);
        struct ggml_tensor* bxy = ggml_view_2d(ctx_, eb, 2, RC_NT, bb4, 0);
        struct ggml_tensor* bwh = ggml_view_2d(ctx_, eb, 2, RC_NT, bb4, 2 * sizeof(float));
        struct ggml_tensor* dxy = ggml_view_2d(ctx_, defb, 2, RC_NT, bb4, 0);
        struct ggml_tensor* dwh = ggml_view_2d(ctx_, defb, 2, RC_NT, bb4, 2 * sizeof(float));
        struct ggml_tensor* exy = ggml_add(ctx_, ggml_mul(ctx_, bxy, dwh), dxy);
        struct ggml_tensor* ewh = ggml_mul(ctx_, ggml_exp(ctx_, bwh), dwh);
        struct ggml_tensor* enc_boxes = ggml_concat(ctx_, exy, ewh, 0);
        struct ggml_tensor* escores = ggml_pool_1d(ctx_, enc_class, GGML_OP_POOL_MAX, 91, 91, 0);
        escores = ggml_reshape_2d(ctx_, escores, RC_NT, 1);
        struct ggml_tensor* etopk = ggml_ops::top_k_indices(ctx_, escores, RC_NQ);
        // Test-only TopK order override (decoder parity tests): F32 score
        // drift flips 82 near-tie pairs vs ORT, which changes slot pairing
        // downstream (fixed per-slot queries). The layer-0 parity test pins
        // ORT's exact order through NEURIPLO_TEST_RFDETR_TOPK_ORDER so the
        // wiring itself is verified; the natural order is covered by the
        // encoder selection test. Active only when explicitly requested.
        if (const char* order_path = std::getenv("NEURIPLO_TEST_RFDETR_TOPK_ORDER")) {
            LOG(WARNING) << "RF-DETR using test TopK order override from " << order_path;
            std::ifstream order_file(order_path);
            std::vector<int32_t> oidx(static_cast<size_t>(RC_NQ), 0);
            for (size_t i = 0; i < static_cast<size_t>(RC_NQ) && order_file; ++i) {
                int v = 0;
                order_file >> v;
                oidx[i] = static_cast<int32_t>(v);
            }
            if (order_file.bad()) {
                fail_model(model_path, "cannot read test TopK order file");
            }
            struct ggml_tensor* ot = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, RC_NQ);
            pending_i32.push_back(PendingFillI32{ot, std::move(oidx)});
            etopk = ot;
        }
        aux_out = ggml_get_rows(ctx_, enc_boxes, etopk);
        // ---- Slice 2b: decoder layer 0 (self-attn + deformable
        // cross-attn + FFN), exposed as third output [1,300,256].
        // Tile_1 and the reference points are GGML constants; the refined
        // anchors come from the selected boxes above.
        // ONNX row-major [..,rows,cols] (leading 1s allowed) -> ggml
        // ne=[cols,rows,1,1]. Needed for rank>2 constants whose last two
        // dims are not interchangeable (e.g. the learned reference points:
        // verbatim copy would transpose them).
        auto rconst_t = [&](const std::string& name, uint32_t rows, uint32_t cols) -> struct ggml_tensor* {
            const LoadedTensor& w = require_tensor(tensors, name, model_path, shapes_seen);
            if (!(w.dims == std::vector<uint32_t>{rows, cols} || w.dims == std::vector<uint32_t>{1, rows, cols})) {
                fail_model(model_path, "RF-DETR constant '" + name +
                                           "' needs [rows,cols] or [1,rows,cols]; tensors:" + shapes_seen);
            }
            std::vector<float> tw(static_cast<size_t>(cols) * rows);
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t c = 0; c < cols; ++c) {
                    tw[static_cast<size_t>(c) + static_cast<size_t>(cols) * r] =
                        w.data[static_cast<size_t>(r) * cols + c];
                }
            }
            return new_const_4d(cols, rows, 1, 1, std::move(tw));
        };
        const char* RC_TILE = seg ? "onnx::Expand_4727" : "onnx::Expand_4178";
        const char* RC_REFP = seg ? "onnx::Expand_4745" : "onnx::Expand_4196";
        struct ggml_tensor* tile1 = rconst_t(RC_TILE, RC_NQ, 256);
        struct ggml_tensor* refpts = rconst_t(RC_REFP, RC_NQ, 4);
        struct ggml_tensor* sel_xy = ggml_view_2d(ctx_, aux_out, 2, RC_NQ, 16, 0);
        struct ggml_tensor* sel_wh = ggml_view_2d(ctx_, aux_out, 2, RC_NQ, 16, 8);
        struct ggml_tensor* ref_xy = ggml_view_2d(ctx_, refpts, 2, RC_NQ, 16, 0);
        struct ggml_tensor* ref_wh = ggml_view_2d(ctx_, refpts, 2, RC_NQ, 16, 8);
        struct ggml_tensor* d_add4 = ggml_add(ctx_, sel_xy, ggml_mul(ctx_, ref_xy, sel_wh));
        struct ggml_tensor* d_mul8 = ggml_mul(ctx_, ggml_exp(ctx_, ref_wh), sel_wh);
        struct ggml_tensor* anchors = ggml_concat(ctx_, d_add4, d_mul8, 0);  // [4,300]
        struct ggml_tensor* c8 = ggml_ops::rfdetr_sine_embed(ctx_, anchors); // [512,300]
        const char* RC_RP0 = seg ? "onnx::MatMul_4805" : "onnx::MatMul_4250";
        const char* RC_RP1 = seg ? "onnx::MatMul_4806" : "onnx::MatMul_4251";
        struct ggml_tensor* rp0 = rlinear(c8, RC_RP0, "transformer.decoder.ref_point_head.layers.0.bias");
        rp0 = ggml_relu(ctx_, rp0);
        struct ggml_tensor* refph = rlinear(rp0, RC_RP1, "transformer.decoder.ref_point_head.layers.1.bias");
        // ---- Slice 2b: decoder layers (self-attn + deformable
        // cross-attn + FFN each). Per-layer table: layer 0 uses Tile_1 as
        // the self-attn value/residual base, later layers use the previous
        // layer's norm3 (DETR convention: q=k=query, v=prev). Detection runs
        // 3 layers over 300 queries; seg runs 4 layers over 200 queries.
        const RfdetrDecLayer* RC_D = seg ? kSegDec : kDetDec;
        struct ggml_tensor* dprev = tile1;
        struct ggml_tensor* dec_norm3[4] = {nullptr, nullptr, nullptr, nullptr};
        for (int dl = 0; dl < RC_NL; ++dl) {
            const std::string dlp = "transformer.decoder.layers." + std::to_string(dl) + ".";
            struct ggml_tensor* d_a0 = ggml_add(ctx_, dprev, refph);
            // Self-attention: 8 heads x 32, q=k=A0, v=prev.
            struct ggml_tensor* dq = rlinear(d_a0, RC_D[dl].q, RC_D[dl].qb);
            struct ggml_tensor* dk = rlinear(d_a0, RC_D[dl].k, RC_D[dl].kb);
            struct ggml_tensor* dv = rlinear(dprev, RC_D[dl].v, RC_D[dl].vb);
            std::vector<struct ggml_tensor*> dheads;
            for (int hh = 0; hh < 8; ++hh) {
                struct ggml_tensor* dqh = ggml_view_2d(ctx_, dq, 32, RC_NQ, 256 * sizeof(float),
                                                       static_cast<size_t>(hh) * 32 * sizeof(float));
                struct ggml_tensor* dkh = ggml_view_2d(ctx_, dk, 32, RC_NQ, 256 * sizeof(float),
                                                       static_cast<size_t>(hh) * 32 * sizeof(float));
                struct ggml_tensor* dsc = ggml_scale(ctx_, ggml_mul_mat(ctx_, dkh, dqh), 0.1767766953f);
                struct ggml_tensor* dpr = ggml_ops::softmax_rows(ctx_, dsc);
                struct ggml_tensor* dvh = ggml_view_2d(ctx_, dv, 32, RC_NQ, 256 * sizeof(float),
                                                       static_cast<size_t>(hh) * 32 * sizeof(float));
                struct ggml_tensor* dvht = ggml_cont(ctx_, ggml_permute(ctx_, dvh, 1, 0, 2, 3)); // [300,32]
                dheads.push_back(ggml_mul_mat(ctx_, dvht, dpr));                                 // [32,300]
            }
            struct ggml_tensor* dct = dheads[0];
            for (int hh = 1; hh < 8; ++hh) {
                dct = ggml_concat(ctx_, dct, dheads[hh], 0);
            }
            dct = ggml_cont(ctx_, dct); // [256,300]
            // Seg self-attn out_proj is a Gemm(transB=1): Y = X @ W^T.
            // Detection uses MatMul: Y = X @ W. Same stored shape, different math.
            struct ggml_tensor* do_ = seg ? rlinear_gemm_tb(dct, RC_D[dl].out, dlp + "self_attn.out_proj.bias")
                                          : rlinear(dct, RC_D[dl].out, dlp + "self_attn.out_proj.bias");
            struct ggml_tensor* da1 = ggml_add(ctx_, dprev, do_);
            struct ggml_tensor* dnorm1 = rln(da1, dlp + "norm1", 1e-5f);
            struct ggml_tensor* dadd2 = ggml_add(ctx_, dnorm1, refph);
            // Cross-attention value path from the encoder memory. NOTE: the
            // memory is the pre-encoder source tokens (transformer/Concat_1),
            // not the post-LayerNorm encoder output.
            struct ggml_tensor* dvp = rlinear(enc_tok, RC_D[dl].vv, dlp + "cross_attn.value_proj.bias");
            struct ggml_tensor* dvt = ggml_cont(ctx_, ggml_permute(ctx_, dvp, 1, 0, 2, 3));
            struct ggml_tensor* dV4 = ggml_reshape_4d(ctx_, dvt, RC_MS, RC_MS, 16, 16);
            struct ggml_tensor* dSO = rlinear(dadd2, RC_D[dl].so, dlp + "cross_attn.sampling_offsets.bias");
            struct ggml_tensor* dSO4 = ggml_reshape_4d(ctx_, dSO, 2, 2, 16, RC_NQ);
            struct ggml_tensor* dSL0 = ggml_view_2d(ctx_, anchors, 2, RC_NQ, 16, 0);
            struct ggml_tensor* dSL1 = ggml_view_2d(ctx_, anchors, 2, RC_NQ, 16, 8);
            struct ggml_tensor* dSL0b = ggml_reshape_4d(ctx_, ggml_cont(ctx_, dSL0), 2, 1, 1, RC_NQ);
            struct ggml_tensor* dSL1b = ggml_reshape_4d(ctx_, ggml_cont(ctx_, dSL1), 2, 1, 1, RC_NQ);
            struct ggml_tensor* dSL0e = ggml_repeat(ctx_, dSL0b, dSO4);
            struct ggml_tensor* dSL1e = ggml_repeat(ctx_, dSL1b, dSO4);
            struct ggml_tensor* dGRID =
                ggml_add(ctx_, dSL0e, ggml_mul(ctx_, ggml_scale(ctx_, dSO4, 0.25f), dSL1e)); // [2,2,16,300]
            std::vector<float> neg1(static_cast<size_t>(2) * 2 * 16 * static_cast<size_t>(RC_NQ), -1.0f);
            struct ggml_tensor* dSUB =
                ggml_add(ctx_, ggml_scale(ctx_, dGRID, 2.0f), new_const_4d(2, 2, 16, RC_NQ, std::move(neg1)));
            struct ggml_tensor* dG4 = ggml_cont(ctx_, ggml_permute(ctx_, dSUB, 0, 1, 3, 2)); // [2,2,300,16]
            struct ggml_tensor* dGS = ggml_ops::grid_sample(ctx_, dV4, dG4);                 // [2,300,16,16]
            struct ggml_tensor* dAW = rlinear(dadd2, RC_D[dl].aw, dlp + "cross_attn.attention_weights.bias");
            struct ggml_tensor* dAW4 = ggml_reshape_4d(ctx_, dAW, 2, 16, RC_NQ, 1);
            struct ggml_tensor* dPROB = ggml_ops::softmax_rows(ctx_, dAW4);
            struct ggml_tensor* dW5 = ggml_cont(ctx_, ggml_permute(ctx_, dPROB, 0, 2, 1, 3)); // [2,300,16]
            struct ggml_tensor* dR5 = ggml_view_4d(ctx_, dW5, 2, RC_NQ, 1, 16, 8, static_cast<size_t>(RC_NQ) * 8,
                                                   static_cast<size_t>(RC_NQ) * 8, 0);
            struct ggml_tensor* dM6 = ggml_mul(ctx_, dGS, ggml_repeat(ctx_, dR5, dGS));
            struct ggml_tensor* dRS = ggml_sum_rows(ctx_, dM6); // [1,300,16,16]
            // Reshape straight to [1,300,256] (k = d2 + 16*d3, same flat order);
            // NO axis swap here: an extra (0,1,3,2) permute would exchange the
            // two 16-dims and scramble T3.
            struct ggml_tensor* dP256 = ggml_reshape_4d(ctx_, dRS, 1, RC_NQ, 256, 1);
            struct ggml_tensor* dT3 = ggml_cont(ctx_, ggml_permute(ctx_, dP256, 2, 1, 0, 3)); // [256,300,1,1]
            struct ggml_tensor* doo = rlinear(dT3, RC_D[dl].oo, dlp + "cross_attn.output_proj.bias");
            struct ggml_tensor* dAdd3 = ggml_add(ctx_, dnorm1, doo);
            // FFN + norm3 (residual from the norm2 output, per ONNX Add_4).
            struct ggml_tensor* dn2 = rln(dAdd3, dlp + "norm2", 1e-5f);
            struct ggml_tensor* dl1 = rlinear(dn2, RC_D[dl].f1, dlp + "linear1.bias");
            dl1 = ggml_relu(ctx_, dl1);
            struct ggml_tensor* dl2 = rlinear(dl1, RC_D[dl].f2, dlp + "linear2.bias");
            struct ggml_tensor* da4 = ggml_add(ctx_, dn2, dl2);
            dec_norm3[dl] = rln(da4, dlp + "norm3", 1e-5f); // [256,300]
            dprev = dec_norm3[dl];
        }
        dec0_out = dec_norm3[0];
        dec1_out = dec_norm3[1];
        dec2_out = dec_norm3[2];
        if (RC_NL > 3) {
            dec3_out = dec_norm3[3];
        }
        // ---- Slice 2c: detection head on the final decoder norm.
        // bbox MLP (4428/4429/4430 + ReLU) -> [4,300]; class linear (4447)
        // -> [91,300]. Box decode mirrors the anchor refine above:
        // dets_xy = bbox_xy*anchor_wh + anchor_xy (anchor = Concat_10),
        // dets_wh = exp(bbox_wh)*anchor_wh.
        const char* RC_HB0 = seg ? "onnx::MatMul_5008" : "onnx::MatMul_4428";
        const char* RC_HB1 = seg ? "onnx::MatMul_5009" : "onnx::MatMul_4429";
        const char* RC_HB2 = seg ? "onnx::MatMul_5010" : "onnx::MatMul_4430";
        const char* RC_HC = seg ? "onnx::MatMul_5027" : "onnx::MatMul_4447";
        struct ggml_tensor* dec_last = seg ? dec_norm3[3] : dec2_out;
        struct ggml_tensor* dec_norm = rln(dec_last, "transformer.decoder.norm", 1e-5f);
        struct ggml_tensor* hb0 = rlinear(dec_norm, RC_HB0, "bbox_embed.layers.0.bias");
        hb0 = ggml_relu(ctx_, hb0);
        struct ggml_tensor* hb1 = rlinear(hb0, RC_HB1, "bbox_embed.layers.1.bias");
        hb1 = ggml_relu(ctx_, hb1);
        struct ggml_tensor* hb2 = rlinear(hb1, RC_HB2, "bbox_embed.layers.2.bias");
        hlabels = rlinear(dec_norm, RC_HC, "class_embed.bias");
        struct ggml_tensor* hb_xy = ggml_view_2d(ctx_, hb2, 2, RC_NQ, 16, 0);
        struct ggml_tensor* hb_wh = ggml_view_2d(ctx_, hb2, 2, RC_NQ, 16, 8);
        struct ggml_tensor* ha_xy = ggml_view_2d(ctx_, anchors, 2, RC_NQ, 16, 0);
        struct ggml_tensor* ha_wh = ggml_view_2d(ctx_, anchors, 2, RC_NQ, 16, 8);
        struct ggml_tensor* hd_xy = ggml_add(ctx_, ha_xy, ggml_mul(ctx_, hb_xy, ha_wh));
        struct ggml_tensor* hd_wh = ggml_mul(ctx_, ggml_exp(ctx_, hb_wh), ha_wh);
        dets_out = ggml_concat(ctx_, hd_xy, hd_wh, 0);
        if (seg) {
            // ---- Slice 2d: seg mask branch (projector map + final decoder
            // norm). Spatial path: bilinear Resize 36->108, 4x depthwise-sep
            // blocks (dwconv 3x3/s1/p1, residual-free chain like the C2f
            // projector), 1x1 spatial proj. Query path: norm_in LN, MLP
            // 256->1024 (GELU) ->256 + residual, query proj. Einsum
            // bchw,bnc->bnhw + scalar bias -> masks [1,200,108,108].
            struct ggml_tensor* mres = ggml_ops::bilinear_resize(ctx_, logits, 108, 108);
            struct ggml_tensor* mcur = mres;
            for (int mb = 0; mb < 4; ++mb) {
                const std::string MP = "segmentation_head.blocks." + std::to_string(mb) + ".";
                const LoadedTensor& dww = require_tensor(tensors, MP + "dwconv.weight", model_path, shapes_seen);
                require_dims(dww, MP + "dwconv.weight", model_path, {256, 1, 3, 3}, shapes_seen);
                const LoadedTensor& dwb = require_tensor(tensors, MP + "dwconv.bias", model_path, shapes_seen);
                require_dims(dwb, MP + "dwconv.bias", model_path, {256}, shapes_seen);
                // Exact depthwise 3x3 (custom op; ggml_conv_2d is ~1e-3 off
                // here and the mask branch amplifies it coherently). The
                // [256,1,3,3] ONNX bytes are already k[c*9+ky*3+kx], i.e. the
                // [9,256] ggml flat order, so copy verbatim.
                struct ggml_tensor* dkw = new_const_4d(9, 256, 1, 1, dww.data);
                struct ggml_tensor* dkb = new_const_4d(256, 1, 1, 1, dwb.data);
                struct ggml_tensor* dw = ggml_ops::depthwise_conv3x3(ctx_, mcur, dkw, dkb);
                auto seg_ln = [&](struct ggml_tensor* xx, const std::string& pre) -> struct ggml_tensor* {
                    const int64_t cc = xx->ne[2];
                    struct ggml_tensor* pp = ggml_cont(ctx_, ggml_permute(ctx_, xx, 1, 2, 0, 3));
                    struct ggml_tensor* nn = ggml_norm(ctx_, pp, 1e-6f);
                    const LoadedTensor& gw = require_tensor(tensors, pre + ".weight", model_path, shapes_seen);
                    require_dims(gw, pre + ".weight", model_path, {static_cast<uint32_t>(cc)}, shapes_seen);
                    const LoadedTensor& gb = require_tensor(tensors, pre + ".bias", model_path, shapes_seen);
                    require_dims(gb, pre + ".bias", model_path, {static_cast<uint32_t>(cc)}, shapes_seen);
                    struct ggml_tensor* w = new_const_4d(cc, 1, 1, 1, gw.data);
                    struct ggml_tensor* bb = new_const_4d(cc, 1, 1, 1, gb.data);
                    struct ggml_tensor* yy =
                        ggml_add(ctx_, ggml_mul(ctx_, nn, ggml_repeat(ctx_, w, nn)), ggml_repeat(ctx_, bb, nn));
                    return ggml_cont(ctx_, ggml_permute(ctx_, yy, 2, 0, 1, 3));
                };
                dw = seg_ln(dw, MP + "norm");
                // pwconv is a MatMul over the channel dim: permute the map to
                // channel-first [256,108,108] (ggml permute (1,2,0,3)), then
                // the flat bytes coincide with [256,11664] with hw = x+108*y.
                struct ggml_tensor* pw =
                    ggml_reshape_2d(ctx_, ggml_cont(ctx_, ggml_permute(ctx_, dw, 1, 2, 0, 3)), 256, 108 * 108);
                struct ggml_tensor* pwq = rlinear(pw, "onnx::MatMul_503" + std::to_string(2 + mb), MP + "pwconv1.bias");
                pwq = ggml_gelu_erf(ctx_, pwq);
                // pwq is [256,11664] with (c,hw=x+108*y): reshape to
                // [256,108,108] (same flat bytes), then permute back to map
                // order [108,108,256] (ggml permute (2,0,1,3)).
                struct ggml_tensor* pwmap =
                    ggml_cont(ctx_, ggml_permute(ctx_, ggml_reshape_4d(ctx_, pwq, 256, 108, 108, 1), 2, 0, 1, 3));
                mcur = ggml_add(ctx_, mcur, pwmap);
            }
            struct ggml_tensor* spat = rconv(mcur, "segmentation_head.spatial_features_proj.weight", 256, 256, 1, 1, 0);
            {
                // The 1x1 spatial proj carries a plain bias (no norm follows).
                const LoadedTensor& spb =
                    require_tensor(tensors, "segmentation_head.spatial_features_proj.bias", model_path, shapes_seen);
                require_dims(spb, "segmentation_head.spatial_features_proj.bias", model_path, {256}, shapes_seen);
                struct ggml_tensor* sbb = new_const_4d(1, 1, 256, 1, spb.data);
                spat = ggml_add(ctx_, spat, ggml_repeat(ctx_, sbb, spat));
            }
            // Query path from the final decoder norm.
            struct ggml_tensor* qn = rln(dec_norm, "segmentation_head.query_features_block.norm_in", 1e-5f);
            struct ggml_tensor* q0 =
                rlinear(qn, "onnx::MatMul_5036", "segmentation_head.query_features_block.layers.0.bias");
            q0 = ggml_gelu_erf(ctx_, q0);
            struct ggml_tensor* q1 =
                rlinear(q0, "onnx::MatMul_5037", "segmentation_head.query_features_block.layers.2.bias");
            struct ggml_tensor* qa = ggml_add(ctx_, q1, dec_norm);
            struct ggml_tensor* qq =
                rlinear(qa, "onnx::MatMul_5038", "segmentation_head.query_features_proj.bias"); // [256,200]
            // Einsum bchw,bnc->bnhw: out[n,h,w] = sum_c spat[c,h,w]*qq[c,n].
            // sflat must be [256,11664] with sflat[c,hw=x+108*y]: permute the
            // map channel-first first (as for pwconv). mm is [11664,200] with
            // mm[hw,n]. The app-facing NCHW [1,200,108,108] coincides
            // flat-wise with ggml [108,108,200,1] (x fastest, then y, then c),
            // so reshape straight to that (NOT [200,108,108,1]).
            struct ggml_tensor* sflat =
                ggml_reshape_2d(ctx_, ggml_cont(ctx_, ggml_permute(ctx_, spat, 1, 2, 0, 3)), 256, 108 * 108);
            struct ggml_tensor* mm = ggml_mul_mat(ctx_, sflat, qq); // [11664,200]
            struct ggml_tensor* mout = ggml_reshape_4d(ctx_, mm, 108, 108, 200, 1);
            const LoadedTensor& sb = require_tensor(tensors, "segmentation_head.bias", model_path, shapes_seen);
            require_dims(sb, "segmentation_head.bias", model_path, {1}, shapes_seen);
            struct ggml_tensor* sc = new_const_4d(1, 1, 1, 1, sb.data);
            masks_out = ggml_add(ctx_, mout, ggml_repeat(ctx_, sc, mout));
        }
    }

    graph_ = (is_rfdetr || is_rfdetr_seg) ? ggml_new_graph_custom(ctx_, is_rfdetr_seg ? 262144 : 131072, false)
                                          : ggml_new_graph(ctx_);
    ggml_build_forward_expand(graph_, logits);

    if (seg_proto != nullptr) {
        ggml_build_forward_expand(graph_, seg_proto);
        output_tensors_ = {logits, seg_proto};
    } else if (masks_out != nullptr) {
        ggml_build_forward_expand(graph_, dets_out);
        ggml_build_forward_expand(graph_, masks_out);
        ggml_build_forward_expand(graph_, hlabels);
        ggml_build_forward_expand(graph_, aux_out);
        ggml_build_forward_expand(graph_, dec0_out);
        ggml_build_forward_expand(graph_, dec1_out);
        ggml_build_forward_expand(graph_, dec2_out);
        ggml_build_forward_expand(graph_, dec3_out);
        ggml_build_forward_expand(graph_, logits);
        output_tensors_ = {dets_out, masks_out, hlabels, logits, aux_out, dec0_out, dec1_out, dec2_out, dec3_out};
    } else if (dec0_out != nullptr) {
        // App-facing order first: RfDetrPostprocessor defaults to dets=0,
        // labels=1 (and its shape resolver only handles exactly 2 tensors),
        // so dets/labels lead; intermediates follow for parity tests.
        ggml_build_forward_expand(graph_, dets_out);
        ggml_build_forward_expand(graph_, hlabels);
        ggml_build_forward_expand(graph_, aux_out);
        ggml_build_forward_expand(graph_, dec0_out);
        ggml_build_forward_expand(graph_, dec1_out);
        ggml_build_forward_expand(graph_, dec2_out);
        ggml_build_forward_expand(graph_, logits);
        output_tensors_ = {dets_out, hlabels, logits, aux_out, dec0_out, dec1_out, dec2_out};
    } else if (aux_out != nullptr) {
        ggml_build_forward_expand(graph_, aux_out);
        output_tensors_ = {logits, aux_out};
    } else {
        output_tensors_ = {logits};
    }

    // Allocate backend buffer (tensors use no_alloc ctx, so pointers are
    // unset until now); allocation failure is fatal.
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
    if (!buffer_) {
        fail_model(model_path, "backend buffer allocation failed");
    }

    // Copy staged constants now that tensor storage is valid.
    for (const auto& fill : pending) {
        std::memcpy(fill.tensor->data, fill.data.data(), fill.data.size() * sizeof(float));
    }
    for (const auto& fill : pending_f16) {
        std::memcpy(fill.tensor->data, fill.data.data(), fill.data.size() * sizeof(ggml_fp16_t));
    }
    for (const auto& fill : pending_i32) {
        std::memcpy(fill.tensor->data, fill.data.data(), fill.data.size() * sizeof(int32_t));
    }

    const std::string rfdetr_msg =
        "RF-DETR (DINOv2-S backbone + projector + neck + 3-layer deformable decoder + head, dets [1,300,4] + "
        "labels [1,300,91], real weights)";
    const std::string rfdetr_seg_msg =
        "RF-DETR-seg (DINOv2-S backbone + projector + neck + 4-layer deformable decoder + head + mask branch, dets "
        "[1,200,4] + labels [1,200,91] + masks [1,200,108,108], real weights)";
    LOG(INFO) << "GGML model loaded: full "
              << (is_yolo_seg_
                      ? "YOLOv8n-seg (CSP stem + C2f + SPPF + PAN + decoupled head + DFL + mask/proto, "
                        "real weights)"
                      : (is_yolo11_
                             ? "YOLO11n (CSP stem + C3k2 + SPPF + C2PSA + PAN + decoupled head + DFL, real "
                               "weights)"
                             : (is_yolo
                                    ? "YOLOv8n (CSP stem + C2f + SPPF + PAN + decoupled head + DFL, real weights)"
                                    : (is_rfdetr_seg
                                           ? rfdetr_seg_msg
                                           : (is_rfdetr ? rfdetr_msg
                                                        : (is_bottleneck ? "ResNet50 (stem + 16 bottleneck blocks + "
                                                                           "GAP + FC 1000, real weights)"
                                                                         : "ResNet18 (stem + 8 basic blocks + GAP + "
                                                                           "FC 1000, real weights)"))))));
}

void GGMLInfer::setup_input_output_tensors(const std::vector<std::vector<int64_t>>& input_sizes) {
    if (input_sizes.empty()) {
        throw std::runtime_error("Input sizes must be specified for GGML backend");
    }

    // Setup input tensors
    for (size_t i = 0; i < input_sizes.size(); i++) {
        const auto& shape = input_sizes[i];
        if (shape.size() != 4) {
            throw std::runtime_error("GGML expects 4D input tensors (batch, channels, height, width)");
        }

        std::string input_name = "input" + std::to_string(i + 1);
        inference_metadata_.addInput(input_name, shape, batch_size_);
    }
    if (is_yolo_) {
        // Fixed YOLOv8n contract: identical-byte-count inputs would silently
        // reinterpret any same-product geometry as 640x640, so enforce it.
        const std::vector<int64_t>& in0 = input_sizes[0];
        if (!(in0.size() == 4 && in0[0] == static_cast<int64_t>(batch_size_) && in0[1] == 3 && in0[2] == 640 &&
              in0[3] == 640)) {
            throw std::runtime_error("GGML YOLOv8n path requires exactly {N,3,640,640} input");
        }
    }
    if (!is_yolo_ && input_height_ == 560 && input_width_ == 560) {
        // RF-DETR S-1 intermediate contract (no is_rfdetr_ member: the header
        // is shared, so the 560 geometry set at dispatch identifies the path).
        const std::vector<int64_t>& in0 = input_sizes[0];
        if (!(in0.size() == 4 && in0[0] == static_cast<int64_t>(batch_size_) && in0[1] == 3 && in0[2] == 560 &&
              in0[3] == 560)) {
            throw std::runtime_error("GGML RF-DETR path requires exactly {N,3,560,560} input");
        }
    }
    if (!is_yolo_ && input_height_ == 432 && input_width_ == 432) {
        // RF-DETR-seg contract (432 geometry set at dispatch identifies the path).
        const std::vector<int64_t>& in0 = input_sizes[0];
        if (!(in0.size() == 4 && in0[0] == static_cast<int64_t>(batch_size_) && in0[1] == 3 && in0[2] == 432 &&
              in0[3] == 432)) {
            throw std::runtime_error("GGML RF-DETR-seg path requires exactly {N,3,432,432} input");
        }
    }

    // Output metadata follows the per-model contract set at load
    // (resnet {N,1000}, detection {1,84,8400}, segmentation
    // {1,116,8400} + {1,32,160,160}).
    output_names_.clear();
    for (size_t i = 0; i < output_shapes_.size(); ++i) {
        std::vector<int64_t> output_shape = output_shapes_[i];
        if (output_shape.size() == 2) {
            output_shape[0] = static_cast<int64_t>(batch_size_);
        }
        const std::string name = output_shapes_.size() == 1 ? "output" : "output" + std::to_string(i);
        inference_metadata_.addOutput(name, output_shape, batch_size_);
        output_names_.push_back(name);
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
GGMLInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {

    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    start_timer();

    try {
        // Process all input tensors
        if (input_tensors.size() != 1) {
            throw std::runtime_error("GGML backend currently supports only single input models, got " +
                                     std::to_string(input_tensors.size()) + " inputs");
        }

        const std::vector<uint8_t>& input_raw = input_tensors[0];

        // Fixed contract: single 4D F32 input, HxWx3 elements per batch entry
        // (geometry set at load: 224 for ResNet, 640 for YOLOv8n).
        const size_t elems_per_sample = static_cast<size_t>(input_height_) * input_width_ * kInputChannels;
        const size_t expected_bytes = elems_per_sample * sizeof(float) * static_cast<size_t>(batch_size_);

        if (input_raw.size() != expected_bytes) {
            throw std::runtime_error("Input data size mismatch. Expected " + std::to_string(expected_bytes) +
                                     " bytes (based on tensor type/shape), got " + std::to_string(input_raw.size()));
        }

        // Feed the graph input [W,H,C,N] (D-8): the NCHW blob [N,3,H,W]
        // is row-major x + W*(y + H*(c + C*n)), which coincides
        // element-for-element with ggml [W,H,C,N] (x + ne0*(y + ne1*(c +
        // ne2*n))), so the layout conversion is a single copy per infer with
        // no in-graph transpose.
        const float* in_f32 = reinterpret_cast<const float*>(input_raw.data());
        float* feat_f32 = static_cast<float*>(feat_tensor_->data);
        std::memcpy(feat_f32, in_f32, input_raw.size());

        // Execute the graph (if backend is available)
        if (backend_) {
            ggml_backend_graph_compute(backend_, graph_);
        } else {
            throw std::runtime_error("GGML backend unavailable during inference");
        }
        // DEBUG: dump intermediate tensors after graph computation (YOLO11 only)
        if (is_yolo11_) {
            if (debug_reg_) {
                std::vector<float> reg_data(ggml_nelements(debug_reg_));
                ggml_backend_tensor_get(debug_reg_, reg_data.data(), 0, reg_data.size() * sizeof(float));
                std::ofstream reg_out("/tmp/debug_reg.bin", std::ios::binary);
                reg_out.write(reinterpret_cast<char*>(reg_data.data()), reg_data.size() * sizeof(float));
            }
            if (debug_dm_) {
                std::vector<float> dm_data(ggml_nelements(debug_dm_));
                ggml_backend_tensor_get(debug_dm_, dm_data.data(), 0, dm_data.size() * sizeof(float));
                std::ofstream dm_out("/tmp/debug_dm.bin", std::ios::binary);
                dm_out.write(reinterpret_cast<char*>(dm_data.data()), dm_data.size() * sizeof(float));
            }
            if (debug_d4_) {
                std::vector<float> d4_data(ggml_nelements(debug_d4_));
                ggml_backend_tensor_get(debug_d4_, d4_data.data(), 0, d4_data.size() * sizeof(float));
                std::ofstream d4_out("/tmp/debug_d4.bin", std::ios::binary);
                d4_out.write(reinterpret_cast<char*>(d4_data.data()), d4_data.size() * sizeof(float));
            }
            for (int s = 0; s < 3; ++s) {
                if (debug_feat_[s]) {
                    std::vector<float> feat_data(ggml_nelements(debug_feat_[s]));
                    ggml_backend_tensor_get(debug_feat_[s], feat_data.data(), 0, feat_data.size() * sizeof(float));
                    std::ofstream feat_out("/tmp/debug_feat" + std::to_string(s) + ".bin", std::ios::binary);
                    feat_out.write(reinterpret_cast<char*>(feat_data.data()), feat_data.size() * sizeof(float));
                }
            }
            // Backbone/neck debug tensors
            auto dump_tensor = [&](struct ggml_tensor* t, const char* name) {
                if (t) {
                    std::vector<float> data(ggml_nelements(t));
                    ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
                    std::ofstream out(std::string("/tmp/debug_") + name + ".bin", std::ios::binary);
                    out.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
                }
            };
            dump_tensor(debug_m4_, "m4");
            dump_tensor(debug_m6_, "m6");
            dump_tensor(debug_p5_, "p5");
            dump_tensor(debug_up1_, "up1");
            dump_tensor(debug_n13_, "n13");
            dump_tensor(debug_up2_, "up2");
            dump_tensor(debug_d5_, "d5");
            dump_tensor(debug_d6_, "d6");
            dump_tensor(debug_m3_, "m3");
            dump_tensor(debug_m4_cv1_, "m4_cv1");
            dump_tensor(debug_m4_m0_cv1_, "m4_m0_cv1");
            dump_tensor(debug_m4_m0_cv2_, "m4_m0_cv2");
            dump_tensor(debug_m4_m0_add_, "m4_m0_add");
            dump_tensor(debug_m4_cv2_, "m4_cv2");
            dump_tensor(debug_m2_, "m2");
            dump_tensor(debug_m1_, "m1");
            dump_tensor(debug_m0_, "m0");
        }

        // Real outputs: read every tensor the graph exposes, flat in
        // NCHW-coincident order. ResNet keeps its per-batch {N,1000} layout;
        // YOLO detection returns {1,84,8400}; segmentation returns
        // {1,116,8400} then {1,32,160,160}.
        if (output_tensors_.empty() || output_tensors_.size() != output_shapes_.size()) {
            throw std::runtime_error(
                "GGML output contract mismatch: tensors=" + std::to_string(output_tensors_.size()) +
                " shapes=" + std::to_string(output_shapes_.size()));
        }
        std::vector<std::vector<TensorElement>> outputs;
        std::vector<std::vector<int64_t>> shapes;
        if (output_tensors_.size() == 1 && output_shapes_[0].size() == 2) {
            // Existing single-output classifier contract: one vector per
            // batch entry, shape {batch_size, classes}.
            struct ggml_tensor* logits = output_tensors_.at(0);
            if (logits->type != GGML_TYPE_F32) {
                throw std::runtime_error("GGML output is not F32 (type=" +
                                         std::to_string(static_cast<int>(logits->type)) + "); refusing to reinterpret");
            }
            const float* logits_f32 = static_cast<const float*>(logits->data);
            const int64_t out_classes = output_shapes_[0].back();
            const int64_t out_rows = 1;
            for (size_t b = 0; b < static_cast<size_t>(batch_size_); ++b) {
                std::vector<TensorElement> output_data;
                output_data.reserve(static_cast<size_t>(out_classes * out_rows));
                for (int64_t c = 0; c < out_classes * out_rows; ++c) {
                    output_data.emplace_back(logits_f32[b * static_cast<size_t>(out_classes * out_rows) + c]);
                }
                outputs.push_back(std::move(output_data));
                shapes.push_back({static_cast<int64_t>(batch_size_), out_classes});
            }
        } else {
            for (size_t oi = 0; oi < output_tensors_.size(); ++oi) {
                if (output_tensors_[oi]->type != GGML_TYPE_F32) {
                    throw std::runtime_error("GGML output[" + std::to_string(oi) + "] is not F32 (type=" +
                                             std::to_string(static_cast<int>(output_tensors_[oi]->type)) +
                                             "); refusing to reinterpret");
                }
                const size_t elements = static_cast<size_t>(ggml_nelements(output_tensors_[oi]));
                const float* data = static_cast<const float*>(output_tensors_[oi]->data);
                std::vector<TensorElement> output_data;
                output_data.reserve(elements);
                for (size_t i = 0; i < elements; ++i) {
                    output_data.emplace_back(data[i]);
                }
                outputs.push_back(std::move(output_data));
                shapes.push_back(output_shapes_[oi]);
            }
        }

        end_timer();

        return std::make_tuple(outputs, shapes);

    } catch (const std::exception& e) {
        end_timer();
        throw InferenceExecutionException(e.what());
    }
}

std::vector<TensorElement> GGMLInfer::tensor_to_vector(struct ggml_tensor* tensor) {
    std::vector<TensorElement> result;
    size_t total_elements = ggml_nelements(tensor);
    result.reserve(total_elements);

    float* data = static_cast<float*>(tensor->data);
    for (size_t i = 0; i < total_elements; ++i) {
        result.push_back(data[i]);
    }

    return result;
}

std::vector<int64_t> GGMLInfer::get_tensor_shape(struct ggml_tensor* tensor) {
    std::vector<int64_t> shape;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] > 1) {
            shape.push_back(tensor->ne[i]);
        }
    }
    return shape;
}
