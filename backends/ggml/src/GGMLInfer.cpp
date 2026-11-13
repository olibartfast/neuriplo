#include "GGMLInfer.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Real-loader weight layout ("NGGL", Q-1 decision):
//   magic:      4 bytes "NGGL"
//   version:    uint32 == 1
//   num_tensors:uint32
//   per tensor: uint32 name_len, name bytes, uint32 ndims, uint32 dims[ndims],
//               uint64 num_elements, float32 data[num_elements]
// All integers little-endian (= native on the supported x86_64 CPU target);
// floats are native-endian F32. dims[] are in ONNX order as written by
// scripts/convert_onnx_to_ggml.py (all float initializers, names verbatim).
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
constexpr size_t kInputHeight = 224;
constexpr size_t kInputWidth = 224;
constexpr int64_t kInputElements = 150528; // 3*224*224
constexpr int64_t kNumClasses = 1000;
// D-7: exact frozen-inference BN fold; epsilon verified uniform (1e-5) in
// resnet18-v1-7.onnx (all 20 BatchNormalization nodes) and in
// resnet50-v1-12.onnx (all 53 nodes; file value 9.9999997e-06, the float32
// spelling of 1e-5).
constexpr double kBatchNormEps = 1e-5;
constexpr uint32_t kFileVersion = 1;
// ResNet-50 carries 299 float initializers; headroom for larger variants.
constexpr uint32_t kMaxTensors = 512;
constexpr uint32_t kMaxNameLen = 256;

struct LoadedTensor {
    std::vector<uint32_t> dims;
    std::vector<float> data;
};

[[noreturn]] void fail_model(const std::string& model_path, const std::string& reason) {
    throw std::runtime_error("Invalid GGML model file '" + model_path + "': " + reason);
}

template <typename T> bool read_pod(std::ifstream& f, T& out) {
    f.read(reinterpret_cast<char*>(&out), sizeof(T));
    return static_cast<size_t>(f.gcount()) == sizeof(T);
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

    LOG(INFO) << "Loading GGML model from: " << model_path;

    // --- header ---
    char magic[4] = {0, 0, 0, 0};
    file.read(magic, 4);
    if (static_cast<size_t>(file.gcount()) != 4) {
        fail_model(model_path, "truncated file header (expected magic 'NGGL')");
    }
    if (std::memcmp(magic, "NGGL", 4) != 0) {
        fail_model(model_path, std::string("bad magic '") + std::string(magic, 4) + "', expected 'NGGL'");
    }
    uint32_t version = 0;
    if (!read_pod(file, version)) {
        fail_model(model_path, "truncated file header (missing version)");
    }
    if (version != kFileVersion) {
        fail_model(model_path, "unsupported version " + std::to_string(version) + ", expected 1");
    }
    uint32_t num_tensors = 0;
    if (!read_pod(file, num_tensors)) {
        fail_model(model_path, "truncated file header (missing tensor count)");
    }
    if (num_tensors < 2 || num_tensors > kMaxTensors) {
        fail_model(model_path, "implausible tensor count " + std::to_string(num_tensors));
    }

    // --- tensors ---
    std::map<std::string, LoadedTensor> tensors;
    for (uint32_t t = 0; t < num_tensors; ++t) {
        uint32_t name_len = 0;
        if (!read_pod(file, name_len)) {
            fail_model(model_path, "truncated tensor entry " + std::to_string(t) + " (missing name length)");
        }
        if (name_len == 0 || name_len > kMaxNameLen) {
            fail_model(model_path, "tensor entry " + std::to_string(t) + " has invalid name length");
        }
        std::string name(name_len, '\0');
        file.read(name.data(), name_len);
        if (static_cast<size_t>(file.gcount()) != name_len) {
            fail_model(model_path, "truncated name of tensor entry " + std::to_string(t));
        }
        uint32_t ndims = 0;
        if (!read_pod(file, ndims)) {
            fail_model(model_path, "truncated dims of tensor '" + name + "'");
        }
        if (ndims < 1 || ndims > 4) {
            fail_model(model_path, "tensor '" + name + "' has invalid ndims " + std::to_string(ndims));
        }
        std::vector<uint32_t> dims(ndims);
        uint64_t dim_product = 1;
        for (uint32_t d = 0; d < ndims; ++d) {
            if (!read_pod(file, dims[d])) {
                fail_model(model_path, "truncated dims of tensor '" + name + "'");
            }
            if (dims[d] == 0) {
                fail_model(model_path, "tensor '" + name + "' has zero-sized dim " + std::to_string(d));
            }
            if (dim_product > (uint64_t(1) << 32) / dims[d]) {
                fail_model(model_path, "tensor '" + name + "' dims overflow");
            }
            dim_product *= dims[d];
        }
        uint64_t num_elements = 0;
        if (!read_pod(file, num_elements)) {
            fail_model(model_path, "truncated element count of tensor '" + name + "'");
        }
        if (num_elements != dim_product) {
            fail_model(model_path, "tensor '" + name + "' declares " + std::to_string(num_elements) +
                                       " elements but dims product is " + std::to_string(dim_product));
        }
        // Bound the allocation by the bytes actually left in the file, so a
        // corrupt count fails with the path in the message instead of OOM.
        const std::streampos data_pos = file.tellg();
        file.seekg(0, std::ios::end);
        const std::streampos file_end = file.tellg();
        file.seekg(data_pos);
        if (data_pos == std::streampos(-1) || file_end == std::streampos(-1)) {
            fail_model(model_path, "cannot determine size of tensor '" + name + "'");
        }
        const uint64_t remaining = static_cast<uint64_t>(file_end - data_pos);
        if (num_elements > remaining / sizeof(float)) {
            fail_model(model_path, "tensor '" + name + "' claims " + std::to_string(num_elements) +
                                       " elements but only " + std::to_string(remaining) + " bytes remain");
        }
        std::vector<float> data;
        data.resize(static_cast<size_t>(num_elements));
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(num_elements * sizeof(float)));
        if (static_cast<size_t>(file.gcount()) != num_elements * sizeof(float)) {
            fail_model(model_path, "truncated data of tensor '" + name + "' (expected " +
                                       std::to_string(num_elements * sizeof(float)) + " bytes)");
        }
        if (tensors.count(name) != 0) {
            fail_model(model_path, "duplicate tensor '" + name + "'");
        }
        tensors.emplace(name, LoadedTensor{std::move(dims), std::move(data)});
    }
    // No trailing garbage: byte counts must be exact.
    char extra = 0;
    file.read(&extra, 1);
    if (static_cast<size_t>(file.gcount()) != 0) {
        fail_model(model_path, "trailing bytes after tensor data");
    }

    // --- ResNet topology from exact tensor names (D-6) ---
    // Dispatch ResNet-18 vs ResNet-50 by FC input dim (512 vs 2048) plus
    // bottleneck tensor presence (enforced by the stage requires below):
    // exact per-file classifier names first (D-1), dims-only structural
    // fallback where names genuinely vary. Anything unrecognized fails loud
    // with the path and the shapes seen (R-5).
    const std::string shapes_seen = describe_shapes(tensors);

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
    std::string prefix;
    std::string fc_w_name;
    std::string fc_b_name;
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
                               "ambiguous classifier weights (need one 2D [1000,K]/[K,1000]); tensors:" + shapes_seen);
                }
                w_name = kv.first;
            }
        }
        if (w_name.empty()) {
            fail_model(model_path, "no recognized ResNet-18/50 classifier pair (need [1000,512]/[1000,2048] weight + "
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
        !((fc_weight_probe.dims[0] == 1000 && (fc_weight_probe.dims[1] == 512 || fc_weight_probe.dims[1] == 2048)) ||
          (fc_weight_probe.dims[1] == 1000 && (fc_weight_probe.dims[0] == 512 || fc_weight_probe.dims[0] == 2048)))) {
        fail_model(model_path, "classifier weight '" + fc_w_name +
                                   "' has unexpected dims (need [1000,512]/[1000,2048] "
                                   "or transposed); tensors:" +
                                   shapes_seen);
    }
    const uint32_t fc_in_dim = (fc_weight_probe.dims[0] == 1000) ? fc_weight_probe.dims[1] : fc_weight_probe.dims[0];
    // FC input dim decides the block topology; the stage requires below are
    // the bottleneck tensor-presence check (a 2048-d file without bottleneck
    // stage tensors fails on the first missing/wrong-shaped conv).
    const bool is_bottleneck = (fc_in_dim == 2048);

    // Constant fills staged here: tensor storage is only valid after the
    // backend buffer is allocated, so weights are memcpy'd post-alloc.
    struct PendingFill {
        struct ggml_tensor* tensor = nullptr;
        std::vector<float> data;
    };
    std::vector<PendingFill> pending;
    auto new_const_4d = [&](int64_t n0, int64_t n1, int64_t n2, int64_t n3,
                            std::vector<float> data) -> struct ggml_tensor* {
        struct ggml_tensor* t = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, n0, n1, n2, n3);
        pending.push_back(PendingFill{t, std::move(data)});
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
    // each inference.
    feat_tensor_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, kInputWidth, kInputHeight, kInputChannels, N);

    struct ggml_tensor* x = feat_tensor_;

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
                    const FoldedConv ds = folded_conv(pre + "conv3_weight", pre + "batchnorm3_", 4 * C, in_ch, 1, 1);
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
    struct ggml_tensor* logits = ggml_add(ctx_, mm, ggml_repeat(ctx_, fc_b, mm));

    graph_ = ggml_new_graph(ctx_);
    ggml_build_forward_expand(graph_, logits);
    output_tensors_ = {logits};

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

    LOG(INFO) << "GGML model loaded: full "
              << (is_bottleneck ? "ResNet50 (stem + 16 bottleneck blocks + GAP + FC 1000, real weights)"
                                : "ResNet18 (stem + 8 basic blocks + GAP + FC 1000, real weights)");
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

    // Setup output tensors (placeholder - would be determined from actual model)
    std::vector<int64_t> output_shape = {static_cast<int64_t>(batch_size_), 1000}; // Example: 1000 classes
    inference_metadata_.addOutput("output", output_shape, batch_size_);

    output_names_.push_back("output");
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

        // Fixed contract: single 4D F32 input, 150528 elements per batch entry.
        const size_t expected_bytes =
            static_cast<size_t>(kInputElements) * sizeof(float) * static_cast<size_t>(batch_size_);

        if (input_raw.size() != expected_bytes) {
            throw std::runtime_error("Input data size mismatch. Expected " + std::to_string(expected_bytes) +
                                     " bytes (based on tensor type/shape), got " + std::to_string(input_raw.size()));
        }

        // Feed the graph input [W,H,C,N] (D-8): the NCHW blob [N,3,224,224]
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

        // Real outputs: read the logits tensor produced by the graph.
        struct ggml_tensor* logits = output_tensors_.at(0);
        const float* logits_f32 = static_cast<const float*>(logits->data);
        std::vector<std::vector<TensorElement>> outputs;
        std::vector<std::vector<int64_t>> shapes;
        for (size_t b = 0; b < static_cast<size_t>(batch_size_); ++b) {
            std::vector<TensorElement> output_data;
            output_data.reserve(static_cast<size_t>(kNumClasses));
            for (int64_t c = 0; c < kNumClasses; ++c) {
                output_data.emplace_back(logits_f32[b * static_cast<size_t>(kNumClasses) + c]);
            }
            outputs.push_back(std::move(output_data));
            shapes.push_back({static_cast<int64_t>(batch_size_), kNumClasses});
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
