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
                // Zero-sized dims are legal (e.g. an empty Resize roi);
                // the product below becomes 0 and num_elements must be 0.
                dim_product = 0;
                // Still consume the dim value check below; skip the rest.
                for (++d; d < ndims; ++d) {
                    if (!read_pod(file, dims[d])) {
                        fail_model(model_path, "truncated dims of tensor '" + name + "'");
                    }
                }
                break;
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

    // YOLOv8n detection: dot-separated ultralytics tensor names anchored at
    // the stem conv. Fixed 1x3x640x640 input, [1,84,8400] raw output.
    const bool is_yolo = tensors.count("model.0.conv.weight") != 0;
    is_yolo_ = is_yolo;
    if (is_yolo) {
        input_height_ = 640;
        input_width_ = 640;
        output_shape_ = {1, 84, 8400};
        if (batch_size_ != 1) {
            fail_model(model_path, "GGML YOLOv8n path is fixed batch-1 (box math constants are single-batch)");
        }
    }

    std::string prefix;
    std::string fc_w_name;
    std::string fc_b_name;
    uint32_t fc_in_dim = 0;
    bool is_bottleneck = false;
    if (!is_yolo) {
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
    // each inference. Geometry is per-model (members set at dispatch).
    feat_tensor_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, input_width_, input_height_, kInputChannels, N);

    struct ggml_tensor* x = feat_tensor_;
    struct ggml_tensor* logits = nullptr;

    if (!is_yolo) {

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
    } else {
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
        auto yact = [&](struct ggml_tensor * xx, struct ggml_tensor * w, struct ggml_tensor * b, int s,
                        int p) -> struct ggml_tensor* {
            struct ggml_tensor* h = ggml_conv_2d(ctx_, w, xx, s, s, p, p, 1, 1);
            h = ggml_add(ctx_, h, ggml_repeat(ctx_, b, h));
            return ggml_mul(ctx_, h, ggml_sigmoid(ctx_, h));
        };
        auto yconv = [&](struct ggml_tensor * xx, struct ggml_tensor * w, struct ggml_tensor * b, int s,
                         int p) -> struct ggml_tensor* {
            struct ggml_tensor* h = ggml_conv_2d(ctx_, w, xx, s, s, p, p, 1, 1);
            return ggml_add(ctx_, h, ggml_repeat(ctx_, b, h));
        };
        // C2f block (nodes 6-20 pattern): cv1 1x1 -> split -> n bottlenecks
        // (3x3+silu, 3x3+silu [, +residual]) chained -> concat all -> cv2 1x1.
        auto c2f = [&](struct ggml_tensor * xx, const std::string& pre, uint32_t C, uint32_t Cin, int n,
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

        // --- backbone ---
        struct ggml_tensor* y = yact(x, ykernel("model.0.conv", 16, 3, 3, 3), ybias("model.0.conv", 16), 2, 1); // 320
        y = yact(y, ykernel("model.1.conv", 32, 16, 3, 3), ybias("model.1.conv", 32), 2, 1);                    // 160
        y = c2f(y, "model.2", 32, 32, 1, true);                                                                 // 160
        y = yact(y, ykernel("model.3.conv", 64, 32, 3, 3), ybias("model.3.conv", 64), 2, 1);                    // 80
        struct ggml_tensor* m4 = c2f(y, "model.4", 64, 64, 2, true);                                            // 80
        y = yact(m4, ykernel("model.5.conv", 128, 64, 3, 3), ybias("model.5.conv", 128), 2, 1);                 // 40
        struct ggml_tensor* m6 = c2f(y, "model.6", 128, 128, 2, true);                                          // 40
        y = yact(m6, ykernel("model.7.conv", 256, 128, 3, 3), ybias("model.7.conv", 256), 2, 1);                // 20
        y = c2f(y, "model.8", 256, 256, 1, true);                                                               // 20
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
            struct ggml_tensor* hb =
                yact(feats[s], ykernel(pre + ".0.conv", 64, feats[s]->ne[2], 3, 3), ybias(pre + ".0.conv", 64), 1, 1);
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
        // /2, concat, *strides. All consts carried (float initializers).
        const size_t row4 = static_cast<size_t>(8400) * sizeof(float);
        struct ggml_tensor* xy = ggml_view_4d(ctx_, d4, 8400, 1, 2, 1, row4, row4, row4, 0);
        struct ggml_tensor* wh = ggml_view_4d(ctx_, d4, 8400, 1, 2, 1, row4, row4, row4, 2 * row4);
        struct ggml_tensor* A = yconst("/model.22/Constant_12_output_0", {1, 2, 8400});
        struct ggml_tensor* B = yconst("/model.22/Constant_13_output_0", {1, 2, 8400});
        struct ggml_tensor* sub0 = ggml_sub(ctx_, A, xy);
        struct ggml_tensor* add1 = ggml_add(ctx_, B, wh);
        struct ggml_tensor* add2 = ggml_add(ctx_, sub0, add1);
        struct ggml_tensor* sub1 = ggml_sub(ctx_, add1, sub0);
        struct ggml_tensor* div1 = ggml_scale(ctx_, add2, 0.5f);
        struct ggml_tensor* cat2 = ggml_concat(ctx_, div1, sub1, 2);
        struct ggml_tensor* mul2 =
            ggml_mul(ctx_, cat2, ggml_repeat(ctx_, yconst("/model.22/Constant_15_output_0", {1, 8400}), cat2));
        // Cls sigmoid: 1 / (1 + exp(-x)).
        struct ggml_tensor* ones = new_const_4d(8400, 1, 80, 1, std::vector<float>(8400 * 80, 1.0f));
        struct ggml_tensor* sig = ggml_div(ctx_, ones, ggml_add(ctx_, ones, ggml_exp(ctx_, ggml_neg(ctx_, clsc))));
        logits = ggml_concat(ctx_, mul2, sig, 2); // [8400,1,84,1] == NCHW [1,84,8400]
    }

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
              << (is_yolo ? "YOLOv8n (CSP stem + C2f + SPPF + PAN + decoupled head + DFL, real weights)"
                          : (is_bottleneck ? "ResNet50 (stem + 16 bottleneck blocks + GAP + FC 1000, real weights)"
                                           : "ResNet18 (stem + 8 basic blocks + GAP + FC 1000, real weights)"));
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

    // Output metadata follows the per-model contract set at load
    // (resnet {N,1000}, yolo {1,84,8400}).
    std::vector<int64_t> output_shape = output_shape_;
    if (output_shape.size() == 2) {
        output_shape[0] = static_cast<int64_t>(batch_size_);
    }
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

        // Real outputs: read the output tensor produced by the graph, flat in
        // NCHW-coincident order (resnet {N,1000}, yolo {1,84,8400}).
        struct ggml_tensor* logits = output_tensors_.at(0);
        const float* logits_f32 = static_cast<const float*>(logits->data);
        const bool is3d = output_shape_.size() == 3;
        const int64_t out_classes = is3d ? output_shape_[1] : output_shape_.back();
        const int64_t out_rows = is3d ? output_shape_[2] : 1;
        std::vector<std::vector<TensorElement>> outputs;
        std::vector<std::vector<int64_t>> shapes;
        for (size_t b = 0; b < static_cast<size_t>(batch_size_); ++b) {
            std::vector<TensorElement> output_data;
            output_data.reserve(static_cast<size_t>(out_classes * out_rows));
            for (int64_t c = 0; c < out_classes * out_rows; ++c) {
                output_data.emplace_back(logits_f32[b * static_cast<size_t>(out_classes * out_rows) + c]);
            }
            outputs.push_back(std::move(output_data));
            if (is3d) {
                shapes.push_back({static_cast<int64_t>(batch_size_), out_classes, out_rows});
            } else {
                shapes.push_back({static_cast<int64_t>(batch_size_), out_classes});
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
