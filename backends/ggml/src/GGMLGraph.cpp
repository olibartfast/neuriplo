#include "GGMLGraph.hpp"

#include "GGMLOps.hpp"

#include <cmath>
#include <cstring>
#include <deque>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ggml_graph {
namespace {

enum ggml_type to_ggml_type(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return GGML_TYPE_F32;
    case DType::I32:
        return GGML_TYPE_I32;
    case DType::I64:
        return GGML_TYPE_I64;
    }
    throw std::invalid_argument("unsupported dtype");
}

DType from_ggml_type(enum ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32:
        return DType::F32;
    case GGML_TYPE_I32:
        return DType::I32;
    case GGML_TYPE_I64:
        return DType::I64;
    default:
        throw std::invalid_argument("unsupported ggml tensor type");
    }
}

struct PendingF32 {
    struct ggml_tensor* tensor;
    std::vector<float> data;
};

struct PendingI32 {
    struct ggml_tensor* tensor;
    std::vector<int32_t> data;
};

struct GridSampleParams {
    int align_corners = 0;
    int error = 0;
};

struct IntegerBinaryParams {
    int op = 0;
    int error = 0;
};

void integer_binary_callback(struct ggml_tensor* dst, const struct ggml_tensor* left, const struct ggml_tensor* right,
                             int ith, int nth, void* userdata) {
    auto* params = static_cast<IntegerBinaryParams*>(userdata);
    const int64_t count = ggml_nelements(dst);
    const int64_t left_count = ggml_nelements(left);
    const int64_t right_count = ggml_nelements(right);
    if (right_count != 1 && right_count != left_count) {
        params->error = 1;
        return;
    }
    const int64_t begin = count * ith / nth;
    const int64_t end = count * (ith + 1) / nth;
    auto read = [](const struct ggml_tensor* tensor, int64_t index) -> int64_t {
        if (tensor->type == GGML_TYPE_I64) {
            return static_cast<const int64_t*>(tensor->data)[index];
        }
        return static_cast<const int32_t*>(tensor->data)[index];
    };
    for (int64_t i = begin; i < end; ++i) {
        const int64_t a = read(left, i % left_count);
        const int64_t b = read(right, right_count == 1 ? 0 : i);
        int64_t value = 0;
        switch (params->op) {
        case 0:
            value = a + b;
            break;
        case 1:
            value = a - b;
            break;
        case 2:
            value = a * b;
            break;
        case 3:
            if (b == 0) {
                params->error = 1;
                return;
            }
            value = a / b;
            break;
        default:
            params->error = 1;
            return;
        }
        if (dst->type == GGML_TYPE_I64) {
            static_cast<int64_t*>(dst->data)[i] = value;
        } else {
            static_cast<int32_t*>(dst->data)[i] = static_cast<int32_t>(value);
        }
    }
}

void grid_sample_bilinear_callback(struct ggml_tensor* dst, const struct ggml_tensor* /*dummy*/,
                                   const struct ggml_tensor* input, const struct ggml_tensor* grid, int ith, int nth,
                                   void* userdata) {
    auto* params = static_cast<GridSampleParams*>(userdata);
    const int64_t width = input->ne[0];
    const int64_t height = input->ne[1];
    const int64_t channels = input->ne[2];
    const int64_t batch = input->ne[3];
    const int64_t out_width = grid->ne[1];
    const int64_t out_height = grid->ne[2];
    if (grid->ne[0] != 2 || grid->ne[3] != batch || dst->ne[0] != out_width || dst->ne[1] != out_height ||
        dst->ne[2] != channels || dst->ne[3] != batch) {
        params->error = 1;
        return;
    }
    const int64_t count = ggml_nelements(dst);
    const int64_t begin = count * ith / nth;
    const int64_t end = count * (ith + 1) / nth;
    const float* x = static_cast<const float*>(input->data);
    const float* g = static_cast<const float*>(grid->data);
    float* output = static_cast<float*>(dst->data);
    auto sample = [&](int64_t xx, int64_t yy, int64_t c, int64_t n) -> float {
        if (xx < 0 || xx >= width || yy < 0 || yy >= height) {
            return 0.0f;
        }
        return x[xx + width * (yy + height * (c + channels * n))];
    };
    for (int64_t flat = begin; flat < end; ++flat) {
        int64_t remainder = flat;
        const int64_t ow = remainder % out_width;
        remainder /= out_width;
        const int64_t oh = remainder % out_height;
        remainder /= out_height;
        const int64_t c = remainder % channels;
        const int64_t n = remainder / channels;
        const int64_t grid_base = 2 * (ow + out_width * (oh + out_height * n));
        const float gx = g[grid_base];
        const float gy = g[grid_base + 1];
        float ix = 0.0f;
        float iy = 0.0f;
        if (params->align_corners != 0) {
            ix = (gx + 1.0f) * static_cast<float>(width - 1) / 2.0f;
            iy = (gy + 1.0f) * static_cast<float>(height - 1) / 2.0f;
        } else {
            ix = ((gx + 1.0f) * static_cast<float>(width) - 1.0f) / 2.0f;
            iy = ((gy + 1.0f) * static_cast<float>(height) - 1.0f) / 2.0f;
        }
        const int64_t x0 = static_cast<int64_t>(std::floor(ix));
        const int64_t y0 = static_cast<int64_t>(std::floor(iy));
        const int64_t x1 = x0 + 1;
        const int64_t y1 = y0 + 1;
        const float dx = ix - static_cast<float>(x0);
        const float dy = iy - static_cast<float>(y0);
        const float value = sample(x0, y0, c, n) * (1.0f - dx) * (1.0f - dy) + sample(x1, y0, c, n) * dx * (1.0f - dy) +
                            sample(x0, y1, c, n) * (1.0f - dx) * dy + sample(x1, y1, c, n) * dx * dy;
        output[ow + out_width * (oh + out_height * (c + channels * n))] = value;
    }
}

void require_f32_binary(const struct ggml_tensor* a, const struct ggml_tensor* b, const char* op, int tag = 0) {
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        throw std::invalid_argument(
            std::string(op) + " requires F32 operands (got " + std::to_string(static_cast<int>(a->type)) + "," +
            std::to_string(static_cast<int>(b->type)) + " tag " + std::to_string(static_cast<int>(tag)) + ")");
    }
}

void erf_callback(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* /*userdata*/) {
    const int64_t count = ggml_nelements(src);
    const int64_t begin = count * ith / nth;
    const int64_t end = count * (ith + 1) / nth;
    const float* input = static_cast<const float*>(src->data);
    float* output = static_cast<float*>(dst->data);
    for (int64_t i = begin; i < end; ++i) {
        output[i] = std::erf(input[i]);
    }
}

void cast_i32_to_f32_callback(struct ggml_tensor* dst, const struct ggml_tensor* /*dummy*/,
                              const struct ggml_tensor* src, int ith, int nth, void* /*userdata*/) {
    const int64_t count = ggml_nelements(src);
    const int64_t begin = count * ith / nth;
    const int64_t end = count * (ith + 1) / nth;
    float* output = static_cast<float*>(dst->data);
    if (src->type == GGML_TYPE_I64) {
        const int64_t* input = static_cast<const int64_t*>(src->data);
        for (int64_t i = begin; i < end; ++i) {
            output[i] = static_cast<float>(input[i]);
        }
        return;
    }
    const int32_t* input = static_cast<const int32_t*>(src->data);
    for (int64_t i = begin; i < end; ++i) {
        output[i] = static_cast<float>(input[i]);
    }
}

void topk_values_callback(struct ggml_tensor* dst, const struct ggml_tensor* /*dummy*/, const struct ggml_tensor* input,
                          const struct ggml_tensor* indices, int ith, int nth, void* /*userdata*/) {
    const int64_t count = ggml_nelements(dst);
    const int64_t begin = count * ith / nth;
    const int64_t end = count * (ith + 1) / nth;
    const float* values = static_cast<const float*>(input->data);
    const int32_t* selected = static_cast<const int32_t*>(indices->data);
    float* output = static_cast<float*>(dst->data);
    for (int64_t flat = begin; flat < end; ++flat) {
        int64_t remainder = flat;
        const int64_t i0 = remainder % dst->ne[0];
        remainder /= dst->ne[0];
        const int64_t i1 = remainder % dst->ne[1];
        remainder /= dst->ne[1];
        const int64_t i2 = remainder % dst->ne[2];
        const int64_t i3 = remainder / dst->ne[2];
        const int32_t index = selected[i0 + dst->ne[0] * (i1 + dst->ne[1] * (i2 + dst->ne[2] * i3))];
        output[i0 + dst->ne[0] * (i1 + dst->ne[1] * (i2 + dst->ne[2] * i3))] =
            values[index + input->ne[0] * (i1 + input->ne[1] * (i2 + input->ne[2] * i3))];
    }
}

void gather_elements_axis1_callback(struct ggml_tensor* dst, const struct ggml_tensor* /*dummy*/,
                                    const struct ggml_tensor* data, const struct ggml_tensor* indices, int ith, int nth,
                                    void* userdata) {
    const int64_t count = ggml_nelements(dst);
    const int64_t begin = count * ith / nth;
    const int64_t end = count * (ith + 1) / nth;
    float* output = static_cast<float*>(dst->data);
    int* error = static_cast<int*>(userdata);
    for (int64_t flat = begin; flat < end; ++flat) {
        int64_t remainder = flat;
        const int64_t i0 = remainder % dst->ne[0];
        remainder /= dst->ne[0];
        const int64_t i1 = remainder % dst->ne[1];
        remainder /= dst->ne[1];
        const int64_t i2 = remainder % dst->ne[2];
        const int64_t i3 = remainder / dst->ne[2];
        const int32_t index =
            *reinterpret_cast<const int32_t*>(static_cast<const char*>(indices->data) + i0 * indices->nb[0] +
                                              i1 * indices->nb[1] + i2 * indices->nb[2] + i3 * indices->nb[3]);
        const int64_t output_index = i0 + dst->ne[0] * (i1 + dst->ne[1] * (i2 + dst->ne[2] * i3));
        if (index < 0 || index >= data->ne[1]) {
            *error = 1;
            output[output_index] = 0.0f;
            continue;
        }
        output[output_index] =
            *reinterpret_cast<const float*>(static_cast<const char*>(data->data) + i0 * data->nb[0] +
                                            index * data->nb[1] + i2 * data->nb[2] + i3 * data->nb[3]);
    }
}

} // namespace

struct Graph::Impl {
    struct ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    bool allocated = false;

    std::vector<struct ggml_tensor*> tensors;
    std::vector<DType> dtypes;
    std::vector<struct ggml_tensor*> inputs;
    std::vector<int> output_ids;
    std::vector<PendingF32> pending_f32;
    std::vector<PendingI32> pending_i32;
    std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> cast_i64_to_i32;
    int callback_error = 0;
    std::deque<GridSampleParams> grid_params;
    std::deque<IntegerBinaryParams> integer_binary_params;

    Impl() {
        ggml_init_params params{};
        params.mem_size = 1024 * 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("ggml_init failed");
        }
        backend = ggml_backend_cpu_init();
        if (backend == nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
            throw std::runtime_error("ggml_backend_cpu_init failed");
        }
    }

    ~Impl() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    struct ggml_tensor* new_tensor(DType dtype, const std::vector<int64_t>& shape) {
        if (shape.empty() || shape.size() > 4) {
            throw std::invalid_argument("tensor shape must have 1..4 dimensions");
        }
        std::vector<int64_t> dims = shape;
        return ggml_new_tensor(ctx, to_ggml_type(dtype), static_cast<int>(dims.size()), dims.data());
    }

    struct ggml_tensor* at(int id) const {
        if (id < 0 || static_cast<size_t>(id) >= tensors.size()) {
            throw std::invalid_argument("tensor id out of range: " + std::to_string(id));
        }
        return tensors[static_cast<size_t>(id)];
    }

    DType dtype_at(int id) const {
        if (id < 0 || static_cast<size_t>(id) >= dtypes.size()) {
            throw std::invalid_argument("tensor id out of range: " + std::to_string(id));
        }
        return dtypes[static_cast<size_t>(id)];
    }

    int push(struct ggml_tensor* tensor, DType dtype) {
        tensors.push_back(tensor);
        dtypes.push_back(dtype);
        return static_cast<int>(tensors.size()) - 1;
    }

    void require_inputs(const std::vector<int>& ids, size_t count, const char* op) const {
        if (ids.size() != count) {
            throw std::invalid_argument(std::string(op) + " expects " + std::to_string(count) + " inputs");
        }
        for (int id : ids) {
            at(id);
        }
    }

    void allocate_once() {
        if (allocated) {
            return;
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("GGML graph backend buffer allocation failed");
        }
        for (const auto& fill : pending_f32) {
            std::memcpy(fill.tensor->data, fill.data.data(), fill.data.size() * sizeof(float));
        }
        for (const auto& fill : pending_i32) {
            std::memcpy(fill.tensor->data, fill.data.data(), fill.data.size() * sizeof(int32_t));
        }
        allocated = true;
    }
};

Graph::Graph() : impl_(std::make_unique<Impl>()) {}
Graph::~Graph() = default;

int Graph::add_input(const std::string& /*name*/, DType dtype, std::vector<int64_t> shape) {
    struct ggml_tensor* tensor = impl_->new_tensor(dtype, shape);
    const int id = impl_->push(tensor, dtype);
    impl_->inputs.push_back(tensor);
    return id;
}

int Graph::add_const_f32(const std::string& /*name*/, std::vector<int64_t> shape, const std::vector<float>& data) {
    struct ggml_tensor* tensor = impl_->new_tensor(DType::F32, shape);
    const size_t expected = static_cast<size_t>(ggml_nelements(tensor));
    if (expected != data.size()) {
        throw std::invalid_argument("F32 constant data size mismatch");
    }
    impl_->pending_f32.push_back({tensor, data});
    return impl_->push(tensor, DType::F32);
}

int Graph::add_const_i32(const std::string& /*name*/, std::vector<int64_t> shape, const std::vector<int32_t>& data) {
    struct ggml_tensor* tensor = impl_->new_tensor(DType::I32, shape);
    const size_t expected = static_cast<size_t>(ggml_nelements(tensor));
    if (expected != data.size()) {
        throw std::invalid_argument("I32 constant data size mismatch");
    }
    impl_->pending_i32.push_back({tensor, data});
    return impl_->push(tensor, DType::I32);
}

int Graph::add_op(Op op, const std::vector<int>& inputs, const std::vector<float>& attrs) {
    struct ggml_tensor* result = nullptr;
    DType result_type = DType::F32;
    switch (op) {
    case Op::Add:
        impl_->require_inputs(inputs, 2, "Add");
        require_f32_binary(impl_->at(inputs[0]), impl_->at(inputs[1]), "Add");
        result = ggml_add(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]));
        break;
    case Op::Sub:
        impl_->require_inputs(inputs, 2, "Sub");
        require_f32_binary(impl_->at(inputs[0]), impl_->at(inputs[1]), "Sub");
        result = ggml_sub(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]));
        break;
    case Op::Mul:
        impl_->require_inputs(inputs, 2, "Mul");
        require_f32_binary(impl_->at(inputs[0]), impl_->at(inputs[1]), "Mul",
                           attrs.empty() ? 0 : static_cast<int>(attrs[0]));
        result = ggml_mul(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]));
        break;
    case Op::Div:
        impl_->require_inputs(inputs, 2, "Div");
        require_f32_binary(impl_->at(inputs[0]), impl_->at(inputs[1]), "Div");
        result = ggml_div(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]));
        break;
    case Op::IntegerBinary: {
        impl_->require_inputs(inputs, 2, "IntegerBinary");
        if (attrs.size() != 1) {
            throw std::invalid_argument("IntegerBinary expects an op-code attribute");
        }
        struct ggml_tensor* left = impl_->at(inputs[0]);
        struct ggml_tensor* right = impl_->at(inputs[1]);
        if ((left->type != GGML_TYPE_I32 && left->type != GGML_TYPE_I64) ||
            (right->type != GGML_TYPE_I32 && right->type != GGML_TYPE_I64)) {
            throw std::invalid_argument("IntegerBinary requires I32 or I64 inputs");
        }
        if (!ggml_is_contiguous(left)) {
            left = ggml_cont(impl_->ctx, left);
        }
        if (!ggml_is_contiguous(right)) {
            right = ggml_cont(impl_->ctx, right);
        }
        impl_->integer_binary_params.push_back({static_cast<int>(attrs[0]), 0});
        result = ggml_map_custom2(impl_->ctx, left, right, integer_binary_callback, GGML_N_TASKS_MAX,
                                  &impl_->integer_binary_params.back());
        result_type = impl_->dtype_at(inputs[0]);
        break;
    }
    case Op::Sqrt:
        impl_->require_inputs(inputs, 1, "Sqrt");
        result = ggml_sqrt(impl_->ctx, impl_->at(inputs[0]));
        break;
    case Op::Clip: {
        impl_->require_inputs(inputs, 1, "Clip");
        if (attrs.size() != 2) {
            throw std::invalid_argument("Clip expects min and max attributes");
        }
        result = ggml_clamp(impl_->ctx, impl_->at(inputs[0]), attrs[0], attrs[1]);
        break;
    }
    case Op::MatMul:
        impl_->require_inputs(inputs, 2, "MatMul");
        result = ggml_ops::matmul(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]));
        break;
    case Op::BatchedMatMul: {
        impl_->require_inputs(inputs, 2, "BatchedMatMul");
        // Inputs are {B, A} so ggml broadcasts the unbatched operand (B) over
        // A's batch dimensions. The result is already in ONNX layout.
        struct ggml_tensor* a = impl_->at(inputs[0]);
        struct ggml_tensor* b = impl_->at(inputs[1]);
        if (a->ne[0] != b->ne[0] || b->ne[2] % a->ne[2] != 0 || b->ne[3] % a->ne[3] != 0) {
            throw std::invalid_argument("BatchedMatMul operands are not broadcast-compatible: a=[" +
                                        std::to_string(a->ne[0]) + "," + std::to_string(a->ne[1]) + "," +
                                        std::to_string(a->ne[2]) + "," + std::to_string(a->ne[3]) + "] b=[" +
                                        std::to_string(b->ne[0]) + "," + std::to_string(b->ne[1]) + "," +
                                        std::to_string(b->ne[2]) + "," + std::to_string(b->ne[3]) + "]");
        }
        result = ggml_mul_mat(impl_->ctx, a, b);
        break;
    }
    case Op::BatchedMatMulTransposed: {
        impl_->require_inputs(inputs, 2, "BatchedMatMulTransposed");
        // Inputs are {A, B}; ggml broadcasts A over B's batch dimensions.
        struct ggml_tensor* a = impl_->at(inputs[0]);
        struct ggml_tensor* b = impl_->at(inputs[1]);
        if (a->ne[0] != b->ne[0] || b->ne[2] % a->ne[2] != 0 || b->ne[3] % a->ne[3] != 0) {
            throw std::invalid_argument("BatchedMatMulTransposed operands are not broadcast-compatible");
        }
        struct ggml_tensor* product = ggml_mul_mat(impl_->ctx, a, b);
        result = ggml_cont(impl_->ctx, ggml_permute(impl_->ctx, product, 1, 0, 2, 3));
        break;
    }
    case Op::Linear:
        impl_->require_inputs(inputs, 3, "Linear");
        require_f32_binary(impl_->at(inputs[0]), impl_->at(inputs[1]), "Linear input/weight");
        require_f32_binary(impl_->at(inputs[1]), impl_->at(inputs[2]), "Linear weight/bias");
        result = ggml_ops::linear(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]), impl_->at(inputs[2]));
        break;
    case Op::ScaledDotProductAttention:
        impl_->require_inputs(inputs, 3, "ScaledDotProductAttention");
        result = ggml_ops::scaled_dot_product_attention(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]),
                                                        impl_->at(inputs[2]));
        break;
    case Op::Softmax:
        impl_->require_inputs(inputs, 1, "Softmax");
        result = ggml_soft_max(impl_->ctx, impl_->at(inputs[0]));
        break;
    case Op::LayerNorm: {
        impl_->require_inputs(inputs, 3, "LayerNorm");
        const float eps = attrs.empty() ? 1e-5f : attrs[0];
        require_f32_binary(impl_->at(inputs[0]), impl_->at(inputs[1]), "LayerNorm input/weight");
        require_f32_binary(impl_->at(inputs[1]), impl_->at(inputs[2]), "LayerNorm weight/bias");
        result =
            ggml_ops::layer_norm(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]), impl_->at(inputs[2]), eps);
        break;
    }
    case Op::Gelu:
        impl_->require_inputs(inputs, 1, "Gelu");
        result = ggml_ops::gelu(impl_->ctx, impl_->at(inputs[0]));
        break;
    case Op::Gather:
        impl_->require_inputs(inputs, 2, "Gather");
        if (impl_->dtype_at(inputs[1]) != DType::I32) {
            throw std::invalid_argument("Gather requires I32 indices");
        }
        result = ggml_ops::gather_rows(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]));
        result_type = impl_->dtype_at(inputs[0]);
        break;
    case Op::Reshape: {
        impl_->require_inputs(inputs, 1, "Reshape");
        if (attrs.empty() || attrs.size() > 4) {
            throw std::invalid_argument("Reshape expects 1..4 target dimensions");
        }
        int64_t dims[4] = {1, 1, 1, 1};
        for (size_t i = 0; i < attrs.size(); ++i) {
            const int64_t dim = static_cast<int64_t>(attrs[i]);
            if (dim <= 0) {
                throw std::invalid_argument("Reshape dimensions must be positive");
            }
            dims[i] = dim;
        }
        struct ggml_tensor* src = impl_->at(inputs[0]);
        if (!ggml_is_contiguous(src)) {
            src = ggml_cont(impl_->ctx, src);
        }
        result = ggml_reshape_4d(impl_->ctx, src, dims[0], dims[1], dims[2], dims[3]);
        result_type = impl_->dtype_at(inputs[0]);
        break;
    }
    case Op::Transpose: {
        impl_->require_inputs(inputs, 1, "Transpose");
        if (attrs.size() != 4) {
            throw std::invalid_argument("Transpose expects four axis attributes");
        }
        result = ggml_cont(impl_->ctx, ggml_permute(impl_->ctx, impl_->at(inputs[0]), static_cast<int>(attrs[0]),
                                                    static_cast<int>(attrs[1]), static_cast<int>(attrs[2]),
                                                    static_cast<int>(attrs[3])));
        result_type = impl_->dtype_at(inputs[0]);
        break;
    }
    case Op::Concat: {
        if (inputs.size() < 2) {
            throw std::invalid_argument("Concat expects at least two inputs");
        }
        if (attrs.size() != 1 || attrs[0] < 0 || attrs[0] > 3) {
            throw std::invalid_argument("Concat expects one axis attribute in 0..3");
        }
        result = impl_->at(inputs[0]);
        const int axis = static_cast<int>(attrs[0]);
        for (size_t i = 1; i < inputs.size(); ++i) {
            result = ggml_concat(impl_->ctx, result, impl_->at(inputs[i]), axis);
        }
        result_type = impl_->dtype_at(inputs[0]);
        break;
    }
    case Op::Relu:
        impl_->require_inputs(inputs, 1, "Relu");
        result = ggml_relu(impl_->ctx, impl_->at(inputs[0]));
        break;
    case Op::Sigmoid:
        impl_->require_inputs(inputs, 1, "Sigmoid");
        result = ggml_sigmoid(impl_->ctx, impl_->at(inputs[0]));
        break;
    case Op::Abs:
        impl_->require_inputs(inputs, 1, "Abs");
        result = ggml_abs(impl_->ctx, impl_->at(inputs[0]));
        break;
    case Op::Log:
        impl_->require_inputs(inputs, 1, "Log");
        result = ggml_log(impl_->ctx, impl_->at(inputs[0]));
        break;
    case Op::Erf: {
        impl_->require_inputs(inputs, 1, "Erf");
        struct ggml_tensor* input = impl_->at(inputs[0]);
        if (input->type != GGML_TYPE_F32) {
            throw std::invalid_argument("Erf requires F32 input");
        }
        if (!ggml_is_contiguous(input)) {
            input = ggml_cont(impl_->ctx, input);
        }
        result = ggml_map_custom1(impl_->ctx, input, erf_callback, GGML_N_TASKS_MAX, nullptr);
        break;
    }
    case Op::Square:
        impl_->require_inputs(inputs, 1, "Square");
        if (impl_->at(inputs[0])->type != GGML_TYPE_F32) {
            throw std::invalid_argument("Square requires an F32 input");
        }
        result = ggml_mul(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[0]));
        break;
    case Op::ReduceMean:
        impl_->require_inputs(inputs, 1, "ReduceMean");
        result = ggml_mean(impl_->ctx, impl_->at(inputs[0]));
        break;
    case Op::ReduceSum:
        impl_->require_inputs(inputs, 1, "ReduceSum");
        result = ggml_sum_rows(impl_->ctx, impl_->at(inputs[0]));
        break;
    case Op::ReduceMax: {
        impl_->require_inputs(inputs, 1, "ReduceMax");
        struct ggml_tensor* input = impl_->at(inputs[0]);
        result = ggml_pool_2d(impl_->ctx, input, GGML_OP_POOL_MAX, static_cast<int>(input->ne[0]), 1,
                              static_cast<int>(input->ne[0]), 1, 0.0f, 0.0f);
        break;
    }
    case Op::TopKIndices: {
        impl_->require_inputs(inputs, 1, "TopKIndices");
        if (attrs.size() != 1) {
            throw std::invalid_argument("TopKIndices expects a k attribute");
        }
        result = ggml_top_k(impl_->ctx, impl_->at(inputs[0]), static_cast<int>(attrs[0]));
        result_type = DType::I32;
        break;
    }
    case Op::TopKValues: {
        impl_->require_inputs(inputs, 1, "TopKValues");
        if (attrs.size() != 1) {
            throw std::invalid_argument("TopKValues expects a k attribute");
        }
        struct ggml_tensor* input = impl_->at(inputs[0]);
        if (!ggml_is_contiguous(input)) {
            input = ggml_cont(impl_->ctx, input);
        }
        const int k = static_cast<int>(attrs[0]);
        struct ggml_tensor* indices = ggml_top_k(impl_->ctx, input, k);
        if (!ggml_is_contiguous(indices)) {
            indices = ggml_cont(impl_->ctx, indices);
        }
        struct ggml_tensor* dummy = ggml_new_tensor_4d(impl_->ctx, GGML_TYPE_F32, indices->ne[0], indices->ne[1],
                                                       indices->ne[2], indices->ne[3]);
        result = ggml_map_custom3(impl_->ctx, dummy, input, indices, topk_values_callback, GGML_N_TASKS_MAX, nullptr);
        break;
    }
    case Op::GatherElements: {
        impl_->require_inputs(inputs, 2, "GatherElements");
        struct ggml_tensor* data = impl_->at(inputs[0]);
        struct ggml_tensor* indices = impl_->at(inputs[1]);
        if (data->type != GGML_TYPE_F32 || indices->type != GGML_TYPE_I32) {
            throw std::invalid_argument("GatherElements requires F32 data and I32 indices");
        }
        if (data->ne[0] != indices->ne[0] || data->ne[2] != indices->ne[2] || data->ne[3] != indices->ne[3]) {
            throw std::invalid_argument("GatherElements shapes do not match");
        }
        if (!ggml_is_contiguous(data)) {
            data = ggml_cont(impl_->ctx, data);
        }
        if (!ggml_is_contiguous(indices)) {
            indices = ggml_cont(impl_->ctx, indices);
        }
        struct ggml_tensor* dummy = ggml_new_tensor_4d(impl_->ctx, GGML_TYPE_F32, indices->ne[0], indices->ne[1],
                                                       indices->ne[2], indices->ne[3]);
        result = ggml_map_custom3(impl_->ctx, dummy, data, indices, gather_elements_axis1_callback, GGML_N_TASKS_MAX,
                                  &impl_->callback_error);
        break;
    }
    case Op::GridSample: {
        impl_->require_inputs(inputs, 2, "GridSample");
        if (attrs.size() != 1) {
            throw std::invalid_argument("GridSample expects an align_corners attribute");
        }
        struct ggml_tensor* input = impl_->at(inputs[0]);
        struct ggml_tensor* grid = impl_->at(inputs[1]);
        if (input->type != GGML_TYPE_F32 || grid->type != GGML_TYPE_F32) {
            throw std::invalid_argument("GridSample requires F32 input and grid");
        }
        if (!ggml_is_contiguous(input)) {
            input = ggml_cont(impl_->ctx, input);
        }
        if (!ggml_is_contiguous(grid)) {
            grid = ggml_cont(impl_->ctx, grid);
        }
        if (grid->ne[0] != 2 || grid->ne[3] != input->ne[3]) {
            throw std::invalid_argument("GridSample grid shape does not match the input");
        }
        struct ggml_tensor* dummy =
            ggml_new_tensor_4d(impl_->ctx, GGML_TYPE_F32, grid->ne[1], grid->ne[2], input->ne[2], input->ne[3]);
        impl_->grid_params.push_back({static_cast<int>(attrs[0]), 0});
        result = ggml_map_custom3(impl_->ctx, dummy, input, grid, grid_sample_bilinear_callback, GGML_N_TASKS_MAX,
                                  &impl_->grid_params.back());
        break;
    }
    case Op::Pad: {
        impl_->require_inputs(inputs, 1, "Pad");
        if (attrs.size() != 4) {
            throw std::invalid_argument("Pad expects four end-padding attributes");
        }
        result = ggml_pad(impl_->ctx, impl_->at(inputs[0]), static_cast<int>(attrs[0]), static_cast<int>(attrs[1]),
                          static_cast<int>(attrs[2]), static_cast<int>(attrs[3]));
        result_type = impl_->dtype_at(inputs[0]);
        break;
    }
    case Op::Interpolate: {
        impl_->require_inputs(inputs, 1, "Interpolate");
        if (attrs.size() != 4) {
            throw std::invalid_argument("Interpolate expects four target dimensions");
        }
        result = ggml_interpolate(impl_->ctx, impl_->at(inputs[0]), static_cast<int64_t>(attrs[0]),
                                  static_cast<int64_t>(attrs[1]), static_cast<int64_t>(attrs[2]),
                                  static_cast<int64_t>(attrs[3]), GGML_SCALE_MODE_NEAREST);
        break;
    }
    case Op::Repeat: {
        impl_->require_inputs(inputs, 1, "Repeat");
        if (attrs.size() != 4) {
            throw std::invalid_argument("Repeat expects four target dimensions");
        }
        struct ggml_tensor* src = impl_->at(inputs[0]);
        if (src->type == GGML_TYPE_I64) {
            throw std::invalid_argument("Repeat does not support I64; narrow to I32 first");
        }
        result = ggml_repeat_4d(impl_->ctx, src, static_cast<int64_t>(attrs[0]), static_cast<int64_t>(attrs[1]),
                                static_cast<int64_t>(attrs[2]), static_cast<int64_t>(attrs[3]));
        result_type = impl_->dtype_at(inputs[0]);
        break;
    }
    case Op::Slice: {
        impl_->require_inputs(inputs, 1, "Slice");
        if (attrs.size() != 8) {
            throw std::invalid_argument("Slice expects four start/size attribute pairs");
        }
        struct ggml_tensor* src = impl_->at(inputs[0]);
        if (!ggml_is_contiguous(src)) {
            src = ggml_cont(impl_->ctx, src);
        }
        int64_t starts[4] = {0, 0, 0, 0};
        int64_t sizes[4] = {1, 1, 1, 1};
        for (size_t i = 0; i < 4; ++i) {
            starts[i] = static_cast<int64_t>(attrs[2 * i]);
            sizes[i] = static_cast<int64_t>(attrs[2 * i + 1]);
            if (starts[i] < 0 || sizes[i] <= 0 || starts[i] + sizes[i] > src->ne[i]) {
                throw std::invalid_argument("Slice range is outside the source tensor");
            }
        }
        const size_t type_size = ggml_type_size(src->type);
        const size_t offset =
            static_cast<size_t>(starts[0] + starts[1] * src->ne[0] + starts[2] * src->ne[0] * src->ne[1] +
                                starts[3] * src->ne[0] * src->ne[1] * src->ne[2]) *
            type_size;
        struct ggml_tensor* view = ggml_view_4d(impl_->ctx, src, sizes[0], sizes[1], sizes[2], sizes[3], src->nb[1],
                                                src->nb[2], src->nb[3], offset);
        result = ggml_cont(impl_->ctx, view);
        result_type = impl_->dtype_at(inputs[0]);
        break;
    }
    case Op::Conv2d:
    case Op::Conv2dDepthwise: {
        impl_->require_inputs(inputs, 3, op == Op::Conv2d ? "Conv2d" : "Conv2dDepthwise");
        if (attrs.size() != 6) {
            throw std::invalid_argument("Conv2d expects stride/pad/dilation attributes");
        }
        struct ggml_tensor* input = impl_->at(inputs[0]);
        struct ggml_tensor* weight = impl_->at(inputs[1]);
        struct ggml_tensor* bias = impl_->at(inputs[2]);
        const int stride_w = static_cast<int>(attrs[0]);
        const int stride_h = static_cast<int>(attrs[1]);
        const int pad_w = static_cast<int>(attrs[2]);
        const int pad_h = static_cast<int>(attrs[3]);
        const int dilation_w = static_cast<int>(attrs[4]);
        const int dilation_h = static_cast<int>(attrs[5]);
        if (op == Op::Conv2dDepthwise) {
            result = ggml_conv_2d_dw_direct(impl_->ctx, weight, input, stride_w, stride_h, pad_w, pad_h, dilation_w,
                                            dilation_h);
        } else {
            result = ggml_conv_2d(impl_->ctx, weight, input, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h);
        }
        if (bias->ne[0] != result->ne[2]) {
            throw std::invalid_argument("Conv2d bias length must match output channels");
        }
        struct ggml_tensor* bias_3d = ggml_reshape_3d(impl_->ctx, bias, 1, 1, bias->ne[0]);
        result = ggml_add(impl_->ctx, result, bias_3d);
        break;
    }
    case Op::MaxPool2d: {
        impl_->require_inputs(inputs, 1, "MaxPool2d");
        if (attrs.size() != 6) {
            throw std::invalid_argument("MaxPool2d expects kernel/stride/pad attributes");
        }
        result = ggml_pool_2d(impl_->ctx, impl_->at(inputs[0]), GGML_OP_POOL_MAX, static_cast<int>(attrs[0]),
                              static_cast<int>(attrs[1]), static_cast<int>(attrs[2]), static_cast<int>(attrs[3]),
                              attrs[4], attrs[5]);
        break;
    }
    case Op::CastI64ToI32: {
        impl_->require_inputs(inputs, 1, "CastI64ToI32");
        if (impl_->dtype_at(inputs[0]) != DType::I64) {
            throw std::invalid_argument("CastI64ToI32 requires an I64 input");
        }
        // ggml's cast/dup kernels do not support I64->I32 on this backend;
        // materialize the I32 tensor on the host after inputs are copied.
        struct ggml_tensor* src = impl_->at(inputs[0]);
        result = ggml_new_tensor(impl_->ctx, GGML_TYPE_I32, GGML_MAX_DIMS, src->ne);
        impl_->cast_i64_to_i32.emplace_back(src, result);
        result_type = DType::I32;
        break;
    }
    case Op::CastI32ToF32: {
        impl_->require_inputs(inputs, 1, "CastI32ToF32");
        struct ggml_tensor* src = impl_->at(inputs[0]);
        if (src->type != GGML_TYPE_I32 && src->type != GGML_TYPE_I64) {
            throw std::invalid_argument("CastI32ToF32 requires an I32 or I64 input");
        }
        if (!ggml_is_contiguous(src)) {
            src = ggml_cont(impl_->ctx, src);
        }
        struct ggml_tensor* dummy =
            ggml_new_tensor_4d(impl_->ctx, GGML_TYPE_F32, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        result = ggml_map_custom2(impl_->ctx, dummy, src, cast_i32_to_f32_callback, GGML_N_TASKS_MAX, nullptr);
        break;
    }
    }
    if (result == nullptr) {
        throw std::runtime_error("GGML graph op produced no tensor");
    }
    return impl_->push(result, result_type);
}

void Graph::mark_output(int tensor_id) {
    impl_->at(tensor_id);
    impl_->output_ids.push_back(tensor_id);
}

DType Graph::dtype(int tensor_id) const { return impl_->dtype_at(tensor_id); }

std::vector<Output> Graph::compute(const std::vector<std::vector<uint8_t>>& input_blobs) {
    if (impl_->inputs.size() != input_blobs.size()) {
        throw std::invalid_argument("input blob count mismatch");
    }
    for (size_t i = 0; i < impl_->inputs.size(); ++i) {
        if (input_blobs[i].size() != ggml_nbytes(impl_->inputs[i])) {
            throw std::invalid_argument("input blob size mismatch for input " + std::to_string(i));
        }
    }
    if (impl_->output_ids.empty()) {
        throw std::invalid_argument("graph has no marked outputs");
    }
    impl_->callback_error = 0;
    for (auto& params : impl_->grid_params) {
        params.error = 0;
    }
    for (auto& params : impl_->integer_binary_params) {
        params.error = 0;
    }
    impl_->allocate_once();
    for (size_t i = 0; i < impl_->inputs.size(); ++i) {
        std::memcpy(impl_->inputs[i]->data, input_blobs[i].data(), input_blobs[i].size());
    }
    for (const auto& cast : impl_->cast_i64_to_i32) {
        const auto* src = static_cast<const int64_t*>(cast.first->data);
        auto* dst = static_cast<int32_t*>(cast.second->data);
        const auto count = static_cast<size_t>(ggml_nelements(cast.first));
        for (size_t i = 0; i < count; ++i) {
            dst[i] = static_cast<int32_t>(src[i]);
        }
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(impl_->ctx, 200000, false);
    for (int id : impl_->output_ids) {
        ggml_build_forward_expand(graph, impl_->at(id));
    }
    if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML graph computation failed");
    }
    if (impl_->callback_error != 0) {
        throw std::runtime_error("GGML graph callback failed");
    }
    for (const auto& params : impl_->grid_params) {
        if (params.error != 0) {
            throw std::runtime_error("GGML GridSample callback failed");
        }
    }
    for (const auto& params : impl_->integer_binary_params) {
        if (params.error != 0) {
            throw std::runtime_error("GGML integer binary callback failed");
        }
    }

    std::vector<Output> outputs;
    outputs.reserve(impl_->output_ids.size());
    for (int id : impl_->output_ids) {
        struct ggml_tensor* tensor = impl_->at(id);
        Output output;
        output.dtype = from_ggml_type(tensor->type);
        output.shape.assign(tensor->ne, tensor->ne + 4);
        while (!output.shape.empty() && output.shape.back() == 1) {
            output.shape.pop_back();
        }
        if (output.shape.empty()) {
            output.shape.push_back(1);
        }
        output.data.resize(static_cast<size_t>(ggml_nbytes(tensor)));
        std::memcpy(output.data.data(), tensor->data, output.data.size());
        outputs.push_back(std::move(output));
    }
    return outputs;
}

} // namespace ggml_graph
