#include "GGMLGraph.hpp"

#include "GGMLOps.hpp"

#include <cstring>
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
        result = ggml_add(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]));
        break;
    case Op::Sub:
        impl_->require_inputs(inputs, 2, "Sub");
        result = ggml_sub(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]));
        break;
    case Op::Mul:
        impl_->require_inputs(inputs, 2, "Mul");
        result = ggml_mul(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]));
        break;
    case Op::Div:
        impl_->require_inputs(inputs, 2, "Div");
        result = ggml_div(impl_->ctx, impl_->at(inputs[0]), impl_->at(inputs[1]));
        break;
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
    case Op::Linear:
        impl_->require_inputs(inputs, 3, "Linear");
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

    struct ggml_cgraph* graph = ggml_new_graph(impl_->ctx);
    for (int id : impl_->output_ids) {
        ggml_build_forward_expand(graph, impl_->at(id));
    }
    if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML graph computation failed");
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
