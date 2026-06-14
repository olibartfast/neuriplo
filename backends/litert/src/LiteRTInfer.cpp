#include "LiteRTInfer.hpp"

#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/kernel_util.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

namespace {

std::string tensor_type_name(TfLiteType type) {
    switch (type) {
    case kTfLiteFloat32:
        return "float32";
    case kTfLiteUInt8:
        return "uint8";
    case kTfLiteInt32:
        return "int32";
    case kTfLiteInt64:
        return "int64";
    default:
        return "unsupported";
    }
}

TensorDataType neuriplo_dtype_from_tflite(TfLiteType type) {
    switch (type) {
    case kTfLiteFloat32:
        return TensorDataType::Float32;
    case kTfLiteInt32:
        return TensorDataType::Int32;
    case kTfLiteInt64:
        return TensorDataType::Int64;
    case kTfLiteUInt8:
        return TensorDataType::UInt8;
    case kTfLiteInt8:
        return TensorDataType::Int8;
    case kTfLiteBool:
        return TensorDataType::Bool;
    default:
        // Unknown/unsupported element types fall back to Float32, matching the
        // previous behaviour where no datatype was reported at all.
        return TensorDataType::Float32;
    }
}

std::vector<int64_t> shape_from_dims(const TfLiteIntArray* dims) {
    std::vector<int64_t> shape;
    if (dims == nullptr) {
        return shape;
    }
    shape.reserve(static_cast<size_t>(dims->size));
    for (int i = 0; i < dims->size; ++i) {
        shape.push_back(dims->data[i]);
    }
    return shape;
}

template <typename T> void append_tensor_data(std::vector<TensorElement>& output, const T* data, size_t count) {
    output.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        output.emplace_back(data[i]);
    }
}

size_t element_count_from_shape(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, [](size_t total, int64_t dim) {
        if (dim <= 0) {
            throw InferenceExecutionException("Tensor has invalid runtime dimension: " + std::to_string(dim));
        }
        return total * static_cast<size_t>(dim);
    });
}

bool is_nhwc_model_input(const TfLiteTensor* tensor) {
    if (tensor == nullptr || tensor->dims == nullptr || tensor->dims->size != 4) {
        return false;
    }
    const int c_dim = tensor->dims->data[3];
    const int h_dim = tensor->dims->data[1];
    return (c_dim == 1 || c_dim == 3) && h_dim > 3;
}

void transpose_nchw_to_nhwc(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int batch, int channels,
                            int height, int width) {
    const auto* src_ptr = reinterpret_cast<const float*>(src.data());
    auto* dst_ptr = reinterpret_cast<float*>(dst.data());

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    const int nchw_idx = ((b * channels + c) * height + h) * width + w;
                    const int nhwc_idx = ((b * height + h) * width + w) * channels + c;
                    dst_ptr[nhwc_idx] = src_ptr[nchw_idx];
                }
            }
        }
    }
}

// Custom kernel for the "ONNX_GRIDSAMPLE" operator emitted by onnx2tf's
// flatbuffer_direct backend (used for ONNX GridSample, e.g. the deformable
// attention in RT-DETR / D-FINE style detectors). onnx2tf keeps this op in
// ONNX-native NCHW layout and encodes no custom options, so we implement the
// ONNX/PyTorch default semantics: bilinear interpolation, padding_mode=zeros,
// align_corners=false.
//   input : [Ni, C, H,  W ]   (NCHW feature map)
//   grid  : [Ng, Hg, Wg, 2]   (last dim = (x, y), normalized to [-1, 1])
//   output: [Ng, C, Hg, Wg]
// The feature batch is broadcast when Ni == 1 (the deformable-attention value
// is shared across the sampling batch).
TfLiteStatus GridSamplePrepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
    TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);
    const TfLiteTensor* input = tflite::GetInput(context, node, 0);
    const TfLiteTensor* grid = tflite::GetInput(context, node, 1);
    TfLiteTensor* output = tflite::GetOutput(context, node, 0);
    TF_LITE_ENSURE(context, input != nullptr && grid != nullptr && output != nullptr);
    TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(input), 4);
    TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(grid), 4);
    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, grid->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, grid->dims->data[3], 2);

    TfLiteIntArray* out_dims = TfLiteIntArrayCreate(4);
    out_dims->data[0] = grid->dims->data[0];  // Ng
    out_dims->data[1] = input->dims->data[1]; // C
    out_dims->data[2] = grid->dims->data[1];  // Hg
    out_dims->data[3] = grid->dims->data[2];  // Wg
    return context->ResizeTensor(context, output, out_dims);
}

TfLiteStatus GridSampleEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = tflite::GetInput(context, node, 0);
    const TfLiteTensor* grid = tflite::GetInput(context, node, 1);
    TfLiteTensor* output = tflite::GetOutput(context, node, 0);

    const int Ni = input->dims->data[0];
    const int C = input->dims->data[1];
    const int H = input->dims->data[2];
    const int W = input->dims->data[3];
    const int Ng = grid->dims->data[0];
    const int Hg = grid->dims->data[1];
    const int Wg = grid->dims->data[2];

    const float* in = input->data.f;
    const float* g = grid->data.f;
    float* out = output->data.f;

    const auto in_index = [&](int n, int c, int y, int x) { return ((n * C + c) * H + y) * W + x; };

    for (int n = 0; n < Ng; ++n) {
        const int in_n = (Ni == 1) ? 0 : (n < Ni ? n : Ni - 1);
        for (int hy = 0; hy < Hg; ++hy) {
            for (int wx = 0; wx < Wg; ++wx) {
                const int gbase = ((n * Hg + hy) * Wg + wx) * 2;
                const float gx = g[gbase + 0];
                const float gy = g[gbase + 1];
                // align_corners=false unnormalization to pixel coordinates.
                const float ix = ((gx + 1.0f) * static_cast<float>(W) - 1.0f) * 0.5f;
                const float iy = ((gy + 1.0f) * static_cast<float>(H) - 1.0f) * 0.5f;
                const int x0 = static_cast<int>(std::floor(ix));
                const int y0 = static_cast<int>(std::floor(iy));
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;
                const float wx1 = ix - static_cast<float>(x0);
                const float wy1 = iy - static_cast<float>(y0);
                const float wx0 = 1.0f - wx1;
                const float wy0 = 1.0f - wy1;
                const bool x0_in = x0 >= 0 && x0 < W;
                const bool x1_in = x1 >= 0 && x1 < W;
                const bool y0_in = y0 >= 0 && y0 < H;
                const bool y1_in = y1 >= 0 && y1 < H;
                for (int c = 0; c < C; ++c) {
                    float v = 0.0f;
                    if (y0_in && x0_in)
                        v += wx0 * wy0 * in[in_index(in_n, c, y0, x0)];
                    if (y0_in && x1_in)
                        v += wx1 * wy0 * in[in_index(in_n, c, y0, x1)];
                    if (y1_in && x0_in)
                        v += wx0 * wy1 * in[in_index(in_n, c, y1, x0)];
                    if (y1_in && x1_in)
                        v += wx1 * wy1 * in[in_index(in_n, c, y1, x1)];
                    out[((n * C + c) * Hg + hy) * Wg + wx] = v;
                }
            }
        }
    }
    return kTfLiteOk;
}

const TfLiteRegistration* Register_ONNX_GRIDSAMPLE() {
    static TfLiteRegistration registration = [] {
        TfLiteRegistration r{};
        r.prepare = GridSamplePrepare;
        r.invoke = GridSampleEval;
        r.custom_name = "ONNX_GRIDSAMPLE";
        return r;
    }();
    return &registration;
}

// Override for the builtin SIGN operator. The stock TFLite kernel only supports
// floating-point tensors, but onnx2tf emits SIGN on integer coordinate math
// (e.g. derived from the INT64 orig_target_sizes input of RT-DETR/D-FINE
// detectors). This drop-in replacement keeps the float behaviour and adds
// INT32/INT64 support: sign(x) = (x > 0) - (x < 0).
template <typename T> void apply_sign(const T* in, T* out, int count) {
    for (int i = 0; i < count; ++i) {
        const T v = in[i];
        out[i] = static_cast<T>((v > T(0)) ? 1 : ((v < T(0)) ? -1 : 0));
    }
}

TfLiteStatus SignPrepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
    TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);
    const TfLiteTensor* input = tflite::GetInput(context, node, 0);
    TfLiteTensor* output = tflite::GetOutput(context, node, 0);
    TF_LITE_ENSURE(context, input != nullptr && output != nullptr);
    output->type = input->type;
    return context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus SignEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = tflite::GetInput(context, node, 0);
    TfLiteTensor* output = tflite::GetOutput(context, node, 0);
    const int count = tflite::NumElements(input);
    switch (input->type) {
    case kTfLiteFloat32:
        apply_sign(input->data.f, output->data.f, count);
        return kTfLiteOk;
    case kTfLiteInt32:
        apply_sign(input->data.i32, output->data.i32, count);
        return kTfLiteOk;
    case kTfLiteInt64:
        apply_sign(input->data.i64, output->data.i64, count);
        return kTfLiteOk;
    default:
        TF_LITE_KERNEL_LOG(context, "SIGN: unsupported input type %d", input->type);
        return kTfLiteError;
    }
}

const TfLiteRegistration* Register_SIGN_WITH_INTEGERS() {
    static TfLiteRegistration registration = [] {
        TfLiteRegistration r{};
        r.prepare = SignPrepare;
        r.invoke = SignEval;
        r.builtin_code = kTfLiteBuiltinSign;
        return r;
    }();
    return &registration;
}

} // namespace

LiteRTInfer::LiteRTInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                         const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
    if (use_gpu) {
        LOG(WARNING) << "LiteRT backend currently uses the CPU interpreter; GPU/delegate selection is not wired";
    }

    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_) {
        throw ModelLoadException("Unable to load LiteRT flatbuffer: " + model_path);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    // onnx2tf lowers ONNX GridSample to this custom op; register our kernel so
    // models that use it (e.g. RT-DETR/D-FINE deformable attention) run on the
    // builtin interpreter.
    resolver.AddCustom("ONNX_GRIDSAMPLE", Register_ONNX_GRIDSAMPLE());
    // Replace the builtin SIGN kernel with one that also handles integer inputs
    // (the stock kernel is float-only and rejects the INT64 sign math in these
    // models).
    resolver.AddBuiltin(static_cast<tflite::BuiltinOperator>(kTfLiteBuiltinSign), Register_SIGN_WITH_INTEGERS(), 1, 2);
    tflite::InterpreterBuilder builder(*model_, resolver);
    if (builder(&interpreter_) != kTfLiteOk || !interpreter_) {
        throw ModelLoadException("Unable to create LiteRT interpreter for: " + model_path);
    }

    const auto& inputs = interpreter_->inputs();
    for (size_t i = 0; i < input_sizes.size() && i < inputs.size(); ++i) {
        const std::vector<int> dims = makeInputDims(inputs[i], input_sizes[i]);
        if (interpreter_->ResizeInputTensor(inputs[i], dims) != kTfLiteOk) {
            throw ModelLoadException("Unable to resize LiteRT input tensor at index " + std::to_string(i));
        }
    }

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        throw ModelLoadException("Unable to allocate LiteRT tensors for: " + model_path);
    }

    refreshMetadata();

    state_ = BackendState::Ready;
}

LiteRTInfer::~LiteRTInfer() = default;

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
LiteRTInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    validate_input(input_tensors);

    const auto& input_indices = interpreter_->inputs();
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        TfLiteTensor* input = interpreter_->tensor(input_indices[i]);
        if (input == nullptr) {
            throw InferenceExecutionException("LiteRT input tensor is null at index " + std::to_string(i));
        }

        if (is_nhwc_model_input(input)) {
            const int batch = input->dims->data[0];
            const int height = input->dims->data[1];
            const int width = input->dims->data[2];
            const int channels = input->dims->data[3];

            std::vector<uint8_t> nhwc_buffer(input->bytes);
            transpose_nchw_to_nhwc(input_tensors[i], nhwc_buffer, batch, channels, height, width);
            std::memcpy(input->data.raw, nhwc_buffer.data(), input->bytes);
        } else {
            if (input_tensors[i].size() != input->bytes) {
                throw InferenceExecutionException("LiteRT input tensor byte size mismatch at index " +
                                                  std::to_string(i) + ": expected " + std::to_string(input->bytes) +
                                                  ", got " + std::to_string(input_tensors[i].size()));
            }
            std::memcpy(input->data.raw, input_tensors[i].data(), input->bytes);
        }
    }

    start_timer();
    if (interpreter_->Invoke() != kTfLiteOk) {
        throw InferenceExecutionException("LiteRT interpreter invocation failed");
    }
    end_timer();

    std::vector<std::vector<TensorElement>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    const auto& output_indices = interpreter_->outputs();
    outputs.reserve(output_indices.size());
    shapes.reserve(output_indices.size());

    for (int tensor_index : output_indices) {
        const TfLiteTensor* output = interpreter_->tensor(tensor_index);
        if (output == nullptr) {
            throw InferenceExecutionException("LiteRT output tensor is null");
        }

        std::vector<int64_t> shape = shape_from_dims(output->dims);
        const size_t element_count = element_count_from_shape(shape);
        std::vector<TensorElement> tensor_data;

        switch (output->type) {
        case kTfLiteFloat32:
            append_tensor_data(tensor_data, output->data.f, element_count);
            break;
        case kTfLiteUInt8:
            append_tensor_data(tensor_data, output->data.uint8, element_count);
            break;
        case kTfLiteInt32:
            append_tensor_data(tensor_data, output->data.i32, element_count);
            break;
        case kTfLiteInt64:
            append_tensor_data(tensor_data, output->data.i64, element_count);
            break;
        default:
            throw InferenceExecutionException("Unsupported LiteRT output tensor type: " +
                                              tensor_type_name(output->type));
        }

        shapes.push_back(std::move(shape));
        outputs.push_back(std::move(tensor_data));
    }

    return std::make_tuple(outputs, shapes);
}

std::vector<int> LiteRTInfer::makeInputDims(int tensor_index, const std::vector<int64_t>& requested_shape) const {
    const TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
    if (tensor == nullptr || tensor->dims == nullptr) {
        throw ModelLoadException("LiteRT input tensor metadata is unavailable");
    }

    std::vector<int> dims;
    dims.reserve(static_cast<size_t>(tensor->dims->size));

    if (requested_shape.size() == static_cast<size_t>(tensor->dims->size)) {
        for (int64_t dim : requested_shape) {
            dims.push_back(static_cast<int>(dim));
        }
    } else if (requested_shape.size() + 1 == static_cast<size_t>(tensor->dims->size)) {
        dims.push_back(static_cast<int>(batch_size_));
        for (int64_t dim : requested_shape) {
            dims.push_back(static_cast<int>(dim));
        }
    } else {
        throw ModelLoadException("LiteRT input shape rank mismatch: model rank " + std::to_string(tensor->dims->size) +
                                 ", requested rank " + std::to_string(requested_shape.size()));
    }

    return dims;
}

std::vector<int64_t> LiteRTInfer::tensorShape(int tensor_index) const {
    const TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
    if (tensor == nullptr) {
        return {};
    }
    return shape_from_dims(tensor->dims);
}

void LiteRTInfer::refreshMetadata() {
    const auto& inputs = interpreter_->inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const TfLiteTensor* tensor = interpreter_->tensor(inputs[i]);
        const std::string name =
            tensor != nullptr && tensor->name != nullptr ? tensor->name : "input" + std::to_string(i);
        const TensorDataType datatype =
            tensor != nullptr ? neuriplo_dtype_from_tflite(tensor->type) : TensorDataType::Float32;
        inference_metadata_.addInput(name, tensorShape(inputs[i]), batch_size_, datatype);
    }

    const auto& outputs = interpreter_->outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        const TfLiteTensor* tensor = interpreter_->tensor(outputs[i]);
        const std::string name =
            tensor != nullptr && tensor->name != nullptr ? tensor->name : "output" + std::to_string(i);
        const TensorDataType datatype =
            tensor != nullptr ? neuriplo_dtype_from_tflite(tensor->type) : TensorDataType::Float32;
        inference_metadata_.addOutput(name, tensorShape(outputs[i]), batch_size_, datatype);
    }
}
