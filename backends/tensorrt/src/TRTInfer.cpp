#include "TRTInfer.hpp"

#include <cuda_fp16.h> // For __half if using half-precision
#include <fstream>
#include <sstream>

namespace {

// Reported tensor shapes must include the batch dimension, because that is what
// every other backend reports and what clients send. Reporting the trailing
// dimensions only made the same model take [3,H,W] when served through
// TensorRT and [1,3,H,W] through ONNX Runtime, so a client that worked against
// one was rejected by the other.
//
// `trailing` is the shape without the batch dimension, as built for the
// input_sizes override logic; `dims` is the engine's own view, whose leading
// dimension is authoritative unless it is dynamic (-1), in which case the
// configured batch size is what will actually be executed.
std::vector<int64_t> withBatchDimension(const std::vector<int64_t>& trailing, const nvinfer1::Dims& dims,
                                        size_t batch_size) {
    if (dims.nbDims <= 0) {
        return trailing; // Scalar tensor: there is no batch dimension to restore.
    }

    std::vector<int64_t> shape;
    shape.reserve(trailing.size() + 1);
    const int64_t leading = dims.d[0];
    shape.push_back(leading == -1 ? static_cast<int64_t>(batch_size) : leading);
    shape.insert(shape.end(), trailing.begin(), trailing.end());
    return shape;
}

} // namespace

// CUDA error checking macro
#define CHECK_CUDA(status)                                                                                             \
    do {                                                                                                               \
        auto ret = (status);                                                                                           \
        if (ret != cudaSuccess) {                                                                                      \
            LOG(ERROR) << "CUDA error: " << cudaGetErrorString(ret);                                                   \
            state_ = BackendState::Failed;                                                                             \
            throw InferenceException(std::string("CUDA error: ") + cudaGetErrorString(ret));                           \
        }                                                                                                              \
    } while (0)

TRTInfer::TRTInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                   const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, true, batch_size, input_sizes} {
    LOG(INFO) << "Initializing TensorRT for model " << model_path;
    initializeBuffers(model_path, input_sizes);
    populateInferenceMetadata(input_sizes);
    state_ = BackendState::Ready;
}

TRTInfer::~TRTInfer() {
    for (size_t i = 0; i < buffers_.size(); ++i) {
        void* buffer = buffers_[i];
        if (buffer) {
            cudaError_t err = cudaFree(buffer);
            if (err != cudaSuccess) {
                LOG(ERROR) << "cudaFree failed for buffer[" << i << "]: " << cudaGetErrorString(err);
            }
            buffers_[i] = nullptr;
        }
    }
    if (context_) {
        delete context_;
        context_ = nullptr;
    }
    engine_.reset();
    if (runtime_) {
        delete runtime_;
        runtime_ = nullptr;
    }
}

void TRTInfer::initializeBuffers(const std::string& engine_path, const std::vector<std::vector<int64_t>>& input_sizes) {
    // Create TensorRT runtime.
    // TensorRT keeps the ILogger reference for the lifetime of the runtime and of
    // every engine and execution context built from it, and calls back into it
    // from arbitrary threads. A stack or member logger would therefore dangle as
    // soon as this function returned (or the TRTInfer was copied/moved), and the
    // next TensorRT diagnostic would call through a freed vtable. The logger is
    // stateless, so a single function-local static outlives every TensorRT object
    // and is safe to share.
    static Logger logger;
    runtime_ = nvinfer1::createInferRuntime(logger);

    // Load engine file
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }
    engine_file.seekg(0, std::ios::end);
    size_t file_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(file_size);
    engine_file.read(engine_data.data(), file_size);
    engine_file.close();

    // Deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), file_size));
    if (!engine_) {
        state_ = BackendState::Failed;
        throw std::runtime_error("Failed to deserialize TensorRT engine (plan built with an "
                                 "incompatible TensorRT version?): " +
                                 engine_path);
    }
    createContextAndAllocateBuffers(input_sizes);
}

// calculate size of tensor
size_t TRTInfer::getSizeByDim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] == -1 || dims.d[i] == 0) {
            continue;
        }
        size *= dims.d[i];
    }
    return size;
}

TensorDataType TRTInfer::toTensorDataType(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT:
        return TensorDataType::Float32;
    case nvinfer1::DataType::kINT32:
        return TensorDataType::Int32;
    case nvinfer1::DataType::kINT64:
        return TensorDataType::Int64;
    case nvinfer1::DataType::kINT8:
        return TensorDataType::Int8;
    case nvinfer1::DataType::kBOOL:
        return TensorDataType::Bool;
    case nvinfer1::DataType::kUINT8:
        return TensorDataType::UInt8;
    default:
        throw ModelLoadException("Unsupported TensorRT tensor element type for metadata datatype: " +
                                 std::to_string(static_cast<int>(type)));
    }
}

void TRTInfer::createContextAndAllocateBuffers(const std::vector<std::vector<int64_t>>& input_sizes) {
    context_ = engine_->createExecutionContext();
    int num_tensors = engine_->getNbIOTensors();
    buffers_.resize(num_tensors);
    input_tensor_names_.clear();
    output_tensor_names_.clear();
    num_inputs_ = 0;
    num_outputs_ = 0;

    // First pass: Identify inputs and set their shapes on the context
    // This is crucial for dynamic shapes so that output shapes can be correctly deduced
    for (int i = 0; i < num_tensors; ++i) {
        std::string tensor_name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(tensor_name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
            input_tensor_names_.push_back(tensor_name);

            // If we have input sizes, set them now
            if (num_inputs_ < input_sizes.size()) {
                nvinfer1::Dims engine_dims = engine_->getTensorShape(tensor_name.c_str());
                nvinfer1::Dims input_dims;
                const auto& current_input_size = input_sizes[num_inputs_];

                if (current_input_size.size() == static_cast<size_t>(engine_dims.nbDims)) {
                    // Exact match
                    input_dims.nbDims = engine_dims.nbDims;
                    for (int k = 0; k < engine_dims.nbDims; ++k) {
                        input_dims.d[k] = static_cast<int>(current_input_size[k]);
                    }
                } else if (current_input_size.size() == static_cast<size_t>(engine_dims.nbDims - 1)) {
                    // Prepend batch size
                    input_dims.nbDims = engine_dims.nbDims;
                    input_dims.d[0] = static_cast<int>(batch_size_);
                    for (int k = 1; k < engine_dims.nbDims; ++k) {
                        input_dims.d[k] = static_cast<int>(current_input_size[k - 1]);
                    }
                } else {
                    // Fallback
                    input_dims.nbDims = std::min((int)current_input_size.size(), nvinfer1::Dims::MAX_DIMS);
                    for (int k = 0; k < input_dims.nbDims; ++k) {
                        input_dims.d[k] = static_cast<int>(current_input_size[k]);
                    }
                }

                if (!context_->setInputShape(tensor_name.c_str(), input_dims)) {
                    // Preserve the existing fallback (the engine's own shape is used
                    // below) but say so explicitly: silently serving a different
                    // shape than the caller asked for is otherwise invisible.
                    nvinfer1::Dims engine_shape = engine_->getTensorShape(tensor_name.c_str());
                    std::ostringstream requested, actual;
                    for (int k = 0; k < input_dims.nbDims; ++k) {
                        requested << (k ? "x" : "") << input_dims.d[k];
                    }
                    for (int k = 0; k < engine_shape.nbDims; ++k) {
                        actual << (k ? "x" : "") << engine_shape.d[k];
                    }
                    LOG(WARNING) << "Failed to set input shape " << requested.str() << " for " << tensor_name
                                 << " in allocation phase; this engine does not support it. Falling back to the "
                                 << "engine's shape " << actual.str() << " -- the requested batch size of "
                                 << batch_size_ << " is being ignored.";
                }
            }
            num_inputs_++;
        } else {
            output_tensor_names_.push_back(tensor_name);
            num_outputs_++;
        }
    }

    // Second pass: Allocate buffers using the context's specific shapes (which now should include dynamic resolutions)
    for (int i = 0; i < num_tensors; ++i) {
        std::string tensor_name = engine_->getIOTensorName(i);
        // Prefer context shape over engine shape for dynamic dimensions
        nvinfer1::Dims dims = context_->getTensorShape(tensor_name.c_str());

        // Fallback to engine shape if context shape is invalid (though it shouldn't be if inputs are set)
        if (dims.nbDims == 0 || dims.d[0] == 0) { // Naive check for invalid dims
            dims = engine_->getTensorShape(tensor_name.c_str());
        }

        auto size = getSizeByDim(dims);
        // Ensure we don't allocate 0 bytes, or if we do, handle it.
        // getSizeByDim ignores -1, so it might return 1 for a purely dynamic shape, which is bad.
        // However, if we setInputShape correctly, dims should be fully concrete now.

        // Debug log
        LOG(INFO) << "Allocating buffer for " << tensor_name << " with shape [";
        for (int k = 0; k < dims.nbDims; ++k)
            LOG(INFO) << dims.d[k] << (k < dims.nbDims - 1 ? ", " : "");
        LOG(INFO) << "] and size " << size;

        size_t binding_size = 0;
        switch (engine_->getTensorDataType(tensor_name.c_str())) {
        case nvinfer1::DataType::kFLOAT:
            binding_size = size * sizeof(float);
            break;
        case nvinfer1::DataType::kINT32:
            binding_size = size * sizeof(int32_t);
            break;
        case nvinfer1::DataType::kINT64:
            binding_size = size * sizeof(int64_t);
            break;
        case nvinfer1::DataType::kHALF:
            binding_size = size * sizeof(__half);
            break;
        default:
            LOG(ERROR) << "Unsupported data type for tensor " << tensor_name;
            state_ = BackendState::Failed;
            throw ModelLoadException("Unsupported data type for tensor " + tensor_name);
        }
        CHECK_CUDA(cudaMalloc(&buffers_[i], binding_size));
    }
}

nvinfer1::Dims TRTInfer::outputDims(size_t index) const {
    const std::string& tensor_name = output_tensor_names_[index];
    if (context_) {
        return context_->getTensorShape(tensor_name.c_str());
    }
    return engine_->getTensorShape(tensor_name.c_str());
}

void TRTInfer::uploadAndEnqueue(const std::vector<std::vector<uint8_t>>& input_tensors) {

    // Process user-provided input tensors
    if (input_tensors.size() != num_inputs_) {
        throw std::runtime_error("Input tensor count mismatch. Expected " + std::to_string(num_inputs_) + ", got " +
                                 std::to_string(input_tensors.size()));
    }

    for (size_t i = 0; i < num_inputs_; ++i) {
        std::string tensor_name = input_tensor_names_[i];

        // 1. Get dimensions to calculate expected element count
        nvinfer1::Dims dims;
        if (context_) {
            dims = context_->getTensorShape(tensor_name.c_str());
        } else {
            dims = engine_->getTensorShape(tensor_name.c_str());
        }

        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++) {
            vol *= (dims.d[d] < 0 ? 1 : dims.d[d]);
        }

        // 2. Query the Engine for the type
        nvinfer1::DataType type = engine_->getTensorDataType(tensor_name.c_str());

        // 3. Calculate required bytes based on type
        size_t element_size = 0;
        switch (type) {
        case nvinfer1::DataType::kFLOAT:
            element_size = 4;
            break;
        case nvinfer1::DataType::kHALF:
            element_size = 2;
            break;
        case nvinfer1::DataType::kINT32:
            element_size = 4;
            break;
        case nvinfer1::DataType::kINT8:
            element_size = 1;
            break;
        case nvinfer1::DataType::kBOOL:
            element_size = 1;
            break;
        case nvinfer1::DataType::kINT64:
            element_size = 8;
            break; // Added support for INT64
        case nvinfer1::DataType::kUINT8:
            element_size = 1;
            break; // Added support for UINT8
        default:
            LOG(ERROR) << "Unsupported input data type for tensor " << tensor_name;
            state_ = BackendState::Failed;
            throw InferenceExecutionException("Unsupported input data type for tensor " + tensor_name);
        }

        size_t expected_bytes = vol * element_size;
        size_t actual_bytes = input_tensors[i].size();

        // 4. Validation
        if (actual_bytes != expected_bytes) {
            LOG(WARNING) << "Input tensor " << tensor_name << " size mismatch. Expected " << expected_bytes
                         << " bytes, got " << actual_bytes << " bytes.";
        }

        // 5. Pass to CUDA (No casting needed!)
        CHECK_CUDA(cudaMemcpy(buffers_[i], input_tensors[i].data(), actual_bytes, cudaMemcpyHostToDevice));
    }

    // Perform inference. RAII so the binding failures below, which throw, do
    // not leak the stream.
    struct StreamGuard {
        cudaStream_t stream = nullptr;
        ~StreamGuard() {
            if (stream) {
                cudaStreamDestroy(stream);
            }
        }
    } guard;
    CHECK_CUDA(cudaStreamCreate(&guard.stream));
    cudaStream_t stream = guard.stream;

    // Note: Dynamic shape checking loop removed as per optimization

    for (size_t i = 0; i < num_inputs_; ++i) {
        if (!context_->setInputTensorAddress(input_tensor_names_[i].c_str(), buffers_[i])) {
            LOG(ERROR) << "Failed to set input tensor address for tensor: " << input_tensor_names_[i];
            state_ = BackendState::Failed;
            throw InferenceExecutionException("Failed to set input tensor address for tensor: " +
                                              input_tensor_names_[i]);
        }
    }

    for (size_t i = 0; i < num_outputs_; ++i) {
        if (!context_->setOutputTensorAddress(output_tensor_names_[i].c_str(), buffers_[i + num_inputs_])) {
            LOG(ERROR) << "Failed to set output tensor address for tensor: " << output_tensor_names_[i];
            state_ = BackendState::Failed;
            throw InferenceExecutionException("Failed to set output tensor address for tensor: " +
                                              output_tensor_names_[i]);
        }
    }

    if (!context_->enqueueV3(stream)) {
        LOG(ERROR) << "Inference failed!";
        state_ = BackendState::Failed;
        throw InferenceExecutionException("TensorRT enqueueV3 inference call failed");
    }

    // Wait here rather than relying on the legacy default stream implicitly
    // synchronising with this one, so the callers' output copies are ordinary
    // reads of finished device memory.
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
TRTInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    uploadAndEnqueue(input_tensors);

    // Extract outputs and their shapes
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<std::vector<TensorElement>> outputs;

    for (size_t i = 0; i < num_outputs_; ++i) {
        std::string tensor_name = output_tensor_names_[i];
        const nvinfer1::Dims dims = outputDims(i);
        auto num_elements = getSizeByDim(dims);

        std::vector<TensorElement> tensor_data;

        switch (engine_->getTensorDataType(tensor_name.c_str())) {
        case nvinfer1::DataType::kFLOAT: {
            std::vector<float> output_data_float(num_elements);
            CHECK_CUDA(cudaMemcpy(output_data_float.data(), buffers_[i + num_inputs_], num_elements * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            for (const auto& value : output_data_float) {
                tensor_data.push_back(static_cast<float>(value));
            }
            break;
        }
        case nvinfer1::DataType::kINT32: {
            std::vector<int32_t> output_data_int(num_elements);
            CHECK_CUDA(cudaMemcpy(output_data_int.data(), buffers_[i + num_inputs_], num_elements * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));

            for (const auto& value : output_data_int) {
                tensor_data.push_back(static_cast<int32_t>(value));
            }
            break;
        }
        case nvinfer1::DataType::kINT64: {
            std::vector<int64_t> output_data_int64(num_elements);
            CHECK_CUDA(cudaMemcpy(output_data_int64.data(), buffers_[i + num_inputs_], num_elements * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost));

            for (const auto& value : output_data_int64) {
                tensor_data.push_back(static_cast<int64_t>(value));
            }
            break;
        }
        case nvinfer1::DataType::kHALF: {
            std::vector<__half> output_data_half(num_elements);
            CHECK_CUDA(cudaMemcpy(output_data_half.data(), buffers_[i + num_inputs_], num_elements * sizeof(__half),
                                  cudaMemcpyDeviceToHost));

            for (const auto& value : output_data_half) {
                tensor_data.push_back(static_cast<float>(__half2float(value)));
            }
            break;
        }
        default:
            LOG(ERROR) << "Unsupported output data type for tensor " << tensor_name;
            state_ = BackendState::Failed;
            throw InferenceExecutionException("Unsupported output data type for tensor " + tensor_name);
        }

        outputs.emplace_back(std::move(tensor_data));

        std::vector<int64_t> out_shape;
        for (int j = 0; j < dims.nbDims; ++j) {
            out_shape.push_back(dims.d[j]);
        }
        output_shapes.emplace_back(out_shape);
    }

    return std::make_tuple(std::move(outputs), std::move(output_shapes));
}

std::vector<RawOutputTensor> TRTInfer::get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) {
    uploadAndEnqueue(input_tensors);

    std::vector<RawOutputTensor> raw_outputs;
    raw_outputs.reserve(num_outputs_);

    for (size_t i = 0; i < num_outputs_; ++i) {
        const std::string& tensor_name = output_tensor_names_[i];
        const nvinfer1::Dims dims = outputDims(i);
        const size_t num_elements = getSizeByDim(dims);
        void* device_buffer = buffers_[i + num_inputs_];

        RawOutputTensor tensor;
        tensor.shape.reserve(static_cast<size_t>(dims.nbDims));
        for (int j = 0; j < dims.nbDims; ++j) {
            tensor.shape.push_back(dims.d[j]);
        }

        // One device-to-host copy straight into the destination byte buffer.
        auto copy_direct = [&](TensorDtype dtype, size_t element_size) {
            tensor.dtype = dtype;
            tensor.bytes.resize(num_elements * element_size);
            CHECK_CUDA(cudaMemcpy(tensor.bytes.data(), device_buffer, tensor.bytes.size(), cudaMemcpyDeviceToHost));
        };

        switch (engine_->getTensorDataType(tensor_name.c_str())) {
        case nvinfer1::DataType::kFLOAT:
            copy_direct(TensorDtype::FP32, sizeof(float));
            break;
        case nvinfer1::DataType::kINT32:
            copy_direct(TensorDtype::INT32, sizeof(int32_t));
            break;
        case nvinfer1::DataType::kINT64:
            copy_direct(TensorDtype::INT64, sizeof(int64_t));
            break;
        case nvinfer1::DataType::kUINT8:
            copy_direct(TensorDtype::UINT8, sizeof(uint8_t));
            break;
        case nvinfer1::DataType::kHALF: {
            // RawOutputTensor has no FP16 tag, so widen on the host. Still one
            // copy plus a flat conversion, not a per-scalar variant.
            std::vector<__half> staging(num_elements);
            CHECK_CUDA(
                cudaMemcpy(staging.data(), device_buffer, num_elements * sizeof(__half), cudaMemcpyDeviceToHost));
            tensor.dtype = TensorDtype::FP32;
            tensor.bytes.resize(num_elements * sizeof(float));
            auto* typed = reinterpret_cast<float*>(tensor.bytes.data());
            for (size_t j = 0; j < num_elements; ++j) {
                typed[j] = __half2float(staging[j]);
            }
            break;
        }
        default:
            LOG(ERROR) << "Unsupported output data type for tensor " << tensor_name;
            state_ = BackendState::Failed;
            throw InferenceExecutionException("Unsupported output data type for tensor " + tensor_name);
        }

        raw_outputs.push_back(std::move(tensor));
    }

    return raw_outputs;
}

void TRTInfer::populateInferenceMetadata(const std::vector<std::vector<int64_t>>& input_sizes) {
    bool dynamic_axis_detected = false;

    // Process input tensors
    for (int i = 0; i < num_inputs_; ++i) {
        std::string tensor_name = input_tensor_names_[i];
        nvinfer1::Dims dims = engine_->getTensorShape(tensor_name.c_str());
        std::vector<int64_t> shape;
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] == -1) {
                dynamic_axis_detected = true;
            }
            if (j > 0) { // Skip the batch dimension (index 0)
                shape.push_back(dims.d[j]);
            }
        }

        if (input_sizes.empty() && dynamic_axis_detected) {
            throw std::runtime_error("Dynamic axis detected in input tensor " + tensor_name +
                                     " but input_sizes is empty.");
        }

        if (!input_sizes.empty()) {
            // Override dynamic dimensions with provided input sizes
            if (i < input_sizes.size()) {
                size_t shape_idx = 0;
                for (size_t j = 0; j < input_sizes[i].size(); ++j) {
                    if (shape_idx < shape.size() && shape[shape_idx] == -1) {
                        shape[shape_idx] = input_sizes[i][j];
                    }
                    if (shape_idx < shape.size()) {
                        shape_idx++;
                    }
                }

                // Set input shape on context if we have valid input sizes
                if (context_) {
                    nvinfer1::Dims engine_dims = engine_->getTensorShape(tensor_name.c_str());
                    nvinfer1::Dims input_dims;

                    if (input_sizes[i].size() == static_cast<size_t>(engine_dims.nbDims)) {
                        // Exact match (includes batch)
                        input_dims.nbDims = engine_dims.nbDims;
                        for (int k = 0; k < engine_dims.nbDims; ++k) {
                            input_dims.d[k] = static_cast<int>(input_sizes[i][k]);
                        }
                    } else if (input_sizes[i].size() == static_cast<size_t>(engine_dims.nbDims - 1)) {
                        // Input sizes exclude batch, prepend it
                        input_dims.nbDims = engine_dims.nbDims;
                        input_dims.d[0] = static_cast<int>(batch_size_);
                        for (int k = 1; k < engine_dims.nbDims; ++k) {
                            input_dims.d[k] = static_cast<int>(input_sizes[i][k - 1]);
                        }
                    } else {
                        LOG(WARNING) << "Input size mismatch for tensor " << tensor_name << ". Expected "
                                     << engine_dims.nbDims << " or " << (engine_dims.nbDims - 1) << " dimensions, got "
                                     << input_sizes[i].size();
                        // Fallback: try to usage what we have, though it will likely fail setInputShape validation if
                        // wrong
                        input_dims.nbDims = std::min((int)input_sizes[i].size(), nvinfer1::Dims::MAX_DIMS);
                        for (int k = 0; k < input_dims.nbDims; ++k) {
                            input_dims.d[k] = static_cast<int>(input_sizes[i][k]);
                        }
                    }

                    if (!context_->setInputShape(tensor_name.c_str(), input_dims)) {
                        LOG(WARNING) << "Failed to set input shape for " << tensor_name;
                    }
                }
            }
        }
        inference_metadata_.addInput(tensor_name, withBatchDimension(shape, dims, batch_size_), batch_size_,
                                     toTensorDataType(engine_->getTensorDataType(tensor_name.c_str())));
    }

    // Process output tensors
    for (int i = 0; i < num_outputs_; ++i) {
        std::string tensor_name = output_tensor_names_[i];

        // Try to get shape from context first (if inputs were set, this gives concrete shapes)
        nvinfer1::Dims dims;
        if (context_ && !input_sizes.empty()) {
            dims = context_->getTensorShape(tensor_name.c_str());
        } else {
            dims = engine_->getTensorShape(tensor_name.c_str());
        }

        std::vector<int64_t> shape;
        for (int j = 0; j < dims.nbDims; ++j) {
            if (j > 0) { // Skip the batch dimension (index 0)
                shape.push_back(dims.d[j]);
            }
        }
        inference_metadata_.addOutput(tensor_name, withBatchDimension(shape, dims, batch_size_), batch_size_,
                                      toTensorDataType(engine_->getTensorDataType(tensor_name.c_str())));
    }
}