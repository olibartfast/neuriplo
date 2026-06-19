#include "ORTInfer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

std::string trim_copy(const std::string& value) {
    const auto first =
        std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last =
        std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) {
        return "";
    }
    return std::string(first, last);
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string canonical_provider_alias(const std::string& provider_alias) {
    const auto alias = lower_copy(trim_copy(provider_alias));
    if (alias == "default" || alias == "default_cpu") {
        return "cpu";
    }
    if (alias == "nvidia_cuda") {
        return "cuda";
    }
    if (alias == "trt") {
        return "tensorrt";
    }
    if (alias == "intel_openvino") {
        return "openvino";
    }
    if (alias == "dml") {
        return "directml";
    }
    if (alias == "amd_migraphx") {
        return "migraphx";
    }
    if (alias == "qualcomm_qnn") {
        return "qnn";
    }
    if (alias == "arm_compute") {
        return "acl";
    }
    if (alias == "arm_nn") {
        return "armnn";
    }
    if (alias == "rockchip") {
        return "rknpu";
    }
    if (alias == "vitis_ai" || alias == "xilinx_vitis_ai") {
        return "vitisai";
    }
    if (alias == "huawei_cann") {
        return "cann";
    }
    return alias;
}

bool provider_available(const std::vector<std::string>& providers, const std::string& ort_name) {
    return std::find(providers.begin(), providers.end(), ort_name) != providers.end();
}

std::string available_provider_list(const std::vector<std::string>& providers) {
    std::ostringstream oss;
    for (size_t i = 0; i < providers.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << providers[i];
    }
    return oss.str();
}

std::unordered_map<std::string, std::string> qnn_provider_options() {
    std::unordered_map<std::string, std::string> options;
    const char* backend_path = std::getenv("NEURIPLO_ORT_QNN_BACKEND_PATH");
    options.emplace("backend_path",
                    backend_path != nullptr && std::string(backend_path).size() > 0 ? backend_path : "libQnnHtp.so");
    return options;
}

void append_provider(Ort::SessionOptions& session_options, const std::string& provider_alias) {
    if (provider_alias == "cuda") {
        OrtCUDAProviderOptions cuda_options;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        return;
    }

#ifdef ORT_ENABLE_TENSORRT_EP
    if (provider_alias == "tensorrt") {
        OrtTensorRTProviderOptions tensorrt_options;
        session_options.AppendExecutionProvider_TensorRT(tensorrt_options);
        return;
    }
#endif

#ifdef ORT_ENABLE_OPENVINO_EP
    if (provider_alias == "openvino") {
        OrtOpenVINOProviderOptions openvino_options;
        session_options.AppendExecutionProvider_OpenVINO(openvino_options);
        return;
    }
#endif

#ifdef ORT_ENABLE_MIGRAPHX_EP
    if (provider_alias == "migraphx") {
        OrtMIGraphXProviderOptions migraphx_options;
        session_options.AppendExecutionProvider_MIGraphX(migraphx_options);
        return;
    }
#endif

#ifdef ORT_ENABLE_QNN_EP
    if (provider_alias == "qnn") {
        session_options.AppendExecutionProvider("QNN", qnn_provider_options());
        return;
    }
#endif

#ifdef ORT_ENABLE_XNNPACK_EP
    if (provider_alias == "xnnpack") {
        session_options.AppendExecutionProvider("XNNPACK");
        return;
    }
#endif

#ifdef ORT_ENABLE_CANN_EP
    if (provider_alias == "cann") {
        // OrtCANNProviderOptions is an opaque type owned by ORT; it has no public
        // struct layout, so it must be created and released through the C API.
        const OrtApi& ort_api = Ort::GetApi();
        OrtCANNProviderOptions* cann_options = nullptr;
        Ort::ThrowOnError(ort_api.CreateCANNProviderOptions(&cann_options));
        try {
            session_options.AppendExecutionProvider_CANN(*cann_options);
        } catch (...) {
            ort_api.ReleaseCANNProviderOptions(cann_options);
            throw;
        }
        ort_api.ReleaseCANNProviderOptions(cann_options);
        return;
    }
#endif

#ifdef ORT_ENABLE_VITISAI_EP
    if (provider_alias == "vitisai") {
        session_options.AppendExecutionProvider_VitisAI();
        return;
    }
#endif

    throw std::runtime_error("ONNX Runtime provider is not build-enabled in neuriplo: " + provider_alias);
}

void configure_explicit_providers(Ort::SessionOptions& session_options, const std::vector<std::string>& requested) {
    const auto available = Ort::GetAvailableProviders();
    LOG(INFO) << "Available ONNX Runtime providers:";
    for (const auto& provider : available) {
        LOG(INFO) << provider;
    }

    const bool allows_cpu = std::find(requested.begin(), requested.end(), "cpu") != requested.end();
    if (!allows_cpu) {
        session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        LOG(INFO) << "ONNX Runtime CPU EP fallback disabled; add 'cpu' to NEURIPLO_ORT_EP to allow fallback";
    }

    for (const auto& provider_alias : requested) {
        const auto ort_name = ORTInfer::providerAliasToOrtName(provider_alias);
        if (ort_name.empty()) {
            throw std::runtime_error("Unsupported ONNX Runtime provider alias: " + provider_alias);
        }

        if (provider_alias == "cpu") {
            LOG(INFO) << "Using ONNX Runtime CPUExecutionProvider";
            continue;
        }

        if (!ORTInfer::isProviderBuildEnabled(provider_alias)) {
            throw std::runtime_error("ONNX Runtime provider '" + provider_alias +
                                     "' is not enabled in this neuriplo build");
        }

        if (!provider_available(available, ort_name)) {
            throw std::runtime_error(
                "Requested ONNX Runtime provider '" + ort_name +
                "' is not available in this ORT build. Available providers: " + available_provider_list(available));
        }

        LOG(INFO) << "Using ONNX Runtime provider: " << ort_name;
        append_provider(session_options, provider_alias);
    }
}

} // namespace

ORTInfer::ORTInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                   const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Onnx Runtime Inference");
    Ort::SessionOptions session_options;
    const char* ep_env = std::getenv("NEURIPLO_ORT_EP");

    if (ep_env != nullptr && std::string(ep_env).size() > 0) {
        configure_explicit_providers(session_options, parseExecutionProviderList(ep_env));
    } else if (use_gpu) {
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        LOG(INFO) << "Available providers:";
        for (const auto& p : providers) {
            LOG(INFO) << p;
        }

        bool is_found = false;

        for (const auto& p : providers) {
            if (p.find("CUDA") != std::string::npos) {
                LOG(INFO) << "Using CUDA GPU";
                OrtCUDAProviderOptions cuda_options;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                is_found = true;
                break;
            }
        }

        if (!is_found) {
            for (const auto& p : providers) {
                if (p.find("ROCM") != std::string::npos) {
                    LOG(WARNING) << "Using deprecated ONNX Runtime ROCm provider for legacy compatibility";
                    OrtROCMProviderOptions rocm_options;
                    session_options.AppendExecutionProvider_ROCM(rocm_options);
                    is_found = true;
                    break;
                }
            }
        }

        if (!is_found) {
            LOG(INFO) << "No GPU provider available (CUDA/ROCm), falling back to CPU";
            session_options = Ort::SessionOptions();
        }
    } else {
        LOG(INFO) << "Using CPU";
        session_options = Ort::SessionOptions();
    }

    try {
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
    } catch (const Ort::Exception& ex) {
        LOG(ERROR) << "Failed to load the ONNX model: " << ex.what();
        state_ = BackendState::Failed;
        throw ModelLoadException(std::string("ONNX model load failed: ") + ex.what());
    }

    Ort::AllocatorWithDefaultOptions allocator;
    LOG(INFO) << "Input Node Name/Shape (" << session_.GetInputCount() << "):";

    // Process inputs
    for (std::size_t i = 0; i < session_.GetInputCount(); i++) {
        const std::string name = session_.GetInputNameAllocated(i, allocator).get();
        auto type_info = session_.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shapes = tensor_info.GetShape();
        auto input_type = tensor_info.GetElementType();

        // Check if this input has dynamic dimensions
        bool has_dynamic = false;
        for (size_t j = 1; j < shapes.size(); j++) { // Skip batch dimension
            if (shapes[j] == -1) {
                has_dynamic = true;
                break;
            }
        }

        // Handle batch dimension first
        shapes[0] = shapes[0] == -1 ? batch_size : shapes[0];

        // Handle dimensions if dynamic or if input_sizes provided
        if (has_dynamic || (!input_sizes.empty() && i < input_sizes.size())) {
            if (input_sizes.empty() || i >= input_sizes.size()) {
                throw std::runtime_error("Dynamic shapes found but no input sizes provided for input '" + name + "'");
            }

            const auto& provided_shape = input_sizes[i];

            if (has_dynamic) {
                // Check if provided shape has enough dimensions for dynamic inputs
                size_t dynamic_dim_count = 0;
                for (size_t j = 1; j < shapes.size(); j++) {
                    if (shapes[j] == -1)
                        dynamic_dim_count++;
                }

                if (provided_shape.size() < dynamic_dim_count) {
                    throw std::runtime_error("Not enough dimensions provided for dynamic shapes in input '" + name +
                                             "'");
                }

                // Apply provided dimensions - map all non-batch dimensions
                if (provided_shape.size() != shapes.size() - 1) {
                    throw std::runtime_error("Provided shape size mismatch for input '" + name + "'. Expected " +
                                             std::to_string(shapes.size() - 1) + " dimensions, got " +
                                             std::to_string(provided_shape.size()));
                }

                for (size_t j = 1; j < shapes.size(); j++) {
                    shapes[j] = provided_shape[j - 1];
                }
            } else {
                // Override fixed dimensions with provided dimensions (skip batch dimension)
                if (provided_shape.size() != shapes.size() - 1) {
                    throw std::runtime_error("Provided shape size mismatch for input '" + name + "'. Expected " +
                                             std::to_string(shapes.size() - 1) + " dimensions, got " +
                                             std::to_string(provided_shape.size()));
                }

                for (size_t j = 1; j < shapes.size(); j++) {
                    shapes[j] = provided_shape[j - 1];
                }
            }
        }

        LOG(INFO) << "\t" << name << " : " << print_shape(shapes);
        inference_metadata_.addInput(name, shapes, batch_size, inputTensorDataType(input_type));

        std::string input_type_str = getDataTypeString(input_type);
        LOG(INFO) << "\tData Type: " << input_type_str;
    }

    // Log network dimensions from first input
    const auto& first_input = inference_metadata_.getInputs()[0].shape;
    const auto channels = static_cast<int>(first_input[1]);
    const auto network_height = static_cast<int>(first_input[2]);
    const auto network_width = static_cast<int>(first_input[3]);

    LOG(INFO) << "channels " << channels;
    LOG(INFO) << "width " << network_width;
    LOG(INFO) << "height " << network_height;

    // Process outputs
    LOG(INFO) << "Output Node Name/Shape (" << session_.GetOutputCount() << "):";
    for (std::size_t i = 0; i < session_.GetOutputCount(); i++) {
        const std::string name = session_.GetOutputNameAllocated(i, allocator).get();
        auto type_info = session_.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shapes = tensor_info.GetShape();
        auto output_type = tensor_info.GetElementType();
        shapes[0] = shapes[0] == -1 ? batch_size : shapes[0];
        LOG(INFO) << "\t" << name << " : " << print_shape(shapes);
        inference_metadata_.addOutput(name, shapes, batch_size, outputTensorDataType(output_type));
    }

    state_ = BackendState::Ready;
}

std::vector<std::string> ORTInfer::parseExecutionProviderList(const std::string& provider_list) {
    std::vector<std::string> providers;
    std::stringstream stream(provider_list);
    std::string item;

    while (std::getline(stream, item, ',')) {
        const auto provider = canonical_provider_alias(item);
        if (!provider.empty()) {
            providers.push_back(provider);
        }
    }

    if (providers.empty()) {
        throw std::runtime_error("NEURIPLO_ORT_EP did not contain any provider names");
    }

    return providers;
}

std::string ORTInfer::providerAliasToOrtName(const std::string& provider_alias) {
    const auto alias = canonical_provider_alias(provider_alias);
    if (alias == "cpu") {
        return "CPUExecutionProvider";
    }
    if (alias == "cuda") {
        return "CUDAExecutionProvider";
    }
    if (alias == "tensorrt") {
        return "TensorrtExecutionProvider";
    }
    if (alias == "openvino") {
        return "OpenVINOExecutionProvider";
    }
    if (alias == "directml") {
        return "DmlExecutionProvider";
    }
    if (alias == "migraphx") {
        return "MIGraphXExecutionProvider";
    }
    if (alias == "qnn") {
        return "QNNExecutionProvider";
    }
    if (alias == "nnapi") {
        return "NnapiExecutionProvider";
    }
    if (alias == "coreml") {
        return "CoreMLExecutionProvider";
    }
    if (alias == "xnnpack") {
        return "XnnpackExecutionProvider";
    }
    if (alias == "acl") {
        return "ACLExecutionProvider";
    }
    if (alias == "armnn") {
        return "ArmNNExecutionProvider";
    }
    if (alias == "rknpu") {
        return "RknpuExecutionProvider";
    }
    if (alias == "vitisai") {
        return "VitisAIExecutionProvider";
    }
    if (alias == "cann") {
        return "CANNExecutionProvider";
    }
    if (alias == "azure") {
        return "AzureExecutionProvider";
    }
    if (alias == "tvm") {
        return "TvmExecutionProvider";
    }
    return "";
}

bool ORTInfer::isProviderBuildEnabled(const std::string& provider_alias) {
    const auto alias = canonical_provider_alias(provider_alias);
    if (alias == "cpu") {
        return true;
    }
    if (alias == "cuda") {
        return true;
    }
#ifdef ORT_ENABLE_TENSORRT_EP
    if (alias == "tensorrt") {
        return true;
    }
#endif
#ifdef ORT_ENABLE_OPENVINO_EP
    if (alias == "openvino") {
        return true;
    }
#endif
#ifdef ORT_ENABLE_MIGRAPHX_EP
    if (alias == "migraphx") {
        return true;
    }
#endif
#ifdef ORT_ENABLE_QNN_EP
    if (alias == "qnn") {
        return true;
    }
#endif
#ifdef ORT_ENABLE_XNNPACK_EP
    if (alias == "xnnpack") {
        return true;
    }
#endif
#ifdef ORT_ENABLE_CANN_EP
    if (alias == "cann") {
        return true;
    }
#endif
#ifdef ORT_ENABLE_VITISAI_EP
    if (alias == "vitisai") {
        return true;
    }
#endif
    return false;
}

std::string ORTInfer::getDataTypeString(ONNXTensorElementDataType type) {
    switch (type) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "Float";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "Int64";
    default:
        return "Unknown";
    }
}

TensorDataType ORTInfer::inputTensorDataType(ONNXTensorElementDataType type) {
    switch (type) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return TensorDataType::Float32;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return TensorDataType::Int32;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return TensorDataType::Int64;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return TensorDataType::UInt8;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return TensorDataType::Int8;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return TensorDataType::Bool;
    default:
        throw std::runtime_error("Unsupported ONNX input tensor element type for metadata datatype: " +
                                 std::to_string(static_cast<int>(type)));
    }
}

TensorDataType ORTInfer::outputTensorDataType(ONNXTensorElementDataType type) {
    // Mirror get_infer_results_raw(): outputs are emitted only as these element
    // kinds, so advertise nothing the infer path cannot actually produce.
    switch (type) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return TensorDataType::Float32;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return TensorDataType::Int32;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return TensorDataType::Int64;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return TensorDataType::UInt8;
    default:
        throw std::runtime_error("Unsupported ONNX output tensor element type for metadata datatype: " +
                                 std::to_string(static_cast<int>(type)));
    }
}

std::string ORTInfer::print_shape(const std::vector<std::int64_t>& v) {
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

size_t ORTInfer::getSizeByDim(const std::vector<int64_t>& dims) {
    size_t size = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == -1 || dims[i] == 0) {
            continue;
        }
        size *= dims[i];
    }
    return size;
}

std::vector<Ort::Value> ORTInfer::run_session(const std::vector<std::vector<uint8_t>>& input_tensors) {

    const auto& inputs = inference_metadata_.getInputs();
    const auto& outputs = inference_metadata_.getOutputs();

    // Create Ort tensors from input data
    // We assume input_tensors[i] already contains the data in the correct layout (e.g. float/int bytes)
    // Warning: We are casting raw bytes to float* if the model expects float.
    // This assumes the input `vector<uint8_t>` is actually a byte view of the float buffer.

    std::vector<Ort::Value> in_ort_tensors;
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<int64_t> orig_target_sizes;

    // Process user-provided input tensors
    size_t num_inputs = session_.GetInputCount();
    if (input_tensors.size() != num_inputs) {
        throw std::runtime_error("Input tensor count mismatch. Expected " + std::to_string(num_inputs) + ", got " +
                                 std::to_string(input_tensors.size()));
    }

    for (size_t i = 0; i < num_inputs; ++i) {
        auto type_info = session_.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto onnx_type = tensor_info.GetElementType();
        const auto& input_shape = inputs[i].shape; // Use our stored shape which handles dynamic/overrides

        // Calculate expected size for validation
        size_t element_size = 1;
        switch (onnx_type) {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            element_size = 4;
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            element_size = 1;
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            element_size = 8;
            break;
        default:
            LOG(ERROR) << "Unsupported input data type: " << onnx_type;
            throw std::runtime_error("Unsupported input data type");
        }

        size_t expected_elements = 1;
        for (int64_t dim : input_shape) {
            if (dim < 0) {
                LOG(WARNING) << "Input shape contains dynamic dimension: " << dim
                             << ". Validation might be inaccurate.";
            }
            expected_elements *= (dim < 0 ? 1 : dim);
        }

        size_t expected_bytes = expected_elements * element_size;
        if (input_tensors[i].size() != expected_bytes) {
            throw std::runtime_error("Input data size mismatch for tensor " + std::to_string(i) + ". Expected " +
                                     std::to_string(expected_bytes) + " bytes, got " +
                                     std::to_string(input_tensors[i].size()));
        }

        // Create tensor from raw bytes using the correct type
        // We cast away constness as Ort::Value::CreateTensor expects mutable pointer
        switch (onnx_type) {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            in_ort_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, reinterpret_cast<float*>(const_cast<uint8_t*>(input_tensors[i].data())), expected_elements,
                input_shape.data(), input_shape.size()));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            in_ort_tensors.emplace_back(
                Ort::Value::CreateTensor<uint8_t>(memory_info, const_cast<uint8_t*>(input_tensors[i].data()),
                                                  expected_elements, input_shape.data(), input_shape.size()));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            in_ort_tensors.emplace_back(Ort::Value::CreateTensor<int8_t>(
                memory_info, reinterpret_cast<int8_t*>(const_cast<uint8_t*>(input_tensors[i].data())),
                expected_elements, input_shape.data(), input_shape.size()));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            in_ort_tensors.emplace_back(Ort::Value::CreateTensor<int32_t>(
                memory_info, reinterpret_cast<int32_t*>(const_cast<uint8_t*>(input_tensors[i].data())),
                expected_elements, input_shape.data(), input_shape.size()));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            in_ort_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, reinterpret_cast<int64_t*>(const_cast<uint8_t*>(input_tensors[i].data())),
                expected_elements, input_shape.data(), input_shape.size()));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            in_ort_tensors.emplace_back(Ort::Value::CreateTensor<bool>(
                memory_info, reinterpret_cast<bool*>(const_cast<uint8_t*>(input_tensors[i].data())), expected_elements,
                input_shape.data(), input_shape.size()));
            break;
        default:
            LOG(ERROR) << "Unsupported input data type for ORT: " << onnx_type;
            state_ = BackendState::Failed;
            throw InferenceExecutionException("Unsupported input data type for ORT: " +
                                              std::to_string(static_cast<int>(onnx_type)));
        }
    }

    // Run inference
    std::vector<const char*> input_names_char(inputs.size());
    std::transform(inputs.begin(), inputs.end(), input_names_char.begin(),
                   [](const LayerInfo& layer) { return layer.name.c_str(); });

    std::vector<const char*> output_names_char(outputs.size());
    std::transform(outputs.begin(), outputs.end(), output_names_char.begin(),
                   [](const LayerInfo& layer) { return layer.name.c_str(); });

    return session_.Run(Ort::RunOptions{nullptr}, input_names_char.data(), in_ort_tensors.data(), in_ort_tensors.size(),
                        output_names_char.data(), outputs.size());
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
ORTInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {

    std::vector<Ort::Value> output_ort_tensors = run_session(input_tensors);

    std::vector<std::vector<TensorElement>> output_tensors;
    std::vector<std::vector<int64_t>> shapes;

    // Process output tensors
    assert(output_ort_tensors.size() == inference_metadata_.getOutputs().size());

    for (const Ort::Value& output_tensor : output_ort_tensors) {
        const auto& shape_ref = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> shape(shape_ref.begin(), shape_ref.end());

        size_t num_elements = 1;
        for (int64_t dim : shape) {
            num_elements *= dim;
        }

        std::vector<TensorElement> tensor_data;
        tensor_data.reserve(num_elements);

        // Retrieve tensor data
        const int onnx_type = output_tensor.GetTensorTypeAndShapeInfo().GetElementType();
        switch (onnx_type) {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            const float* output_data_float = output_tensor.GetTensorData<float>();
            for (size_t i = 0; i < num_elements; ++i) {
                tensor_data.emplace_back(output_data_float[i]);
            }
            break;
        }
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            const int64_t* output_data_int64 = output_tensor.GetTensorData<int64_t>();
            for (size_t i = 0; i < num_elements; ++i) {
                tensor_data.emplace_back(output_data_int64[i]);
            }
            break;
        }
        default:
            LOG(ERROR) << "Unsupported tensor type: " << onnx_type;
            state_ = BackendState::Failed;
            throw InferenceExecutionException("Unsupported output tensor type for ORT: " + std::to_string(onnx_type));
        }

        output_tensors.emplace_back(std::move(tensor_data));
        shapes.emplace_back(shape);
    }

    return std::make_tuple(output_tensors, shapes);
}

std::vector<RawOutputTensor> ORTInfer::get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) {

    std::vector<Ort::Value> output_ort_tensors = run_session(input_tensors);

    std::vector<RawOutputTensor> raw_outputs;
    raw_outputs.reserve(output_ort_tensors.size());

    for (const Ort::Value& output_tensor : output_ort_tensors) {
        const auto& shape_ref = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

        size_t num_elements = 1;
        for (int64_t dim : shape_ref) {
            num_elements *= dim;
        }

        RawOutputTensor raw;
        raw.shape.assign(shape_ref.begin(), shape_ref.end());

        // Copy the typed output buffer once, as bytes; no per-element boxing.
        auto copy_bytes = [&](const void* data, size_t element_size, TensorDtype dtype) {
            raw.dtype = dtype;
            const auto* bytes = static_cast<const uint8_t*>(data);
            raw.bytes.assign(bytes, bytes + num_elements * element_size);
        };

        const int onnx_type = output_tensor.GetTensorTypeAndShapeInfo().GetElementType();
        switch (onnx_type) {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            copy_bytes(output_tensor.GetTensorData<float>(), sizeof(float), TensorDtype::FP32);
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            copy_bytes(output_tensor.GetTensorData<int32_t>(), sizeof(int32_t), TensorDtype::INT32);
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            copy_bytes(output_tensor.GetTensorData<int64_t>(), sizeof(int64_t), TensorDtype::INT64);
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            copy_bytes(output_tensor.GetTensorData<uint8_t>(), sizeof(uint8_t), TensorDtype::UINT8);
            break;
        default:
            LOG(ERROR) << "Unsupported tensor type: " << onnx_type;
            state_ = BackendState::Failed;
            throw InferenceExecutionException("Unsupported output tensor type for ORT: " + std::to_string(onnx_type));
        }

        raw_outputs.push_back(std::move(raw));
    }

    return raw_outputs;
}
