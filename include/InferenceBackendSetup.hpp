#pragma once
#include "common.hpp"
#include "InferenceInterface.hpp"
#include <string>

#ifdef USE_ONNX_RUNTIME
#include "ORTInfer.hpp"
#elif USE_LIBTORCH 
#include "LibtorchInfer.hpp"
#elif USE_LIBTENSORFLOW 
#include "TFDetectionAPI.hpp"
#elif USE_OPENCV_DNN 
#include "OCVDNNInfer.hpp"
#elif USE_TENSORRT
#include "TRTInfer.hpp"
#elif USE_OPENVINO
#include "OVInfer.hpp"
#elif USE_GGML
#include "GGMLInfer.hpp"
#endif

// Inference mode enumeration
enum class InferenceMode {
    OFFLINE,  // Local inference (default, current behavior)
    CLIENT    // Remote inference via HTTP client
};

// Setup inference engine in OFFLINE mode (local inference)
std::unique_ptr<InferenceInterface> setup_inference_engine(
    const std::string& model_path, 
    bool use_gpu = false, 
    size_t batch_size = 1, 
    const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

// Setup inference engine with mode selection
std::unique_ptr<InferenceInterface> setup_inference_engine(
    InferenceMode mode,
    const std::string& model_path_or_server, 
    bool use_gpu = false, 
    size_t batch_size = 1, 
    const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>(),
    int server_port = 8080);