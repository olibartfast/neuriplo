#include "InferenceBackendSetup.hpp"
#include "InferenceClient.hpp"
#include <glog/logging.h>

// Original function for backward compatibility (OFFLINE mode)
std::unique_ptr<InferenceInterface> setup_inference_engine(
    const std::string& model_path, 
    bool use_gpu, 
    size_t batch_size, 
    const std::vector<std::vector<int64_t>>& input_sizes)
{
    #ifdef USE_ONNX_RUNTIME
    return std::make_unique<ORTInfer>(model_path, use_gpu, batch_size, input_sizes); 
    #elif USE_LIBTORCH 
    return std::make_unique<LibtorchInfer>(model_path, use_gpu, batch_size, input_sizes); 
    #elif USE_LIBTENSORFLOW 
    return std::make_unique<TFDetectionAPI>(model_path, use_gpu, batch_size, input_sizes); 
    #elif USE_OPENCV_DNN 
    return std::make_unique<OCVDNNInfer>(model_path, use_gpu, batch_size, input_sizes); 
    #elif USE_TENSORRT
    return std::make_unique<TRTInfer>(model_path, true, batch_size, input_sizes); 
    #elif USE_OPENVINO
    return std::make_unique<OVInfer>(model_path, use_gpu, batch_size, input_sizes); 
    #elif USE_GGML
    return std::make_unique<GGMLInfer>(model_path, use_gpu, batch_size, input_sizes); 
    #endif
    return nullptr;
}

// New function with mode selection
std::unique_ptr<InferenceInterface> setup_inference_engine(
    InferenceMode mode,
    const std::string& model_path_or_server, 
    bool use_gpu, 
    size_t batch_size, 
    const std::vector<std::vector<int64_t>>& input_sizes,
    int server_port)
{
    switch (mode) {
        case InferenceMode::OFFLINE:
            LOG(INFO) << "Setting up inference engine in OFFLINE mode";
            return setup_inference_engine(model_path_or_server, use_gpu, batch_size, input_sizes);
            
        case InferenceMode::CLIENT:
            LOG(INFO) << "Setting up inference engine in CLIENT mode (server: " 
                     << model_path_or_server << ":" << server_port << ")";
            return std::make_unique<neuriplo::client::InferenceClient>(
                model_path_or_server, server_port);
            
        default:
            LOG(ERROR) << "Unknown inference mode";
            return nullptr;
    }
}
