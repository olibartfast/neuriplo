#include "InferenceBackendSetup.hpp"


std::unique_ptr<InferenceInterface> setup_inference_engine(const std::string& model_path, bool use_gpu, size_t batch_size, const std::vector<std::vector<int64_t>>& input_sizes)
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
    #endif
    return nullptr;


}