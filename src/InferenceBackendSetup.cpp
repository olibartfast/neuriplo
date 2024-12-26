#include "InferenceBackendSetup.hpp"


std::unique_ptr<InferenceInterface> setup_inference_engine(const std::string& weights, const std::string& modelConfiguration , const bool use_gpu)
{
    #ifdef USE_ONNX_RUNTIME
    return std::make_unique<ORTInfer>(weights, use_gpu); 
    #elif USE_LIBTORCH 
    return std::make_unique<LibtorchInfer>(weights, use_gpu); 
    #elif USE_LIBTENSORFLOW 
    return std::make_unique<TFDetectionAPI>(weights, use_gpu); 
    #elif USE_OPENCV_DNN 
    return std::make_unique<OCVDNNInfer>(weights, modelConfiguration); 
    #elif USE_TENSORRT
    return std::make_unique<TRTInfer>(weights); 
    #elif USE_OPENVINO
    return std::make_unique<OVInfer>(weights, use_gpu); 
    #endif
    return nullptr;


}