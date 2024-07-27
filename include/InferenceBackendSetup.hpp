#pragma once
#include "common.hpp"
#include "InferenceInterface.hpp"
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
#endif

std::unique_ptr<InferenceInterface> setup_inference_engine(const std::string& weights, const std::string& modelConfiguration , const bool use_gpu);