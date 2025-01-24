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

std::unique_ptr<InferenceInterface> setup_inference_engine(const std::string& model_path, bool use_gpu = false, 
            size_t batch_size = 1, 
            const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());