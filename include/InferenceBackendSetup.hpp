#pragma once
#include "InferenceInterface.hpp"
#include "common.hpp"
// Each backend ships a concrete Abstract Factory (IBackendRuntimeFactory) that
// owns construction of its adapter. The factory header transitively includes
// the corresponding *Infer adapter, so only the factory needs to be pulled in.
#ifdef USE_ONNX_RUNTIME
#include "ORTRuntimeFactory.hpp"
#elif USE_LIBTORCH
#include "LibtorchRuntimeFactory.hpp"
#elif USE_LIBTENSORFLOW
#include "TFRuntimeFactory.hpp"
#elif USE_OPENCV_DNN
#include "OCVDNNRuntimeFactory.hpp"
#elif USE_TENSORRT
#include "TRTRuntimeFactory.hpp"
#elif USE_OPENVINO
#include "OVRuntimeFactory.hpp"
#elif USE_GGML
#include "GGMLRuntimeFactory.hpp"
#elif USE_CACTUS
#include "CactusRuntimeFactory.hpp"
#elif USE_MIGRAPHX
#include "MIGraphXRuntimeFactory.hpp"
#elif USE_LLAMACPP
#include "LlamaCppRuntimeFactory.hpp"
#elif USE_EXECUTORCH
#include "ExecuTorchRuntimeFactory.hpp"
#elif USE_LITERT
#include "LiteRTRuntimeFactory.hpp"
#endif

std::unique_ptr<InferenceInterface>
setup_inference_engine(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                       const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());
