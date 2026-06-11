#pragma once
#include "InferenceInterface.hpp"
#include "common.hpp"

// Options for selecting and configuring an inference backend at runtime.
// backend_id matches a BackendRuntimeRegistration id (e.g. "ONNX_RUNTIME",
// "TENSORRT"); empty selects the build's default backend. A string-id overload
// of setup_inference_engine is deliberately avoided: a `const char*` argument
// would convert to the legacy `bool use_gpu` parameter instead of std::string.
struct EngineOptions {
    std::string model_path;
    std::string backend_id;
    bool use_gpu = false;
    size_t batch_size = 1;
    std::vector<std::vector<int64_t>> input_sizes;
};

std::unique_ptr<InferenceInterface> setup_inference_engine(const EngineOptions& options);

std::unique_ptr<InferenceInterface>
setup_inference_engine(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                       const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());
