#include "InferenceBackendSetup.hpp"

#include "decorators/LoggingBackend.hpp"
#include "decorators/ProfilingBackend.hpp"

#include <cstdlib>
#include <glog/logging.h>
#include <memory>
#include <string>

namespace {

// Selects the concrete Abstract Factory for the backend compiled into this
// translation unit. Only one branch is live (single-backend compile model).
std::unique_ptr<IBackendRuntimeFactory> make_runtime_factory() {
#ifdef USE_ONNX_RUNTIME
    return std::make_unique<ORTRuntimeFactory>();
#elif USE_LIBTORCH
    return std::make_unique<LibtorchRuntimeFactory>();
#elif USE_LIBTENSORFLOW
    return std::make_unique<TFRuntimeFactory>();
#elif USE_OPENCV_DNN
    return std::make_unique<OCVDNNRuntimeFactory>();
#elif USE_TENSORRT
    return std::make_unique<TRTRuntimeFactory>();
#elif USE_OPENVINO
    return std::make_unique<OVRuntimeFactory>();
#elif USE_GGML
    return std::make_unique<GGMLRuntimeFactory>();
#elif USE_CACTUS
    return std::make_unique<CactusRuntimeFactory>();
#elif USE_MIGRAPHX
    return std::make_unique<MIGraphXRuntimeFactory>();
#elif USE_LLAMACPP
    return std::make_unique<LlamaCppRuntimeFactory>();
#elif USE_EXECUTORCH
    return std::make_unique<ExecuTorchRuntimeFactory>();
#elif USE_LITERT
    return std::make_unique<LiteRTRuntimeFactory>();
#else
    return nullptr;
#endif
}

// Opt-in only: defaults to false so the production path is byte-identical to
// before unless the operator explicitly enables instrumentation.
bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return std::string(value) == "1" || std::string(value) == "true";
}

// Wraps the backend in optional, opt-in Decorators. Order: profiling is the
// inner decorator (measures the raw backend), logging is outermost so it also
// observes the profiling layer. Both are pure pass-through for results.
std::unique_ptr<InferenceInterface> apply_optional_decorators(std::unique_ptr<InferenceInterface> backend) {
    if (env_flag_enabled("NEURIPLO_ENABLE_PROFILING")) {
        backend = std::make_unique<ProfilingBackend>(std::move(backend));
    }
    if (env_flag_enabled("NEURIPLO_ENABLE_LOGGING")) {
        backend = std::make_unique<LoggingBackend>(std::move(backend));
    }
    return backend;
}

} // namespace

std::unique_ptr<InferenceInterface> setup_inference_engine(const std::string& model_path, bool use_gpu,
                                                           size_t batch_size,
                                                           const std::vector<std::vector<int64_t>>& input_sizes) {
    auto factory = make_runtime_factory();
    if (!factory) {
        return nullptr;
    }

    // Preserve TensorRT's historical device-placement quirk: the engine is
    // always built for GPU regardless of the caller's use_gpu argument.
    bool effective_use_gpu = use_gpu;
#ifdef USE_TENSORRT
    effective_use_gpu = true;
#endif

    try {
        auto backend = factory->create_backend(model_path, effective_use_gpu, batch_size, input_sizes);
        if (!backend) {
            return nullptr;
        }

        backend = apply_optional_decorators(std::move(backend));

        // Eager load preserves the "constructed == ready" contract that callers
        // rely on (they query metadata immediately after this returns). Backends
        // that still load in their constructor treat this as a no-op that
        // confirms the Ready state.
        backend->load();
        if (backend->state() == BackendState::Failed) {
            LOG(ERROR) << "setup_inference_engine: backend failed to load model '" << model_path << "'";
            return nullptr;
        }

        return backend;
    } catch (const InferenceException& e) {
        // Translate load failures into the nullptr contract both downstream
        // consumers already handle, instead of terminating the process.
        LOG(ERROR) << "setup_inference_engine: " << e.what();
        return nullptr;
    }
}
