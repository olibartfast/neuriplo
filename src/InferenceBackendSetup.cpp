#include "InferenceBackendSetup.hpp"

#include "BackendRuntimeRegistry.hpp"
#include "decorators/LoggingBackend.hpp"
#include "decorators/ProfilingBackend.hpp"

#include <cstdlib>
#include <glog/logging.h>
#include <memory>
#include <string>

namespace {

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

std::string registered_backend_ids() {
    std::string ids;
    for (const BackendRuntimeRegistration& registration : get_registered_backends()) {
        if (!ids.empty()) {
            ids += ", ";
        }
        ids += registration.id;
    }
    return ids;
}

} // namespace

std::unique_ptr<InferenceInterface> setup_inference_engine(const EngineOptions& options) {
    const BackendRuntimeRegistration* registration = nullptr;
    if (options.backend_id.empty()) {
        registration = get_compiled_backend_registration();
    } else {
        registration = find_backend_registration(options.backend_id);
        if (!registration) {
            LOG(ERROR) << "setup_inference_engine: backend '" << options.backend_id
                       << "' is not compiled into this build; available backends: " << registered_backend_ids();
            return nullptr;
        }
    }

    if (!registration || !registration->create_factory) {
        return nullptr;
    }
    auto factory = registration->create_factory();
    if (!factory) {
        return nullptr;
    }

    bool effective_use_gpu = registration->force_gpu ? true : options.use_gpu;

    try {
        auto backend =
            factory->create_backend(options.model_path, effective_use_gpu, options.batch_size, options.input_sizes);
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
            LOG(ERROR) << "setup_inference_engine: backend failed to load model '" << options.model_path << "'";
            return nullptr;
        }

        return backend;
    } catch (const InferenceException& e) {
        // Translate load failures into the nullptr contract both downstream
        // consumers already handle, instead of terminating the process.
        LOG(ERROR) << "setup_inference_engine: " << e.what();
        return nullptr;
    } catch (const std::exception& e) {
        // Vendor SDKs throw their own exception types on load failure (e.g.
        // cv::Exception for an unreadable model file); honor the same contract.
        LOG(ERROR) << "setup_inference_engine: " << e.what();
        return nullptr;
    }
}

std::unique_ptr<InferenceInterface> setup_inference_engine(const std::string& model_path, bool use_gpu,
                                                           size_t batch_size,
                                                           const std::vector<std::vector<int64_t>>& input_sizes) {
    EngineOptions options;
    options.model_path = model_path;
    options.use_gpu = use_gpu;
    options.batch_size = batch_size;
    options.input_sizes = input_sizes;
    return setup_inference_engine(options);
}
