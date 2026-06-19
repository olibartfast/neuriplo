#include "InferenceBackendSetup.hpp"

#include "BackendRuntimeRegistry.hpp"
#include "decorators/LoggingBackend.hpp"
#include "decorators/ProfilingBackend.hpp"
#include "plugin/PluginLoader.hpp"

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

// Scans the explicit plugin directory and NEURIPLO_PLUGIN_DIR (idempotent per
// library, so calling this on every setup is cheap).
void load_configured_plugins(const std::string& plugin_dir) {
    if (!plugin_dir.empty()) {
        load_backend_plugins(plugin_dir);
    }
    if (const char* env_dir = std::getenv("NEURIPLO_PLUGIN_DIR")) {
        if (env_dir[0] != '\0') {
            load_backend_plugins(env_dir);
        }
    }
}

std::string known_backend_ids(const std::string& plugin_dir) {
    std::string ids;
    for (const std::string& id : available_backend_ids(plugin_dir)) {
        if (!ids.empty()) {
            ids += ", ";
        }
        ids += id;
    }
    return ids;
}

std::unique_ptr<InferenceInterface> finalize_backend(std::unique_ptr<InferenceInterface> backend,
                                                     const std::string& model_path) {
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
}

} // namespace

std::vector<std::string> available_backend_ids(const std::string& plugin_dir) {
    load_configured_plugins(plugin_dir);

    std::vector<std::string> ids;
    for (const BackendRuntimeRegistration& registration : get_registered_backends()) {
        ids.emplace_back(registration.id);
    }
    for (const PluginBackendDescriptor& descriptor : get_plugin_backends()) {
        bool already_registered = false;
        for (const std::string& id : ids) {
            if (id == descriptor.id) {
                already_registered = true;
                break;
            }
        }
        if (!already_registered) {
            ids.push_back(descriptor.id);
        }
    }
    return ids;
}

std::unique_ptr<InferenceInterface> setup_inference_engine(const EngineOptions& options) {
    load_configured_plugins(options.plugin_dir);

    // Compiled-in backends win id collisions with plugins (the loader already
    // warns when a plugin id is shadowed).
    const BackendRuntimeRegistration* registration = nullptr;
    const PluginBackendDescriptor* plugin = nullptr;
    if (options.backend_id.empty()) {
        registration = get_compiled_backend_registration();
        if (registration == nullptr && !get_plugin_backends().empty()) {
            plugin = &get_plugin_backends().front();
        }
    } else {
        registration = find_backend_registration(options.backend_id);
        if (registration == nullptr) {
            plugin = find_plugin_backend(options.backend_id);
        }
        if (registration == nullptr && plugin == nullptr) {
            LOG(ERROR) << "setup_inference_engine: backend '" << options.backend_id
                       << "' is not available in this process; available backends: "
                       << known_backend_ids(options.plugin_dir);
            return nullptr;
        }
    }

    if (plugin != nullptr) {
        auto backend = create_plugin_backend(*plugin, options.model_path, options.use_gpu, options.batch_size,
                                             options.input_sizes);
        return finalize_backend(std::move(backend), options.model_path);
    }

    if (registration == nullptr || registration->create_factory == nullptr) {
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
        return finalize_backend(std::move(backend), options.model_path);
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
