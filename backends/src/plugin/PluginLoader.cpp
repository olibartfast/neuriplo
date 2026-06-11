#include "PluginLoader.hpp"

#include <dlfcn.h>
#include <filesystem>
#include <glog/logging.h>
#include <mutex>
#include <set>
#include <tuple>

namespace {

constexpr size_t kErrorBufferSize = 1024;

// Plugin handles are intentionally never dlclose()d: backend objects and the
// api structs they hand out must stay valid for the process lifetime.
struct PluginState {
    std::mutex mutex;
    std::set<std::string> loaded_paths;
    std::vector<PluginBackendDescriptor> descriptors;
};

PluginState& plugin_state() {
    static PluginState state;
    return state;
}

void host_log(neuriplo_log_severity_t severity, const char* message, void* /*user_data*/) {
    switch (severity) {
    case NEURIPLO_LOG_WARNING:
        LOG(WARNING) << "[plugin] " << message;
        break;
    case NEURIPLO_LOG_ERROR:
        LOG(ERROR) << "[plugin] " << message;
        break;
    case NEURIPLO_LOG_INFO:
    default:
        LOG(INFO) << "[plugin] " << message;
        break;
    }
}

neuriplo_host_services_t host_services() {
    neuriplo_host_services_t services{};
    services.struct_size = sizeof(neuriplo_host_services_t);
    services.log = &host_log;
    services.log_user_data = nullptr;
    return services;
}

bool load_plugin_locked(PluginState& state, const std::string& library_path) {
    const std::string canonical = std::filesystem::weakly_canonical(library_path).string();
    if (state.loaded_paths.count(canonical) != 0) {
        return true;
    }

    // RTLD_LOCAL keeps the plugin's framework symbols (ORT, TensorRT, ggml,
    // ...) out of the global namespace so plugins cannot collide with each
    // other or with compiled-in backends.
    void* handle = dlopen(canonical.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        LOG(WARNING) << "skipping plugin " << canonical << ": " << dlerror();
        return false;
    }

    auto entry = reinterpret_cast<neuriplo_plugin_get_api_v1_fn>(dlsym(handle, NEURIPLO_PLUGIN_ENTRY_SYMBOL));
    if (entry == nullptr) {
        LOG(WARNING) << "skipping plugin " << canonical << ": missing " << NEURIPLO_PLUGIN_ENTRY_SYMBOL;
        dlclose(handle);
        return false;
    }

    const neuriplo_plugin_api_v1* api = entry();
    if (api == nullptr) {
        LOG(WARNING) << "skipping plugin " << canonical << ": entry point returned null";
        dlclose(handle);
        return false;
    }
    if (api->abi_version != NEURIPLO_PLUGIN_ABI_VERSION) {
        LOG(WARNING) << "skipping plugin " << canonical << ": ABI version " << api->abi_version << " != host "
                     << NEURIPLO_PLUGIN_ABI_VERSION;
        dlclose(handle);
        return false;
    }
    if (api->backend_id == nullptr || api->create == nullptr || api->destroy == nullptr || api->infer == nullptr ||
        api->get_metadata == nullptr || api->release_outputs == nullptr) {
        LOG(WARNING) << "skipping plugin " << canonical << ": incomplete api table";
        dlclose(handle);
        return false;
    }

    for (const PluginBackendDescriptor& existing : state.descriptors) {
        if (existing.id == api->backend_id) {
            LOG(WARNING) << "skipping plugin " << canonical << ": backend id '" << api->backend_id
                         << "' already provided by " << existing.library_path;
            dlclose(handle);
            return false;
        }
    }

    PluginBackendDescriptor descriptor;
    descriptor.id = api->backend_id;
    descriptor.display_name = api->display_name != nullptr ? api->display_name : api->backend_id;
    descriptor.force_gpu = api->force_gpu != 0;
    descriptor.library_path = canonical;
    descriptor.api = api;
    state.descriptors.push_back(std::move(descriptor));
    state.loaded_paths.insert(canonical);
    LOG(INFO) << "loaded backend plugin '" << api->backend_id << "' from " << canonical;
    return true;
}

// Bridges a plugin backend behind the existing InferenceInterface so every
// in-process consumer (ModelRunner, decorators, serving adapters) works
// unchanged.
class PluginBackendAdapter final : public InferenceInterface {
  public:
    PluginBackendAdapter(const PluginBackendDescriptor& descriptor, neuriplo_backend_t* handle,
                         const std::string& model_path, bool use_gpu, size_t batch_size,
                         const std::vector<std::vector<int64_t>>& input_sizes)
        : InferenceInterface(model_path, use_gpu, batch_size, input_sizes), descriptor_(descriptor), handle_(handle) {
        state_ = BackendState::Ready;
        populate_metadata();
    }

    ~PluginBackendAdapter() override {
        if (handle_ != nullptr) {
            descriptor_.api->destroy(handle_);
        }
    }

    PluginBackendAdapter(const PluginBackendAdapter&) = delete;
    PluginBackendAdapter& operator=(const PluginBackendAdapter&) = delete;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override {
        std::vector<neuriplo_input_buffer_t> buffers;
        buffers.reserve(input_tensors.size());
        for (const auto& tensor : input_tensors) {
            neuriplo_input_buffer_t buffer{};
            buffer.data = tensor.data();
            buffer.size_bytes = tensor.size();
            buffers.push_back(buffer);
        }

        neuriplo_output_tensor_t* tensors = nullptr;
        size_t count = 0;
        char error[kErrorBufferSize] = {0};
        start_timer();
        const int rc =
            descriptor_.api->infer(handle_, buffers.data(), buffers.size(), &tensors, &count, error, sizeof(error));
        end_timer();
        if (rc != 0) {
            throw InferenceExecutionException(std::string(descriptor_.id) +
                                              " plugin: " + (error[0] != '\0' ? error : "inference failed"));
        }

        std::vector<std::vector<TensorElement>> outputs;
        std::vector<std::vector<int64_t>> shapes;
        outputs.reserve(count);
        shapes.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            outputs.push_back(to_elements(tensors[i]));
            shapes.emplace_back(tensors[i].shape, tensors[i].shape + tensors[i].ndim);
        }
        descriptor_.api->release_outputs(handle_, tensors, count);
        return std::make_tuple(std::move(outputs), std::move(shapes));
    }

  private:
    void populate_metadata() {
        neuriplo_metadata_t metadata{};
        if (descriptor_.api->get_metadata(handle_, &metadata) != 0) {
            throw ModelLoadException(std::string(descriptor_.id) + " plugin: metadata query failed");
        }
        for (size_t i = 0; i < metadata.n_inputs; ++i) {
            const neuriplo_layer_info_t& layer = metadata.inputs[i];
            inference_metadata_.addInput(layer.name, std::vector<int64_t>(layer.shape, layer.shape + layer.ndim),
                                         layer.batch_size);
        }
        for (size_t i = 0; i < metadata.n_outputs; ++i) {
            const neuriplo_layer_info_t& layer = metadata.outputs[i];
            inference_metadata_.addOutput(layer.name, std::vector<int64_t>(layer.shape, layer.shape + layer.ndim),
                                          layer.batch_size);
        }
    }

    static std::vector<TensorElement> to_elements(const neuriplo_output_tensor_t& tensor) {
        std::vector<TensorElement> elements;
        auto widen = [&](auto sample) {
            using Element = decltype(sample);
            const auto* typed = reinterpret_cast<const Element*>(tensor.data);
            const size_t count = tensor.size_bytes / sizeof(Element);
            elements.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                elements.emplace_back(typed[i]);
            }
        };
        switch (tensor.dtype) {
        case NEURIPLO_DTYPE_FP32:
            widen(float{});
            break;
        case NEURIPLO_DTYPE_INT32:
            widen(int32_t{});
            break;
        case NEURIPLO_DTYPE_INT64:
            widen(int64_t{});
            break;
        case NEURIPLO_DTYPE_UINT8:
            widen(uint8_t{});
            break;
        }
        return elements;
    }

    PluginBackendDescriptor descriptor_;
    neuriplo_backend_t* handle_ = nullptr;
};

} // namespace

size_t load_backend_plugins(const std::string& directory) {
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec)) {
        return 0;
    }

    PluginState& state = plugin_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    size_t loaded = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const std::string filename = entry.path().filename().string();
        if (filename.rfind("libneuriplo_backend_", 0) != 0 || entry.path().extension() != ".so") {
            continue;
        }
        const size_t before = state.descriptors.size();
        if (load_plugin_locked(state, entry.path().string()) && state.descriptors.size() > before) {
            ++loaded;
        }
    }
    return loaded;
}

bool load_backend_plugin(const std::string& library_path) {
    PluginState& state = plugin_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return load_plugin_locked(state, library_path);
}

const std::vector<PluginBackendDescriptor>& get_plugin_backends() noexcept { return plugin_state().descriptors; }

const PluginBackendDescriptor* find_plugin_backend(std::string_view id) noexcept {
    for (const PluginBackendDescriptor& descriptor : plugin_state().descriptors) {
        if (id == descriptor.id) {
            return &descriptor;
        }
    }
    return nullptr;
}

std::unique_ptr<InferenceInterface> create_plugin_backend(const PluginBackendDescriptor& descriptor,
                                                          const std::string& model_path, bool use_gpu,
                                                          size_t batch_size,
                                                          const std::vector<std::vector<int64_t>>& input_sizes) {
    std::vector<std::vector<int64_t>> shapes = input_sizes;
    std::vector<neuriplo_shape_t> shape_views;
    shape_views.reserve(shapes.size());
    for (const auto& shape : shapes) {
        neuriplo_shape_t view{};
        view.dims = shape.data();
        view.ndim = shape.size();
        shape_views.push_back(view);
    }

    neuriplo_engine_options_t options{};
    options.struct_size = sizeof(options);
    options.model_path = model_path.c_str();
    options.use_gpu = (use_gpu || descriptor.force_gpu) ? 1 : 0;
    options.batch_size = batch_size;
    options.input_sizes = shape_views.empty() ? nullptr : shape_views.data();
    options.n_input_sizes = shape_views.size();

    const neuriplo_host_services_t services = host_services();
    char error[kErrorBufferSize] = {0};
    neuriplo_backend_t* handle = descriptor.api->create(&options, &services, error, sizeof(error));
    if (handle == nullptr) {
        LOG(ERROR) << "plugin backend '" << descriptor.id
                   << "' failed to create: " << (error[0] != '\0' ? error : "unknown error");
        return nullptr;
    }

    try {
        return std::make_unique<PluginBackendAdapter>(descriptor, handle, model_path, use_gpu, batch_size, input_sizes);
    } catch (const std::exception& e) {
        LOG(ERROR) << "plugin backend '" << descriptor.id << "': " << e.what();
        descriptor.api->destroy(handle);
        return nullptr;
    }
}
