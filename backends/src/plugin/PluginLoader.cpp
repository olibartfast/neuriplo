#include "PluginLoader.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <filesystem>
#include <glog/logging.h>
#include <limits>
#include <mutex>
#include <set>
#include <tuple>
#include <type_traits>

namespace {

constexpr size_t kErrorBufferSize = 1024;

// Dynamic loading, spelled for both platforms. The operation is the same shape
// either way; the one behavioural difference worth naming is that dlerror()
// latches its message until read, while GetLastError() is overwritten by the
// next API call -- so on Windows the message has to be taken at the failure
// site, before anything else runs.
#ifdef _WIN32

using PluginHandle = HMODULE;

// LOAD_WITH_ALTERED_SEARCH_PATH looks for the plugin's dependencies in the
// plugin's own directory first, the nearest Windows equivalent of a .so
// resolving the framework libraries shipped beside it. It is only honoured for
// an absolute path, which is what callers pass -- the path is made canonical
// before it gets here.
PluginHandle plugin_open(const std::string& path) {
    return LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}

neuriplo_plugin_get_api_v1_fn plugin_entry(PluginHandle handle) {
    return reinterpret_cast<neuriplo_plugin_get_api_v1_fn>(GetProcAddress(handle, NEURIPLO_PLUGIN_ENTRY_SYMBOL));
}

void plugin_close(PluginHandle handle) { FreeLibrary(handle); }

std::string plugin_error() {
    const DWORD code = GetLastError();
    char* buffer = nullptr;
    const DWORD length =
        FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                       nullptr, code, 0, reinterpret_cast<LPSTR>(&buffer), 0, nullptr);
    std::string message = length != 0 ? std::string(buffer, length) : "error " + std::to_string(code);
    LocalFree(buffer);
    // FormatMessage terminates system messages with CRLF.
    while (!message.empty() && (message.back() == '\r' || message.back() == '\n')) {
        message.pop_back();
    }
    return message;
}

constexpr const char* kPluginExtension = ".dll";

#else

using PluginHandle = void*;

// RTLD_LOCAL keeps the plugin's framework symbols (ORT, TensorRT, ggml, ...)
// out of the global namespace so plugins cannot collide with each other or
// with compiled-in backends.
PluginHandle plugin_open(const std::string& path) { return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL); }

neuriplo_plugin_get_api_v1_fn plugin_entry(PluginHandle handle) {
    return reinterpret_cast<neuriplo_plugin_get_api_v1_fn>(dlsym(handle, NEURIPLO_PLUGIN_ENTRY_SYMBOL));
}

void plugin_close(PluginHandle handle) { dlclose(handle); }

std::string plugin_error() {
    const char* message = dlerror();
    return message != nullptr ? message : "unknown error";
}

constexpr const char* kPluginExtension = ".so";

#endif

// Upper bound on a layer's rank that the host accepts from a plugin. No
// real model needs anything close to this; it exists to reject malformed or
// hostile ndim values before they are used to construct a shape vector.
constexpr size_t kMaxLayerRank = 16;

// neuriplo_dtype_t has no fixed underlying type, so loading a value outside
// its enumerator range as that enum type is undefined behaviour in C++17. A
// hostile or buggy plugin can put any bit pattern in a dtype field, so every
// check here compares the underlying integer, never the enum value itself.
using NeuriploDtypeUnderlying = std::underlying_type_t<neuriplo_dtype_t>;

bool dtype_is_valid(neuriplo_dtype_t dtype) {
    const auto value = static_cast<NeuriploDtypeUnderlying>(dtype);
    return value == static_cast<NeuriploDtypeUnderlying>(NEURIPLO_DTYPE_FP32) ||
           value == static_cast<NeuriploDtypeUnderlying>(NEURIPLO_DTYPE_INT32) ||
           value == static_cast<NeuriploDtypeUnderlying>(NEURIPLO_DTYPE_INT64) ||
           value == static_cast<NeuriploDtypeUnderlying>(NEURIPLO_DTYPE_UINT8);
}

bool metadata_dtype_is_valid(neuriplo_dtype_t dtype) { return dtype_is_valid(dtype); }

// Only ever called after metadata_dtype_is_valid (or the output-side
// dtype_is_valid) has accepted `dtype`; the throw below is defence in depth
// so an unreachable value can never silently become Float32.
TensorDataType metadata_dtype_from_abi(neuriplo_dtype_t dtype) {
    switch (dtype) {
    case NEURIPLO_DTYPE_FP32:
        return TensorDataType::Float32;
    case NEURIPLO_DTYPE_INT32:
        return TensorDataType::Int32;
    case NEURIPLO_DTYPE_INT64:
        return TensorDataType::Int64;
    case NEURIPLO_DTYPE_UINT8:
        return TensorDataType::UInt8;
    }
    throw ModelLoadException("plugin metadata invalid: unreachable element_type");
}

// Plugin handles are intentionally never unloaded: backend objects and the
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

    PluginHandle handle = plugin_open(canonical);
    if (handle == nullptr) {
        LOG(WARNING) << "skipping plugin " << canonical << ": " << plugin_error();
        return false;
    }

    const neuriplo_plugin_get_api_v1_fn entry = plugin_entry(handle);
    if (entry == nullptr) {
        LOG(WARNING) << "skipping plugin " << canonical << ": missing " << NEURIPLO_PLUGIN_ENTRY_SYMBOL;
        plugin_close(handle);
        return false;
    }

    const neuriplo_plugin_api_v1* api = entry();
    if (api == nullptr) {
        LOG(WARNING) << "skipping plugin " << canonical << ": entry point returned null";
        plugin_close(handle);
        return false;
    }
    if (api->abi_version != NEURIPLO_PLUGIN_ABI_VERSION) {
        LOG(WARNING) << "skipping plugin " << canonical << ": ABI version " << api->abi_version << " != host "
                     << NEURIPLO_PLUGIN_ABI_VERSION;
        plugin_close(handle);
        return false;
    }
    if (api->backend_id == nullptr || api->create == nullptr || api->destroy == nullptr || api->infer == nullptr ||
        api->get_metadata == nullptr || api->release_outputs == nullptr) {
        LOG(WARNING) << "skipping plugin " << canonical << ": incomplete api table";
        plugin_close(handle);
        return false;
    }

    for (const PluginBackendDescriptor& existing : state.descriptors) {
        if (existing.id == api->backend_id) {
            LOG(WARNING) << "skipping plugin " << canonical << ": backend id '" << api->backend_id
                         << "' already provided by " << existing.library_path;
            plugin_close(handle);
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

// Owns an `infer` call's out-parameters from the moment infer returns 0 until
// this guard goes out of scope, and releases them exactly once via
// release_outputs -- on a normal return, on a validation rejection, or on any
// exception thrown while copying (e.g. std::bad_alloc). Must only be
// constructed after infer has returned 0; the caller is responsible for not
// constructing one when infer failed, since a failed call never handed the
// host anything to release.
class OutputReleaseGuard {
  public:
    OutputReleaseGuard(const neuriplo_plugin_api_v1* api, neuriplo_backend_t* handle, neuriplo_output_tensor_t* tensors,
                       size_t count)
        : api_(api), handle_(handle), tensors_(tensors), count_(count) {}

    OutputReleaseGuard(const OutputReleaseGuard&) = delete;
    OutputReleaseGuard& operator=(const OutputReleaseGuard&) = delete;
    OutputReleaseGuard(OutputReleaseGuard&&) = delete;
    OutputReleaseGuard& operator=(OutputReleaseGuard&&) = delete;

    ~OutputReleaseGuard() noexcept {
        // Called unconditionally, even when the plugin reported count > 0 but
        // handed back a NULL tensors pointer: the ABI puts no restriction on
        // release_outputs's arguments for that case, a conforming plugin (see
        // the fixture backend) already tolerates a NULL tensors pointer here,
        // and skipping the call would need its own special case for no
        // benefit -- release_outputs is the one place that knows whether
        // there is anything to free.
        api_->release_outputs(handle_, tensors_, count_);
    }

  private:
    const neuriplo_plugin_api_v1* api_;
    neuriplo_backend_t* handle_;
    neuriplo_output_tensor_t* tensors_;
    size_t count_;
};

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

    // Primary path: one ABI call, outputs copied once as typed bytes.
    std::vector<RawOutputTensor>
    get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) override {
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
            // infer failed: it handed nothing to the host, so there is
            // nothing to release.
            throw InferenceExecutionException(std::string(descriptor_.id) +
                                              " plugin: " + (error[0] != '\0' ? error : "inference failed"));
        }

        // infer returned 0: from here on the plugin's out-parameters must be
        // released exactly once, however this function leaves -- including
        // through a validation rejection or a std::bad_alloc while copying.
        const OutputReleaseGuard release_guard(descriptor_.api, handle_, tensors, count);

        if (tensors == nullptr && count > 0) {
            throw InferenceExecutionException(std::string(descriptor_.id) + " plugin: output 0: tensors is null");
        }

        std::vector<RawOutputTensor> outputs;
        outputs.reserve(std::min(count, inference_metadata_.getOutputs().size()));
        for (size_t i = 0; i < count; ++i) {
            validate_output_tensor(tensors[i], i);
            RawOutputTensor output;
            output.dtype = static_cast<TensorDtype>(tensors[i].dtype);
            if (tensors[i].size_bytes != 0) {
                const auto* data = static_cast<const uint8_t*>(tensors[i].data);
                output.bytes.assign(data, data + tensors[i].size_bytes);
            }
            output.shape.assign(tensors[i].shape, tensors[i].shape + tensors[i].ndim);
            outputs.push_back(std::move(output));
        }
        return outputs;
    }

    // Legacy variant view, derived from the raw path.
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override {
        std::vector<RawOutputTensor> raw_outputs = get_infer_results_raw(input_tensors);

        std::vector<std::vector<TensorElement>> outputs;
        std::vector<std::vector<int64_t>> shapes;
        outputs.reserve(raw_outputs.size());
        shapes.reserve(raw_outputs.size());
        for (RawOutputTensor& raw : raw_outputs) {
            outputs.push_back(to_elements(raw));
            shapes.push_back(std::move(raw.shape));
        }
        return std::make_tuple(std::move(outputs), std::move(shapes));
    }

  private:
    // Every field of every layer is validated against attacker/bug-controlled
    // plugin memory before it is used to build a std::vector or std::string;
    // nothing here trusts pointer or size values coming from the plugin.
    void validate_layer_array(const neuriplo_layer_info_t* layers, size_t count, const char* array_field) const {
        if (layers == nullptr && count > 0) {
            throw ModelLoadException(descriptor_.library_path + ": plugin metadata invalid: " + array_field +
                                     " is null");
        }
    }

    void validate_layer(const neuriplo_layer_info_t& layer, const char* layer_label) const {
        if (layer.name == nullptr) {
            throw ModelLoadException(descriptor_.library_path + ": plugin metadata invalid: " + layer_label +
                                     " has null name");
        }
        if (layer.shape == nullptr && layer.ndim > 0) {
            throw ModelLoadException(descriptor_.library_path + ": plugin metadata invalid: " + layer_label +
                                     " has null shape");
        }
        if (layer.ndim > kMaxLayerRank) {
            throw ModelLoadException(descriptor_.library_path + ": plugin metadata invalid: " + layer_label +
                                     " has out-of-bound ndim");
        }
        if (!metadata_dtype_is_valid(layer.element_type)) {
            throw ModelLoadException(descriptor_.library_path + ": plugin metadata invalid: " + layer_label +
                                     " has unknown element_type");
        }
    }

    // Every field of an output tensor is validated against attacker/bug-
    // controlled plugin memory before get_infer_results_raw reads it. `index`
    // names the tensor in rejection messages exactly as "output <index>".
    void validate_output_tensor(const neuriplo_output_tensor_t& tensor, size_t index) const {
        const std::string label = "output " + std::to_string(index);
        if (tensor.shape == nullptr && tensor.ndim > 0) {
            throw InferenceExecutionException(std::string(descriptor_.id) + " plugin: " + label + ": shape is null");
        }
        if (tensor.ndim > kMaxLayerRank) {
            throw InferenceExecutionException(std::string(descriptor_.id) + " plugin: " + label +
                                              ": ndim exceeds the maximum layer rank");
        }
        for (size_t d = 0; d < tensor.ndim; ++d) {
            if (tensor.shape[d] < 0) {
                throw InferenceExecutionException(std::string(descriptor_.id) + " plugin: " + label +
                                                  ": shape has a negative dimension");
            }
        }
        // dtype must be validated before it is used to compute an element
        // size: an out-of-range value must never reach tensor_dtype_size.
        if (!dtype_is_valid(tensor.dtype)) {
            throw InferenceExecutionException(std::string(descriptor_.id) + " plugin: " + label +
                                              ": dtype is not a recognized neuriplo_dtype_t");
        }
        const TensorDtype dtype = static_cast<TensorDtype>(tensor.dtype);
        constexpr size_t kSizeMax = std::numeric_limits<size_t>::max();
        size_t element_count = 1;
        for (size_t d = 0; d < tensor.ndim; ++d) {
            const size_t dim = static_cast<size_t>(tensor.shape[d]);
            if (dim != 0 && element_count > kSizeMax / dim) {
                throw InferenceExecutionException(std::string(descriptor_.id) + " plugin: " + label +
                                                  ": shape product overflows computing the tensor's element count");
            }
            element_count *= dim;
        }
        const size_t element_size = tensor_dtype_size(dtype);
        if (element_size != 0 && element_count > kSizeMax / element_size) {
            throw InferenceExecutionException(std::string(descriptor_.id) + " plugin: " + label +
                                              ": shape product overflows computing the expected byte size");
        }
        const size_t expected_size_bytes = element_count * element_size;
        if (tensor.size_bytes != expected_size_bytes) {
            throw InferenceExecutionException(std::string(descriptor_.id) + " plugin: " + label +
                                              ": size_bytes does not match dtype size x product(shape)");
        }
        // A zero-element tensor is conforming and may carry NULL data (e.g.
        // shape [0, 6] produced from an empty std::vector's .data()). Only a
        // non-zero byte size with NULL data is a real defect.
        if (tensor.data == nullptr && expected_size_bytes != 0) {
            throw InferenceExecutionException(std::string(descriptor_.id) + " plugin: " + label + ": data is null");
        }
    }

    void populate_metadata() {
        neuriplo_metadata_t metadata{};
        if (descriptor_.api->get_metadata(handle_, &metadata) != 0) {
            throw ModelLoadException(descriptor_.library_path + " plugin: metadata query failed");
        }

        validate_layer_array(metadata.inputs, metadata.n_inputs, "inputs");
        validate_layer_array(metadata.outputs, metadata.n_outputs, "outputs");
        for (size_t i = 0; i < metadata.n_inputs; ++i) {
            validate_layer(metadata.inputs[i], ("input layer " + std::to_string(i)).c_str());
        }
        for (size_t i = 0; i < metadata.n_outputs; ++i) {
            validate_layer(metadata.outputs[i], ("output layer " + std::to_string(i)).c_str());
        }

        for (size_t i = 0; i < metadata.n_inputs; ++i) {
            const neuriplo_layer_info_t& layer = metadata.inputs[i];
            inference_metadata_.addInput(layer.name, std::vector<int64_t>(layer.shape, layer.shape + layer.ndim),
                                         layer.batch_size, metadata_dtype_from_abi(layer.element_type));
        }
        for (size_t i = 0; i < metadata.n_outputs; ++i) {
            const neuriplo_layer_info_t& layer = metadata.outputs[i];
            inference_metadata_.addOutput(layer.name, std::vector<int64_t>(layer.shape, layer.shape + layer.ndim),
                                          layer.batch_size, metadata_dtype_from_abi(layer.element_type));
        }
    }

    static std::vector<TensorElement> to_elements(const RawOutputTensor& tensor) {
        std::vector<TensorElement> elements;
        auto widen = [&](auto sample) {
            using Element = decltype(sample);
            const auto* typed = reinterpret_cast<const Element*>(tensor.bytes.data());
            const size_t count = tensor.bytes.size() / sizeof(Element);
            elements.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                elements.emplace_back(typed[i]);
            }
        };
        switch (tensor.dtype) {
        case TensorDtype::FP32:
            widen(float{});
            break;
        case TensorDtype::INT32:
            widen(int32_t{});
            break;
        case TensorDtype::INT64:
            widen(int64_t{});
            break;
        case TensorDtype::UINT8:
            widen(uint8_t{});
            break;
        default:
            // get_infer_results_raw already rejects an unknown dtype before a
            // RawOutputTensor is ever built, so this is unreachable in
            // practice; it must fail loudly rather than yield an empty
            // vector that looks like a zero-element tensor.
            throw InferenceExecutionException("plugin output: unknown tensor dtype");
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
        if (filename.rfind("libneuriplo_backend_", 0) != 0 || entry.path().extension() != kPluginExtension) {
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
