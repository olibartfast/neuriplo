// Consumer C ABI (include/neuriplo/neuriplo_c.h) -- implementation.
//
// Group 1a implements: the error model, the infallible functions, and
// neuriplo_engine_create up to backend resolution (steps 1-3 of [D-11]),
// plus _destroy / _backend_id. Group 1b adds the backend list
// (neuriplo_available_backends and friends) and engine construction (steps
// 4-5 of [D-11]). Metadata, infer, result, and log stay stubbed
// (NEURIPLO_STATUS_UNIMPLEMENTED / an inert value) for later groups
// (specs/2026-09-25-consumer-c-abi).
//
// The signatures below are the contract and must stay exactly as declared
// (NEURIPLO_CALL and NEURIPLO_NOEXCEPT repeated, NEURIPLO_C_API not
// repeated).

#include "neuriplo/neuriplo_c.h"

#include "BackendRuntimeRegistry.hpp"
#include "InferenceBackendSetup.hpp"
#include "neuriplo_c_internal.hpp"
#include "plugin/PluginLoader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;
// Set when storing a new message into g_last_error itself threw (e.g. an
// allocation failure); neuriplo_last_error() then falls back to a static
// string instead of touching g_last_error again.
thread_local bool g_last_error_unavailable = false;

} // namespace

namespace neuriplo_capi {

void set_last_error(const char* message) noexcept {
    try {
        g_last_error = (message != nullptr) ? message : "";
        g_last_error_unavailable = false;
    } catch (...) {
        g_last_error_unavailable = true;
    }
}

void clear_last_error() noexcept {
    try {
        g_last_error.clear();
        g_last_error_unavailable = false;
    } catch (...) {
        g_last_error_unavailable = true;
    }
}

neuriplo_status_t fail(neuriplo_status_t status, const char* message) noexcept {
    set_last_error(message);
    return status;
}

neuriplo_status_t fail_prefixed(neuriplo_status_t status, const char* function, const char* detail) noexcept {
    try {
        const std::string message =
            std::string(function != nullptr ? function : "") + ": " + (detail != nullptr ? detail : "");
        return fail(status, message.c_str());
    } catch (...) {
        g_last_error_unavailable = true;
        return status;
    }
}

neuriplo_status_t invalid_argument(const char* function, const char* detail) noexcept {
    return fail_prefixed(NEURIPLO_STATUS_INVALID_ARGUMENT, function, detail);
}

} // namespace neuriplo_capi

// ---------------------------------------------------------------------------
// Version, status, and errors
// ---------------------------------------------------------------------------

uint32_t NEURIPLO_CALL neuriplo_api_version(void) NEURIPLO_NOEXCEPT { return NEURIPLO_C_API_VERSION; }

const char* NEURIPLO_CALL neuriplo_status_string(neuriplo_status_t status) NEURIPLO_NOEXCEPT {
    switch (status) {
    case NEURIPLO_STATUS_OK:
        return "ok";
    case NEURIPLO_STATUS_INVALID_ARGUMENT:
        return "invalid argument";
    case NEURIPLO_STATUS_BACKEND_NOT_FOUND:
        return "backend not found";
    case NEURIPLO_STATUS_MODEL_LOAD:
        return "model load failed";
    case NEURIPLO_STATUS_INFERENCE:
        return "inference failed";
    case NEURIPLO_STATUS_OUT_OF_MEMORY:
        return "out of memory";
    case NEURIPLO_STATUS_INTERNAL:
        return "internal error";
    case NEURIPLO_STATUS_UNIMPLEMENTED:
        return "not implemented";
    default:
        return "unknown status";
    }
}

const char* NEURIPLO_CALL neuriplo_last_error(void) NEURIPLO_NOEXCEPT {
    if (g_last_error_unavailable) {
        return "neuriplo: last-error message unavailable";
    }
    return g_last_error.c_str();
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

neuriplo_status_t NEURIPLO_CALL neuriplo_set_log_callback(neuriplo_log_callback_t /*callback*/,
                                                          neuriplo_log_level_t /*min_level*/,
                                                          void* /*user_data*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

// ---------------------------------------------------------------------------
// Backend discovery
// ---------------------------------------------------------------------------

struct neuriplo_backend_list_t {
    std::vector<std::string> ids;
};

neuriplo_status_t NEURIPLO_CALL neuriplo_available_backends(const char* plugin_dir,
                                                            neuriplo_backend_list_t** out_list) NEURIPLO_NOEXCEPT {
    if (out_list == nullptr) {
        return neuriplo_capi::invalid_argument("neuriplo_available_backends", "out_list is NULL");
    }
    *out_list = nullptr;
    return neuriplo_capi::guarded(neuriplo_capi::GuardContext::Other, "neuriplo_available_backends",
                                  [&]() -> neuriplo_status_t {
                                      auto list = std::make_unique<neuriplo_backend_list_t>();
                                      list->ids = available_backend_ids(plugin_dir != nullptr ? plugin_dir : "");
                                      *out_list = list.release();
                                      neuriplo_capi::clear_last_error();
                                      return NEURIPLO_STATUS_OK;
                                  });
}

neuriplo_status_t NEURIPLO_CALL neuriplo_backend_list_count(const neuriplo_backend_list_t* list,
                                                            size_t* out_count) NEURIPLO_NOEXCEPT {
    if (out_count != nullptr) {
        *out_count = 0;
    }
    if (list == nullptr) {
        return neuriplo_capi::invalid_argument("neuriplo_backend_list_count", "list is NULL");
    }
    if (out_count == nullptr) {
        return neuriplo_capi::invalid_argument("neuriplo_backend_list_count", "out_count is NULL");
    }
    *out_count = list->ids.size();
    neuriplo_capi::clear_last_error();
    return NEURIPLO_STATUS_OK;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_backend_list_get(const neuriplo_backend_list_t* list, size_t index,
                                                          const char** out_id) NEURIPLO_NOEXCEPT {
    if (out_id != nullptr) {
        *out_id = nullptr;
    }
    if (list == nullptr) {
        return neuriplo_capi::invalid_argument("neuriplo_backend_list_get", "list is NULL");
    }
    if (out_id == nullptr) {
        return neuriplo_capi::invalid_argument("neuriplo_backend_list_get", "out_id is NULL");
    }
    if (index >= list->ids.size()) {
        return neuriplo_capi::invalid_argument("neuriplo_backend_list_get", "index is out of range");
    }
    *out_id = list->ids[index].c_str();
    neuriplo_capi::clear_last_error();
    return NEURIPLO_STATUS_OK;
}

void NEURIPLO_CALL neuriplo_backend_list_release(neuriplo_backend_list_t* list) NEURIPLO_NOEXCEPT { delete list; }

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

struct neuriplo_engine_t {
    std::unique_ptr<InferenceInterface> backend;
    std::string backend_id;
};

neuriplo_status_t NEURIPLO_CALL neuriplo_engine_create(const neuriplo_engine_config_t* config,
                                                       neuriplo_engine_t** out_engine) NEURIPLO_NOEXCEPT {
    return neuriplo_capi::guarded(
        neuriplo_capi::GuardContext::Create, "neuriplo_engine_create", [&]() -> neuriplo_status_t {
            // Step 1: validate, in order. Read no field before struct_size
            // has been checked.
            if (out_engine == nullptr) {
                return neuriplo_capi::invalid_argument("neuriplo_engine_create", "out_engine is NULL");
            }
            *out_engine = nullptr;
            if (config == nullptr) {
                return neuriplo_capi::invalid_argument("neuriplo_engine_create", "config is NULL");
            }
            if (config->struct_size < sizeof(neuriplo_engine_config_t)) {
                return neuriplo_capi::invalid_argument("neuriplo_engine_create",
                                                       "struct_size is smaller than the size this library requires");
            }
            if (config->struct_size > sizeof(neuriplo_engine_config_t)) {
                const auto* bytes = reinterpret_cast<const unsigned char*>(config);
                for (uint32_t i = static_cast<uint32_t>(sizeof(neuriplo_engine_config_t)); i < config->struct_size;
                     ++i) {
                    if (bytes[i] != 0) {
                        return neuriplo_capi::invalid_argument(
                            "neuriplo_engine_create",
                            "struct_size covers a non-zero field this library version does not know");
                    }
                }
            }
            if (config->model_path == nullptr) {
                return neuriplo_capi::invalid_argument("neuriplo_engine_create", "model_path is NULL");
            }
            if (config->input_sizes == nullptr && config->n_input_sizes > 0) {
                return neuriplo_capi::invalid_argument("neuriplo_engine_create",
                                                       "input_sizes is NULL with n_input_sizes > 0");
            }
            for (size_t i = 0; i < config->n_input_sizes; ++i) {
                const neuriplo_dims_t& entry = config->input_sizes[i];
                if (entry.dims == nullptr && entry.ndim > 0) {
                    return neuriplo_capi::invalid_argument("neuriplo_engine_create",
                                                           "an input_sizes entry has dims NULL and ndim > 0");
                }
            }

            // Step 2: scan for backends first, then resolve the id.
            const std::string plugin_dir = (config->plugin_dir != nullptr) ? config->plugin_dir : "";
            const std::vector<std::string> available = available_backend_ids(plugin_dir);

            std::string resolved_id;
            bool have_resolved_id = false;
            if (config->backend_id != nullptr && config->backend_id[0] != '\0') {
                resolved_id = config->backend_id;
                have_resolved_id = true;
            } else if (const BackendRuntimeRegistration* registration = get_compiled_backend_registration();
                       registration != nullptr) {
                resolved_id = registration->id;
                have_resolved_id = true;
            } else {
                const PluginBackendSnapshot plugins = get_plugin_backends();
                if (!plugins.empty()) {
                    resolved_id = plugins.front().id;
                    have_resolved_id = true;
                }
            }

            // Step 3: the resolved id must be available.
            bool found = false;
            if (have_resolved_id) {
                for (const std::string& id : available) {
                    if (id == resolved_id) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                std::string ids;
                for (const std::string& id : available) {
                    if (!ids.empty()) {
                        ids += ", ";
                    }
                    ids += id;
                }
                std::string message = "neuriplo_engine_create: backend '";
                message += have_resolved_id ? resolved_id : "default";
                message += "' is not available; available backends: ";
                message += ids.empty() ? "(none)" : ids;
                return neuriplo_capi::fail(NEURIPLO_STATUS_BACKEND_NOT_FOUND, message.c_str());
            }

            // Step 4: build EngineOptions and construct the backend. plugin_dir
            // is always "" here: step 2 already scanned config->plugin_dir, and
            // scanning it again would re-log and re-dlopen every rejected
            // plugin ([D-18], amended by the specifier 2026-09-26).
            EngineOptions options;
            options.model_path = config->model_path;
            options.backend_id = resolved_id;
            options.use_gpu = config->use_gpu != 0;
            options.batch_size = (config->batch_size == 0) ? 1 : config->batch_size;
            options.plugin_dir = "";
            options.input_sizes.reserve(config->n_input_sizes);
            for (size_t i = 0; i < config->n_input_sizes; ++i) {
                const neuriplo_dims_t& entry = config->input_sizes[i];
                if (entry.ndim == 0) {
                    options.input_sizes.emplace_back();
                } else {
                    options.input_sizes.emplace_back(entry.dims, entry.dims + entry.ndim);
                }
            }

            std::unique_ptr<InferenceInterface> backend;
            try {
                backend = setup_inference_engine(options);
            } catch (const std::bad_alloc& e) {
                // returned directly (not rethrown) to avoid cppcheck's
                // throwInNoexceptFunction from throwing inside a lambda
                // invoked under guarded()'s noexcept boundary.
                return neuriplo_capi::fail_prefixed(NEURIPLO_STATUS_OUT_OF_MEMORY, "neuriplo_engine_create", e.what());
            } catch (const std::exception& e) {
                std::string message = "neuriplo_engine_create: backend '";
                message += resolved_id;
                message += "' failed to load model '";
                message += config->model_path;
                message += "': ";
                message += e.what();
                // Safe to build this message here only because this lambda always
                // runs under guarded(): any bad_alloc during construction above is
                // caught separately and mapped to OUT_OF_MEMORY. Do not copy this
                // pattern outside a guarded() context.
                return neuriplo_capi::fail(NEURIPLO_STATUS_MODEL_LOAD, message.c_str());
            }

            // Step 5: a NULL backend without an exception is also a load
            // failure; both the id and the path are required in the message.
            if (!backend) {
                std::string message = "neuriplo_engine_create: backend '";
                message += resolved_id;
                message += "' failed to load model '";
                message += config->model_path;
                message += "'";
                return neuriplo_capi::fail(NEURIPLO_STATUS_MODEL_LOAD, message.c_str());
            }

            auto engine = std::make_unique<neuriplo_engine_t>();
            engine->backend = std::move(backend);
            engine->backend_id = resolved_id;
            *out_engine = engine.release();
            neuriplo_capi::clear_last_error();
            return NEURIPLO_STATUS_OK;
        });
}

void NEURIPLO_CALL neuriplo_engine_destroy(neuriplo_engine_t* engine) NEURIPLO_NOEXCEPT { delete engine; }

neuriplo_status_t NEURIPLO_CALL neuriplo_engine_backend_id(const neuriplo_engine_t* engine,
                                                           const char** out_id) NEURIPLO_NOEXCEPT {
    if (out_id != nullptr) {
        *out_id = nullptr;
    }
    if (engine == nullptr) {
        return neuriplo_capi::invalid_argument("neuriplo_engine_backend_id", "engine is NULL");
    }
    if (out_id == nullptr) {
        return neuriplo_capi::invalid_argument("neuriplo_engine_backend_id", "out_id is NULL");
    }
    *out_id = engine->backend_id.c_str();
    neuriplo_capi::clear_last_error();
    return NEURIPLO_STATUS_OK;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_engine_input_count(const neuriplo_engine_t* /*engine*/,
                                                            size_t* /*out_count*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_engine_input(const neuriplo_engine_t* /*engine*/, size_t /*index*/,
                                                      const neuriplo_tensor_info_t** /*out_info*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_engine_output_count(const neuriplo_engine_t* /*engine*/,
                                                             size_t* /*out_count*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_engine_output(const neuriplo_engine_t* /*engine*/, size_t /*index*/,
                                                       const neuriplo_tensor_info_t** /*out_info*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_infer(neuriplo_engine_t* /*engine*/, const neuriplo_input_view_t* /*inputs*/,
                                               size_t /*n_inputs*/,
                                               neuriplo_result_t** /*out_result*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_result_output_count(const neuriplo_result_t* /*result*/,
                                                             size_t* /*out_count*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_result_output(const neuriplo_result_t* /*result*/, size_t /*index*/,
                                                       const neuriplo_tensor_view_t** /*out_view*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

void NEURIPLO_CALL neuriplo_result_release(neuriplo_result_t* /*result*/) NEURIPLO_NOEXCEPT {}
