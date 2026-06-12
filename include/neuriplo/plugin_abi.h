#ifndef NEURIPLO_PLUGIN_ABI_H
#define NEURIPLO_PLUGIN_ABI_H

/* Stable C ABI for neuriplo backend plugins.
 *
 * A backend plugin is a shared library exporting exactly one symbol,
 * neuriplo_plugin_get_api_v1, returning a static neuriplo_plugin_api_v1
 * describing the backend and its entry points. The host loads plugins with
 * dlopen(RTLD_NOW | RTLD_LOCAL) so each plugin's framework dependencies stay
 * private to it.
 *
 * Rules:
 *  - No C++ types, STL containers, or exceptions cross this boundary.
 *  - Memory allocated by the plugin is released by the plugin
 *    (release_outputs / destroy); the host never frees plugin memory.
 *  - Increment NEURIPLO_PLUGIN_ABI_VERSION on any breaking change to these
 *    structs; the host skips plugins whose version does not match.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEURIPLO_PLUGIN_ABI_VERSION 1u

#define NEURIPLO_PLUGIN_ENTRY_SYMBOL "neuriplo_plugin_get_api_v1"

/* Mirrors the TensorElement alternatives used by in-process backends. */
typedef enum neuriplo_dtype_t {
    NEURIPLO_DTYPE_FP32 = 0,
    NEURIPLO_DTYPE_INT32 = 1,
    NEURIPLO_DTYPE_INT64 = 2,
    NEURIPLO_DTYPE_UINT8 = 3
} neuriplo_dtype_t;

/* Log severities map to the host's logging backend (glog). */
typedef enum neuriplo_log_severity_t {
    NEURIPLO_LOG_INFO = 0,
    NEURIPLO_LOG_WARNING = 1,
    NEURIPLO_LOG_ERROR = 2
} neuriplo_log_severity_t;

typedef void (*neuriplo_log_fn)(neuriplo_log_severity_t severity, const char* message, void* user_data);

/* Services the host provides to plugins. Plugins must not initialize their
 * own logging; they report through the host callback. */
typedef struct neuriplo_host_services_t {
    uint32_t struct_size; /* sizeof(neuriplo_host_services_t), for extension */
    neuriplo_log_fn log;  /* may be NULL: plugin stays silent */
    void* log_user_data;
} neuriplo_host_services_t;

typedef struct neuriplo_shape_t {
    const int64_t* dims;
    size_t ndim;
} neuriplo_shape_t;

/* C mirror of EngineOptions. */
typedef struct neuriplo_engine_options_t {
    uint32_t struct_size; /* sizeof(neuriplo_engine_options_t), for extension */
    const char* model_path;
    int use_gpu;
    size_t batch_size;
    const neuriplo_shape_t* input_sizes;
    size_t n_input_sizes;
} neuriplo_engine_options_t;

typedef struct neuriplo_layer_info_t {
    const char* name;
    const int64_t* shape;
    size_t ndim;
    size_t batch_size;
} neuriplo_layer_info_t;

typedef struct neuriplo_metadata_t {
    const neuriplo_layer_info_t* inputs;
    size_t n_inputs;
    const neuriplo_layer_info_t* outputs;
    size_t n_outputs;
} neuriplo_metadata_t;

/* Raw input bytes; the backend interprets layout from model metadata, exactly
 * as the in-process get_infer_results contract does. */
typedef struct neuriplo_input_buffer_t {
    const uint8_t* data;
    size_t size_bytes;
} neuriplo_input_buffer_t;

/* Typed contiguous output buffer. */
typedef struct neuriplo_output_tensor_t {
    neuriplo_dtype_t dtype;
    const void* data;
    size_t size_bytes;
    const int64_t* shape;
    size_t ndim;
} neuriplo_output_tensor_t;

/* Opaque backend instance handle. */
typedef struct neuriplo_backend_t neuriplo_backend_t;

typedef struct neuriplo_plugin_api_v1 {
    uint32_t abi_version; /* NEURIPLO_PLUGIN_ABI_VERSION */
    const char* backend_id;
    const char* display_name;
    int force_gpu;

    /* Create and eagerly load a backend instance. Returns NULL on failure and
     * writes a message into error (if error_size > 0). */
    neuriplo_backend_t* (*create)(const neuriplo_engine_options_t* options, const neuriplo_host_services_t* host,
                                  char* error, size_t error_size);

    void (*destroy)(neuriplo_backend_t* backend);

    /* Metadata views stay valid until the next get_metadata call on the same
     * handle, or destroy. Returns 0 on success. */
    int (*get_metadata)(neuriplo_backend_t* backend, neuriplo_metadata_t* out_metadata);

    /* Run inference. On success (return 0) *out_tensors points to an array of
     * *out_count tensors owned by the plugin; the host must release it with
     * release_outputs. On failure writes a message into error. */
    int (*infer)(neuriplo_backend_t* backend, const neuriplo_input_buffer_t* inputs, size_t n_inputs,
                 neuriplo_output_tensor_t** out_tensors, size_t* out_count, char* error, size_t error_size);

    void (*release_outputs)(neuriplo_backend_t* backend, neuriplo_output_tensor_t* tensors, size_t count);
} neuriplo_plugin_api_v1;

/* The single symbol every plugin exports. */
const neuriplo_plugin_api_v1* neuriplo_plugin_get_api_v1(void);

typedef const neuriplo_plugin_api_v1* (*neuriplo_plugin_get_api_v1_fn)(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NEURIPLO_PLUGIN_ABI_H */
