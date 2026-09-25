/* Build-time checks for include/neuriplo/neuriplo_c.h ([V-1], [V-11]),
 * compiled as strict C99 with warnings as errors (see CMakeLists.txt).
 *
 *  - neuriplo_c.h compiles as strict C99 (the include audit in CMakeLists.txt
 *    separately guarantees it includes nothing but <stddef.h>/<stdint.h>).
 *  - It can share a translation unit with plugin_abi.h: no name collides.
 *  - Struct layouts, enum widths, and enumerator values are pinned. An
 *    accidental reorder, retype, removal, or renumbering fails the build.
 *
 * Offsets are written in terms of P = sizeof(void*) and S = sizeof(size_t) so
 * the same assertions hold on LP64, LLP64 (Windows x64), and ILP32 targets.
 * The v1 structs are laid out with no internal padding on all three.
 *
 * Owned by the specifier; read-only to every implementer. */

#include "neuriplo/neuriplo_c.h"
#include "neuriplo/plugin_abi.h"

#include <stddef.h>

/* C99 has no _Static_assert: a negative array size is the portable spelling. */
#define CAPI_STATIC_ASSERT(cond, tag) typedef char capi_static_assert_##tag[(cond) ? 1 : -1]

#define P sizeof(void*)
#define S sizeof(size_t)

CAPI_STATIC_ASSERT(NEURIPLO_C_API_VERSION == 1u, api_version_is_1);

/* Enums are 32 bits wide whatever the compiler's enum sizing. */
CAPI_STATIC_ASSERT(sizeof(neuriplo_status_t) == 4, status_is_32_bit);
CAPI_STATIC_ASSERT(sizeof(neuriplo_tensor_dtype_t) == 4, dtype_is_32_bit);
CAPI_STATIC_ASSERT(sizeof(neuriplo_log_level_t) == 4, log_level_is_32_bit);

/* Enumerator values are ABI. */
CAPI_STATIC_ASSERT(NEURIPLO_STATUS_OK == 0, status_ok);
CAPI_STATIC_ASSERT(NEURIPLO_STATUS_INVALID_ARGUMENT == 1, status_invalid_argument);
CAPI_STATIC_ASSERT(NEURIPLO_STATUS_BACKEND_NOT_FOUND == 2, status_backend_not_found);
CAPI_STATIC_ASSERT(NEURIPLO_STATUS_MODEL_LOAD == 3, status_model_load);
CAPI_STATIC_ASSERT(NEURIPLO_STATUS_INFERENCE == 4, status_inference);
CAPI_STATIC_ASSERT(NEURIPLO_STATUS_OUT_OF_MEMORY == 5, status_out_of_memory);
CAPI_STATIC_ASSERT(NEURIPLO_STATUS_INTERNAL == 6, status_internal);
CAPI_STATIC_ASSERT(NEURIPLO_STATUS_UNIMPLEMENTED == 7, status_unimplemented);
CAPI_STATIC_ASSERT(NEURIPLO_TENSOR_DTYPE_FLOAT32 == 0, dtype_float32);
CAPI_STATIC_ASSERT(NEURIPLO_TENSOR_DTYPE_INT32 == 1, dtype_int32);
CAPI_STATIC_ASSERT(NEURIPLO_TENSOR_DTYPE_INT64 == 2, dtype_int64);
CAPI_STATIC_ASSERT(NEURIPLO_TENSOR_DTYPE_UINT8 == 3, dtype_uint8);
CAPI_STATIC_ASSERT(NEURIPLO_TENSOR_DTYPE_INT8 == 4, dtype_int8);
CAPI_STATIC_ASSERT(NEURIPLO_TENSOR_DTYPE_BOOL == 5, dtype_bool);
CAPI_STATIC_ASSERT(NEURIPLO_LOG_LEVEL_INFO == 0, level_info);
CAPI_STATIC_ASSERT(NEURIPLO_LOG_LEVEL_WARNING == 1, level_warning);
CAPI_STATIC_ASSERT(NEURIPLO_LOG_LEVEL_ERROR == 2, level_error);

/* The output dtypes agree with the plugin ABI's (both mirror TensorDtype). */
CAPI_STATIC_ASSERT((int)NEURIPLO_TENSOR_DTYPE_FLOAT32 == (int)NEURIPLO_DTYPE_FP32, dtype_matches_plugin_fp32);
CAPI_STATIC_ASSERT((int)NEURIPLO_TENSOR_DTYPE_INT32 == (int)NEURIPLO_DTYPE_INT32, dtype_matches_plugin_int32);
CAPI_STATIC_ASSERT((int)NEURIPLO_TENSOR_DTYPE_INT64 == (int)NEURIPLO_DTYPE_INT64, dtype_matches_plugin_int64);
CAPI_STATIC_ASSERT((int)NEURIPLO_TENSOR_DTYPE_UINT8 == (int)NEURIPLO_DTYPE_UINT8, dtype_matches_plugin_uint8);

/* neuriplo_dims_t (frozen) */
CAPI_STATIC_ASSERT(offsetof(neuriplo_dims_t, dims) == 0, dims_dims);
CAPI_STATIC_ASSERT(offsetof(neuriplo_dims_t, ndim) == P, dims_ndim);
CAPI_STATIC_ASSERT(sizeof(neuriplo_dims_t) == P + S, dims_size);

/* neuriplo_engine_config_t (v1) */
CAPI_STATIC_ASSERT(offsetof(neuriplo_engine_config_t, struct_size) == 0, config_struct_size);
CAPI_STATIC_ASSERT(offsetof(neuriplo_engine_config_t, use_gpu) == 4, config_use_gpu);
CAPI_STATIC_ASSERT(offsetof(neuriplo_engine_config_t, backend_id) == 8, config_backend_id);
CAPI_STATIC_ASSERT(offsetof(neuriplo_engine_config_t, model_path) == 8 + P, config_model_path);
CAPI_STATIC_ASSERT(offsetof(neuriplo_engine_config_t, batch_size) == 8 + 2 * P, config_batch_size);
CAPI_STATIC_ASSERT(offsetof(neuriplo_engine_config_t, input_sizes) == 8 + 2 * P + S, config_input_sizes);
CAPI_STATIC_ASSERT(offsetof(neuriplo_engine_config_t, n_input_sizes) == 8 + 3 * P + S, config_n_input_sizes);
CAPI_STATIC_ASSERT(offsetof(neuriplo_engine_config_t, plugin_dir) == 8 + 3 * P + 2 * S, config_plugin_dir);
CAPI_STATIC_ASSERT(sizeof(neuriplo_engine_config_t) == 8 + 4 * P + 2 * S, config_size);

/* neuriplo_tensor_info_t (v1) */
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_info_t, struct_size) == 0, info_struct_size);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_info_t, dtype) == 4, info_dtype);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_info_t, name) == 8, info_name);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_info_t, shape) == 8 + P, info_shape);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_info_t, ndim) == 8 + 2 * P, info_ndim);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_info_t, batch_size) == 8 + 2 * P + S, info_batch_size);
CAPI_STATIC_ASSERT(sizeof(neuriplo_tensor_info_t) == 8 + 2 * P + 2 * S, info_size);

/* neuriplo_input_view_t (frozen) */
CAPI_STATIC_ASSERT(offsetof(neuriplo_input_view_t, data) == 0, input_data);
CAPI_STATIC_ASSERT(offsetof(neuriplo_input_view_t, size_bytes) == P, input_size_bytes);
CAPI_STATIC_ASSERT(sizeof(neuriplo_input_view_t) == P + S, input_size);

/* neuriplo_tensor_view_t (v1) */
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_view_t, struct_size) == 0, view_struct_size);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_view_t, dtype) == 4, view_dtype);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_view_t, data) == 8, view_data);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_view_t, size_bytes) == 8 + P, view_size_bytes);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_view_t, element_count) == 8 + P + S, view_element_count);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_view_t, shape) == 8 + P + 2 * S, view_shape);
CAPI_STATIC_ASSERT(offsetof(neuriplo_tensor_view_t, ndim) == 8 + 2 * P + 2 * S, view_ndim);
CAPI_STATIC_ASSERT(sizeof(neuriplo_tensor_view_t) == 8 + 2 * P + 3 * S, view_size);

/* Every declared function has exactly the declared type: taking its address
 * into a pointer of the spelled-out type fails to compile on any drift. */
typedef uint32_t(NEURIPLO_CALL* capi_api_version_fn)(void);
typedef const char*(NEURIPLO_CALL* capi_status_string_fn)(neuriplo_status_t);
typedef const char*(NEURIPLO_CALL* capi_last_error_fn)(void);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_set_log_callback_fn)(neuriplo_log_callback_t, neuriplo_log_level_t,
                                                                   void*);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_available_backends_fn)(const char*, neuriplo_backend_list_t**);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_backend_list_count_fn)(const neuriplo_backend_list_t*, size_t*);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_backend_list_get_fn)(const neuriplo_backend_list_t*, size_t,
                                                                   const char**);
typedef void(NEURIPLO_CALL* capi_backend_list_release_fn)(neuriplo_backend_list_t*);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_engine_create_fn)(const neuriplo_engine_config_t*, neuriplo_engine_t**);
typedef void(NEURIPLO_CALL* capi_engine_destroy_fn)(neuriplo_engine_t*);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_engine_backend_id_fn)(const neuriplo_engine_t*, const char**);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_engine_count_fn)(const neuriplo_engine_t*, size_t*);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_engine_info_fn)(const neuriplo_engine_t*, size_t,
                                                              const neuriplo_tensor_info_t**);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_infer_fn)(neuriplo_engine_t*, const neuriplo_input_view_t*, size_t,
                                                        neuriplo_result_t**);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_result_count_fn)(const neuriplo_result_t*, size_t*);
typedef neuriplo_status_t(NEURIPLO_CALL* capi_result_output_fn)(const neuriplo_result_t*, size_t,
                                                                const neuriplo_tensor_view_t**);
typedef void(NEURIPLO_CALL* capi_result_release_fn)(neuriplo_result_t*);

/* Referenced, never called: this object is not linked into anything. */
void neuriplo_capi_header_check_signatures(void);
void neuriplo_capi_header_check_signatures(void) {
    capi_api_version_fn api_version = neuriplo_api_version;
    capi_status_string_fn status_string = neuriplo_status_string;
    capi_last_error_fn last_error = neuriplo_last_error;
    capi_set_log_callback_fn set_log_callback = neuriplo_set_log_callback;
    capi_available_backends_fn available_backends = neuriplo_available_backends;
    capi_backend_list_count_fn backend_list_count = neuriplo_backend_list_count;
    capi_backend_list_get_fn backend_list_get = neuriplo_backend_list_get;
    capi_backend_list_release_fn backend_list_release = neuriplo_backend_list_release;
    capi_engine_create_fn engine_create = neuriplo_engine_create;
    capi_engine_destroy_fn engine_destroy = neuriplo_engine_destroy;
    capi_engine_backend_id_fn engine_backend_id = neuriplo_engine_backend_id;
    capi_engine_count_fn engine_input_count = neuriplo_engine_input_count;
    capi_engine_info_fn engine_input = neuriplo_engine_input;
    capi_engine_count_fn engine_output_count = neuriplo_engine_output_count;
    capi_engine_info_fn engine_output = neuriplo_engine_output;
    capi_infer_fn infer = neuriplo_infer;
    capi_result_count_fn result_output_count = neuriplo_result_output_count;
    capi_result_output_fn result_output = neuriplo_result_output;
    capi_result_release_fn result_release = neuriplo_result_release;
    (void)api_version;
    (void)status_string;
    (void)last_error;
    (void)set_log_callback;
    (void)available_backends;
    (void)backend_list_count;
    (void)backend_list_get;
    (void)backend_list_release;
    (void)engine_create;
    (void)engine_destroy;
    (void)engine_backend_id;
    (void)engine_input_count;
    (void)engine_input;
    (void)engine_output_count;
    (void)engine_output;
    (void)infer;
    (void)result_output_count;
    (void)result_output;
    (void)result_release;
}
