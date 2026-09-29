#ifndef NEURIPLO_C_H
#define NEURIPLO_C_H

/* neuriplo consumer C ABI -- the stable boundary for third-party applications.
 *
 * Direction: applications call this header; libneuriplo implements it. (The
 * other C header, plugin_abi.h, is the opposite direction -- backends plugged
 * into neuriplo -- and consumers never need it. The two headers may be
 * included in the same translation unit; no name is shared.)
 *
 * This header is C99 and C++ clean, includes only <stddef.h> and <stdint.h>,
 * and exchanges no C++ type, STL container, or exception with the caller.
 * Spec: specs/2026-09-25-consumer-c-abi/.
 *
 * ---------------------------------------------------------------------------
 * General contract (applies to every function below unless its comment says
 * otherwise)
 * ---------------------------------------------------------------------------
 *
 * Errors.
 *   Every function either returns neuriplo_status_t or is documented as
 *   infallible. No C++ exception ever leaves the library. A NULL handle, a
 *   NULL out-pointer, an out-of-range index, or a malformed struct yields
 *   NEURIPLO_STATUS_INVALID_ARGUMENT, never a crash (a dangling or foreign
 *   pointer that is not NULL cannot be detected and is undefined behaviour).
 *
 * Last error.
 *   Every status-returning function overwrites the calling thread's last-error
 *   message: with a human-readable description on failure, with the empty
 *   string on success. Infallible functions never touch it. Read it with
 *   neuriplo_last_error(). The message is per thread: a failure on one thread
 *   never changes what another thread reads. For INVALID_ARGUMENT the message
 *   starts with the name of the function that rejected the call.
 *
 * Out-pointers.
 *   On failure, a non-NULL out-pointer argument is set to NULL (handles and
 *   views) or 0 (counts) before returning; on success it is written. A caller
 *   therefore never needs to initialise out-variables, and never sees a stale
 *   value after a failure.
 *
 * Ownership.
 *   Handles (neuriplo_engine_t, neuriplo_result_t, neuriplo_backend_list_t)
 *   are created by the library and released exactly once by the caller with
 *   the matching destroy/release function. Pointers the library returns
 *   (strings, views, shape arrays) are owned by the library and stay valid for
 *   the lifetime stated on each function; the caller never frees them. Memory
 *   the caller passes in (configs, input views, strings) is only read, and
 *   only for the duration of the call; the library keeps no pointer to it.
 *
 * Extensible structs.
 *   Structs whose first member is `struct_size` are extensible: fields are only
 *   ever appended. For a struct the CALLER fills in (neuriplo_engine_config_t)
 *   the caller sets struct_size = sizeof(the struct as the caller compiled it)
 *   and zero-initialises the whole struct (memset) before setting fields. The
 *   library rejects struct_size smaller than the version-1 size, reads only
 *   the fields it knows, and accepts a larger struct_size (a caller built
 *   against a newer header) only if every byte beyond the fields it knows is
 *   zero -- a non-zero byte there is a request this library version cannot
 *   honour, and is rejected with INVALID_ARGUMENT rather than silently
 *   ignored. For a struct the LIBRARY returns (neuriplo_tensor_info_t,
 *   neuriplo_tensor_view_t) the library sets struct_size to the size it
 *   filled; a caller reads an appended field only if struct_size covers it.
 *   Array-element types without struct_size (neuriplo_dims_t,
 *   neuriplo_input_view_t) are frozen: changing them needs a new type.
 *
 * Enums.
 *   Every enum is 32 bits wide (the *_MAX_ENUM_ sentinel forces it, whatever
 *   the compiler's enum sizing). Adding an enumerator is a compatible change;
 *   callers must treat an unknown value as "unsupported", not as an error in
 *   the library.
 *
 * Threads.
 *   Every function may be called from any thread. Functions that do not take
 *   an engine or result handle are thread-safe. Different engines are fully
 *   independent and may be used concurrently. Calls on the SAME engine:
 *   neuriplo_infer calls are serialised by a per-engine lock (they are safe
 *   but do not run in parallel); metadata queries are lock-free and safe
 *   concurrently with anything except neuriplo_engine_destroy. Destroying or
 *   releasing a handle while another thread still uses it is undefined
 *   behaviour -- the caller orders destroy/release after every other use.
 *
 * Versioning.
 *   NEURIPLO_C_API_VERSION is independent of the plugin ABI version. Adding a
 *   function, appending a field behind struct_size, or adding an enumerator
 *   is a compatible change and does not bump it. Removing or changing
 *   anything that exists bumps it.
 */

#include <stddef.h>
#include <stdint.h>

#define NEURIPLO_C_API_VERSION 1u

/* Export and calling-convention macros. Declared here, once: the library's
 * definitions (src/neuriplo_c.cpp) must not repeat NEURIPLO_C_API -- MSVC
 * rejects dllexport added on a redeclaration -- but must repeat NEURIPLO_CALL
 * and NEURIPLO_NOEXCEPT exactly as declared.
 *
 *   NEURIPLO_C_BUILDING  defined only while compiling libneuriplo itself
 *   NEURIPLO_C_STATIC    define when linking a static libneuriplo (Windows)
 *
 * NEURIPLO_CALL is __cdecl on Windows so that C# [DllImport] (whose default is
 * StdCall on x86) and other FFIs can state it explicitly; it is a no-op on
 * x64 and on every other platform. */
#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(NEURIPLO_C_STATIC)
#define NEURIPLO_C_API
#elif defined(NEURIPLO_C_BUILDING)
#define NEURIPLO_C_API __declspec(dllexport)
#else
#define NEURIPLO_C_API __declspec(dllimport)
#endif
#define NEURIPLO_CALL __cdecl
#else
#if defined(__GNUC__) || defined(__clang__)
#define NEURIPLO_C_API __attribute__((visibility("default")))
#else
#define NEURIPLO_C_API
#endif
#define NEURIPLO_CALL
#endif

/* Seen by C++ callers only: states in the type system that nothing throws.
 * The library's definitions must carry it too, so a definition that could
 * leak an exception at least terminates instead of unwinding into C. */
#ifdef __cplusplus
#define NEURIPLO_NOEXCEPT noexcept
#else
#define NEURIPLO_NOEXCEPT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------ */
/* Status codes                                                              */
/* ------------------------------------------------------------------------ */

typedef enum neuriplo_status_t {
    /* The call succeeded. */
    NEURIPLO_STATUS_OK = 0,
    /* A NULL handle or out-pointer, NULL data with a non-zero size, an
     * out-of-range index or level, a malformed struct_size, or a non-zero
     * field this library version does not know. */
    NEURIPLO_STATUS_INVALID_ARGUMENT = 1,
    /* The requested backend_id is neither compiled in nor provided by a loaded
     * plugin (or no backend exists at all). The message lists the ids that are
     * available. */
    NEURIPLO_STATUS_BACKEND_NOT_FOUND = 2,
    /* The backend exists but could not create an engine for the model: model
     * file missing or unreadable, backend or plugin create failure, or plugin
     * metadata the host rejected. */
    NEURIPLO_STATUS_MODEL_LOAD = 3,
    /* Inference ran and failed: backend error, inputs the backend rejected, or
     * plugin outputs the host rejected. */
    NEURIPLO_STATUS_INFERENCE = 4,
    /* An allocation failed inside the library. */
    NEURIPLO_STATUS_OUT_OF_MEMORY = 5,
    /* An unexpected failure that fits no other status (a library defect or an
     * exception of unknown type). */
    NEURIPLO_STATUS_INTERNAL = 6,
    /* This build of the library does not implement the function. No v1
     * function returns it from a released library. */
    NEURIPLO_STATUS_UNIMPLEMENTED = 7,
    NEURIPLO_STATUS_MAX_ENUM_ = 0x7FFFFFFF
} neuriplo_status_t;

/* ------------------------------------------------------------------------ */
/* Element types                                                             */
/* ------------------------------------------------------------------------ */

/* Element type of a tensor. Values 0-3 match neuriplo's output dtypes; metadata
 * may additionally report INT8 and BOOL (wire formats a backend widens on
 * decode). Outputs are only ever FLOAT32, INT32, INT64, or UINT8 in v1. */
typedef enum neuriplo_tensor_dtype_t {
    NEURIPLO_TENSOR_DTYPE_FLOAT32 = 0,
    NEURIPLO_TENSOR_DTYPE_INT32 = 1,
    NEURIPLO_TENSOR_DTYPE_INT64 = 2,
    NEURIPLO_TENSOR_DTYPE_UINT8 = 3,
    NEURIPLO_TENSOR_DTYPE_INT8 = 4,
    NEURIPLO_TENSOR_DTYPE_BOOL = 5,
    NEURIPLO_TENSOR_DTYPE_MAX_ENUM_ = 0x7FFFFFFF
} neuriplo_tensor_dtype_t;

/* ------------------------------------------------------------------------ */
/* Logging                                                                   */
/* ------------------------------------------------------------------------ */

typedef enum neuriplo_log_level_t {
    NEURIPLO_LOG_LEVEL_INFO = 0,
    NEURIPLO_LOG_LEVEL_WARNING = 1,
    NEURIPLO_LOG_LEVEL_ERROR = 2,
    NEURIPLO_LOG_LEVEL_MAX_ENUM_ = 0x7FFFFFFF
} neuriplo_log_level_t;

/* Receives one log message. `message` is NUL-terminated, carries no trailing
 * newline, and is valid only for the duration of the call (copy it to keep
 * it). `user_data` is the pointer given to neuriplo_set_log_callback.
 *
 * Invoked synchronously on whichever thread logged -- possibly several threads
 * at once, possibly a thread the application did not create -- so the
 * callback must be thread-safe. It must not call any neuriplo function
 * (doing so may deadlock) and must not unwind (no C++ exception, no longjmp)
 * out of the call. */
typedef void(NEURIPLO_CALL* neuriplo_log_callback_t)(neuriplo_log_level_t level, const char* message, void* user_data);

/* ------------------------------------------------------------------------ */
/* Handles and structs                                                       */
/* ------------------------------------------------------------------------ */

/* A ready-to-run inference engine. Created by neuriplo_engine_create,
 * destroyed by neuriplo_engine_destroy. */
typedef struct neuriplo_engine_t neuriplo_engine_t;

/* The outputs of one neuriplo_infer call. Created by neuriplo_infer, released
 * by neuriplo_result_release. Independent of the engine that produced it: it
 * may outlive that engine. */
typedef struct neuriplo_result_t neuriplo_result_t;

/* A snapshot of available backend ids. Created by neuriplo_available_backends,
 * released by neuriplo_backend_list_release. */
typedef struct neuriplo_backend_list_t neuriplo_backend_list_t;

/* A shape given by the caller: `ndim` dimensions at `dims`. `dims` may be NULL
 * only when ndim is 0. Frozen layout. */
typedef struct neuriplo_dims_t {
    const int64_t* dims;
    size_t ndim;
} neuriplo_dims_t;

/* Engine configuration, filled in by the caller. Extensible (see "Extensible
 * structs" above): memset to zero, set struct_size = sizeof(*config), then set
 * the fields you need; every zero field means "default". */
typedef struct neuriplo_engine_config_t {
    /* sizeof(neuriplo_engine_config_t) as the caller compiled it. */
    uint32_t struct_size;
    /* Non-zero requests GPU execution where the backend supports it; the
     * backend's own rules apply exactly as for the C++ EngineOptions. */
    int32_t use_gpu;
    /* Backend to use, e.g. "OPENCV_DNN", "ONNX_RUNTIME", or a plugin's id.
     * NULL or "" selects the build's default backend (the compiled-in default,
     * or the first loaded plugin when nothing is compiled in). */
    const char* backend_id;
    /* Model file path, passed to the backend unchanged. Must not be NULL; ""
     * is passed through for backends that need no file. */
    const char* model_path;
    /* Batch size; 0 means the default, 1. */
    size_t batch_size;
    /* Optional input shapes for backends that need them; `n_input_sizes`
     * entries. input_sizes may be NULL only when n_input_sizes is 0. */
    const neuriplo_dims_t* input_sizes;
    size_t n_input_sizes;
    /* Directory scanned for libneuriplo_backend_* plugins before the backend
     * is resolved; NULL or "" scans nothing extra. The NEURIPLO_PLUGIN_DIR
     * environment variable is honoured in addition, as in the C++ API. */
    const char* plugin_dir;
} neuriplo_engine_config_t;

/* Metadata for one model input or output, owned by the engine. Valid (same
 * address, same contents) until neuriplo_engine_destroy on that engine,
 * across any number of neuriplo_infer calls. */
typedef struct neuriplo_tensor_info_t {
    /* Size the library filled; read an appended field only if this covers it. */
    uint32_t struct_size;
    neuriplo_tensor_dtype_t dtype;
    /* Layer name, NUL-terminated, never NULL (may be ""). */
    const char* name;
    /* `ndim` dimensions; NULL only when ndim is 0. A dimension may be -1 when
     * the backend reports it as dynamic. */
    const int64_t* shape;
    size_t ndim;
    size_t batch_size;
} neuriplo_tensor_info_t;

/* One input tensor given to neuriplo_infer: `size_bytes` raw bytes at `data`,
 * laid out as the model's metadata describes (the same raw-bytes contract as
 * the C++ get_infer_results). `data` may be NULL only when size_bytes is 0.
 * Frozen layout. */
typedef struct neuriplo_input_view_t {
    const void* data;
    size_t size_bytes;
} neuriplo_input_view_t;

/* One output tensor, owned by its result. Valid (same address, same contents)
 * until neuriplo_result_release on that result -- including after the engine
 * that produced it is destroyed. */
typedef struct neuriplo_tensor_view_t {
    /* Size the library filled; read an appended field only if this covers it. */
    uint32_t struct_size;
    /* One of FLOAT32, INT32, INT64, UINT8. */
    neuriplo_tensor_dtype_t dtype;
    /* Contiguous, native-endian elements, suitably aligned for dtype. May be
     * NULL when size_bytes is 0 (a zero-element output, e.g. no detections). */
    const void* data;
    size_t size_bytes;
    /* size_bytes / element size; 0 for a zero-element output. */
    size_t element_count;
    /* `ndim` concrete dimensions (all >= 0); NULL only when ndim is 0. */
    const int64_t* shape;
    size_t ndim;
} neuriplo_tensor_view_t;

/* ------------------------------------------------------------------------ */
/* Version, status, and errors                                               */
/* ------------------------------------------------------------------------ */

/* Infallible. The NEURIPLO_C_API_VERSION the library was built with. A caller
 * compares it with the header's NEURIPLO_C_API_VERSION to detect a mismatched
 * library at run time. */
NEURIPLO_C_API uint32_t NEURIPLO_CALL neuriplo_api_version(void) NEURIPLO_NOEXCEPT;

/* Infallible. A static, non-empty, human-readable name for `status` (e.g.
 * "invalid argument"); distinct for every defined status; a generic non-empty
 * string for an unknown value. Never NULL; never freed. */
NEURIPLO_C_API const char* NEURIPLO_CALL neuriplo_status_string(neuriplo_status_t status) NEURIPLO_NOEXCEPT;

/* Infallible. The calling thread's last-error message (see "Last error"
 * above): non-empty after a status-returning call on this thread failed, ""
 * after one succeeded or before any was made. Never NULL. Valid until the
 * next status-returning neuriplo call on the same thread, or thread exit. */
NEURIPLO_C_API const char* NEURIPLO_CALL neuriplo_last_error(void) NEURIPLO_NOEXCEPT;

/* ------------------------------------------------------------------------ */
/* Logging                                                                   */
/* ------------------------------------------------------------------------ */

/* Installs `callback` as the process-wide log receiver for messages from
 * neuriplo and its plugins at `min_level` or above, replacing any previous
 * callback. A NULL callback removes the current one (min_level and user_data
 * are then ignored). Without a callback -- the initial state -- logging
 * behaves exactly as before this API existed (glog's default: stderr). A
 * callback is additional: it does not silence that default output. The
 * application never has to initialise glog.
 *
 * When this function returns, no invocation of the previous callback is in
 * progress and none will start, so its user_data may be freed (and a managed
 * delegate may be collected) immediately afterwards. Install the callback
 * before creating the first engine or listing backends to see messages from
 * the first plugin scan (plugins are scanned once per process).
 *
 * Thread-safe. Must not be called from inside a log callback.
 * Errors: INVALID_ARGUMENT if callback is non-NULL and min_level is not a
 * defined neuriplo_log_level_t. */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_set_log_callback(neuriplo_log_callback_t callback,
                                                                         neuriplo_log_level_t min_level,
                                                                         void* user_data) NEURIPLO_NOEXCEPT;

/* ------------------------------------------------------------------------ */
/* Backend discovery                                                         */
/* ------------------------------------------------------------------------ */

/* Scans `plugin_dir` (NULL or "" = none; NEURIPLO_PLUGIN_DIR is honoured in
 * addition) and returns the backend ids available in this process: the
 * compiled-in backends first, then loaded plugins in load order, without
 * duplicates. Plugins the host rejected are not listed. Plugins stay loaded
 * for the process lifetime.
 *
 * On success *out_list is a new list owned by the caller; release it with
 * neuriplo_backend_list_release. Thread-safe.
 * Errors: INVALID_ARGUMENT (out_list NULL), OUT_OF_MEMORY, INTERNAL. */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL
neuriplo_available_backends(const char* plugin_dir, neuriplo_backend_list_t** out_list) NEURIPLO_NOEXCEPT;

/* Number of ids in `list`.
 * Errors: INVALID_ARGUMENT (list or out_count NULL). */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_backend_list_count(const neuriplo_backend_list_t* list,
                                                                           size_t* out_count) NEURIPLO_NOEXCEPT;

/* The id at `index` (0-based), NUL-terminated, owned by `list` and valid until
 * neuriplo_backend_list_release on it.
 * Errors: INVALID_ARGUMENT (list or out_id NULL, index >= count). */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_backend_list_get(const neuriplo_backend_list_t* list,
                                                                         size_t index,
                                                                         const char** out_id) NEURIPLO_NOEXCEPT;

/* Infallible. Releases `list` and every string it handed out. NULL is a
 * no-op. Releasing the same list twice is undefined behaviour. */
NEURIPLO_C_API void NEURIPLO_CALL neuriplo_backend_list_release(neuriplo_backend_list_t* list) NEURIPLO_NOEXCEPT;

/* ------------------------------------------------------------------------ */
/* Engine lifecycle                                                          */
/* ------------------------------------------------------------------------ */

/* Creates an engine and loads the model eagerly: on success the engine is
 * ready for metadata queries and inference, with no separate load step.
 * Scans config->plugin_dir first, then resolves config->backend_id, then
 * creates the backend -- the same path, host validation, and backend
 * selection rules as the C++ setup_inference_engine(EngineOptions).
 *
 * On success *out_engine is a new engine owned by the caller; destroy it with
 * neuriplo_engine_destroy. Thread-safe (concurrent creates are allowed).
 *
 * Errors:
 *   INVALID_ARGUMENT  config or out_engine NULL; config->struct_size smaller
 *                     than the v1 size, or larger with a non-zero byte beyond
 *                     the fields this library knows; model_path NULL;
 *                     input_sizes NULL with n_input_sizes > 0; an entry of
 *                     input_sizes with dims NULL and ndim > 0.
 *   BACKEND_NOT_FOUND the backend id (or the default, when none is given)
 *                     is not available; the message names the requested id
 *                     and lists the available ones.
 *   MODEL_LOAD        the backend was found but failed to create or load;
 *                     the message names the backend id and the model path.
 *                     Detail the backend logged goes to the log (see
 *                     neuriplo_set_log_callback).
 *   OUT_OF_MEMORY, INTERNAL. */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_engine_create(const neuriplo_engine_config_t* config,
                                                                      neuriplo_engine_t** out_engine) NEURIPLO_NOEXCEPT;

/* Infallible. Destroys `engine` and the backend instance it owns, and
 * invalidates every metadata view and string it handed out. Results already
 * obtained from it stay valid. NULL is a no-op. Must not run concurrently with
 * any other call on the same engine; destroying twice is undefined. */
NEURIPLO_C_API void NEURIPLO_CALL neuriplo_engine_destroy(neuriplo_engine_t* engine) NEURIPLO_NOEXCEPT;

/* The id of the backend this engine actually runs on (the resolved default
 * when config->backend_id was NULL or ""), NUL-terminated, owned by the
 * engine, valid until neuriplo_engine_destroy.
 * Errors: INVALID_ARGUMENT (engine or out_id NULL). */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_engine_backend_id(const neuriplo_engine_t* engine,
                                                                          const char** out_id) NEURIPLO_NOEXCEPT;

/* ------------------------------------------------------------------------ */
/* Metadata                                                                  */
/* ------------------------------------------------------------------------ */

/* Number of model inputs. Lock-free; see "Threads".
 * Errors: INVALID_ARGUMENT (engine or out_count NULL). */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_engine_input_count(const neuriplo_engine_t* engine,
                                                                           size_t* out_count) NEURIPLO_NOEXCEPT;

/* Metadata of input `index` (0-based, in model order). *out_info points into
 * the engine and stays valid until neuriplo_engine_destroy.
 * Errors: INVALID_ARGUMENT (engine or out_info NULL, index >= input count). */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_engine_input(
    const neuriplo_engine_t* engine, size_t index, const neuriplo_tensor_info_t** out_info) NEURIPLO_NOEXCEPT;

/* Number of model outputs. Lock-free; see "Threads".
 * Errors: INVALID_ARGUMENT (engine or out_count NULL). */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_engine_output_count(const neuriplo_engine_t* engine,
                                                                            size_t* out_count) NEURIPLO_NOEXCEPT;

/* Metadata of output `index` (0-based, in model order). *out_info points into
 * the engine and stays valid until neuriplo_engine_destroy.
 * Errors: INVALID_ARGUMENT (engine or out_info NULL, index >= output count). */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_engine_output(
    const neuriplo_engine_t* engine, size_t index, const neuriplo_tensor_info_t** out_info) NEURIPLO_NOEXCEPT;

/* ------------------------------------------------------------------------ */
/* Inference                                                                 */
/* ------------------------------------------------------------------------ */

/* Runs one inference on `n_inputs` input tensors (normally one per model
 * input, in model order). The input bytes are read during the call only.
 * Whether the count and sizes fit the model is the backend's decision:
 * a mismatch it rejects is INFERENCE, not INVALID_ARGUMENT.
 *
 * On success *out_result is a new result owned by the caller; release it with
 * neuriplo_result_release. The result holds the outputs neuriplo's raw C++
 * path produced, moved -- the C API adds no output copy.
 *
 * Calls on the same engine are serialised (a per-engine lock); calls on
 * different engines run in parallel.
 *
 * Errors:
 *   INVALID_ARGUMENT  engine or out_result NULL; inputs NULL with
 *                     n_inputs > 0; an input with data NULL and
 *                     size_bytes > 0.
 *   INFERENCE         the backend failed or rejected the inputs, or a plugin
 *                     returned outputs the host rejected.
 *   OUT_OF_MEMORY, INTERNAL. */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_infer(neuriplo_engine_t* engine,
                                                              const neuriplo_input_view_t* inputs, size_t n_inputs,
                                                              neuriplo_result_t** out_result) NEURIPLO_NOEXCEPT;

/* Number of output tensors in `result`. A result is immutable: any number of
 * threads may read it concurrently.
 * Errors: INVALID_ARGUMENT (result or out_count NULL). */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_result_output_count(const neuriplo_result_t* result,
                                                                            size_t* out_count) NEURIPLO_NOEXCEPT;

/* Output `index` (0-based, in the order the backend produced them). *out_view
 * points into the result and stays valid until neuriplo_result_release.
 * Errors: INVALID_ARGUMENT (result or out_view NULL, index >= output count). */
NEURIPLO_C_API neuriplo_status_t NEURIPLO_CALL neuriplo_result_output(
    const neuriplo_result_t* result, size_t index, const neuriplo_tensor_view_t** out_view) NEURIPLO_NOEXCEPT;

/* Infallible. Releases `result` and every view it handed out. NULL is a no-op.
 * Must not run concurrently with a read of the same result; releasing twice
 * is undefined behaviour. */
NEURIPLO_C_API void NEURIPLO_CALL neuriplo_result_release(neuriplo_result_t* result) NEURIPLO_NOEXCEPT;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NEURIPLO_C_H */
