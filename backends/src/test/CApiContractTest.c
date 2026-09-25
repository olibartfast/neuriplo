/* Consumer C ABI contract suite: the acceptance tests for
 * specs/2026-09-25-consumer-c-abi. Plain C, compiled as C99, linked only
 * against libneuriplo (plus the platform's thread and dynamic-loader
 * libraries), exercising include/neuriplo/neuriplo_c.h exactly as a
 * third-party C application would. Every case runs against the first-party
 * fixture plugins in plugin_fixtures/, so no vendor SDK or model is needed.
 *
 * Owned by the specifier and read-only to every implementer (see that
 * packet's orchestration.md): a worker implementing the C API does not edit
 * the tests that score it.
 *
 * One executable, one case per process: `CApiContractTest <Group>.<Name>`.
 * CMake reads the CAPI_CASE(...) lines of the case table at the bottom of this
 * file and registers each as the ctest test `CApi.<Group>.<Name>` -- the table
 * is the single list of cases. A process per case matters because the plugin
 * table is process-global and never unloads, and because a log callback must
 * be installed before the first plugin scan to see it.
 *
 * Groups map to the validation checks so `ctest -R` can select them:
 *   Lifecycle   version, create/destroy, backend listing   [V-1] [V-2]
 *   StructSize  struct_size rules                          [V-3]
 *   Error       create-side failure statuses               [V-6]
 *   LastError   thread-local message                       [V-7]
 *   Metadata    input/output metadata views                [V-4]
 *   Infer       inference and result ownership            [V-5]
 *   InferError  inference-side failure statuses            [V-6]
 *   Thread      concurrency and same-engine policy         [V-8]
 *   Log         log callback                               [V-9]
 *
 * A C++ exception escaping the library would terminate this process: that no
 * case ever does is part of [V-6]. */

#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE /* realpath, RTLD_NOLOAD, dup/dup2, pthreads */
#endif

#include "neuriplo/neuriplo_c.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <unistd.h>
#endif

/* CMake always defines these; the fallbacks only let static analysers that do
 * not see the build's definitions parse this file (every case then fails at
 * run time, loudly, because no fixture is found). */
#ifndef NEURIPLO_DEFAULT_BACKEND
#define NEURIPLO_DEFAULT_BACKEND "UNCONFIGURED_DEFAULT_BACKEND"
#endif
#ifndef NEURIPLO_FIXTURE_PLUGIN_DIR
#define NEURIPLO_FIXTURE_PLUGIN_DIR "unconfigured-fixture-dir"
#endif
#ifndef NEURIPLO_FIXTURE_DUPLICATE_DIR
#define NEURIPLO_FIXTURE_DUPLICATE_DIR "unconfigured-fixture-dir/duplicate"
#endif
#ifndef NEURIPLO_FIXTURE_PLUGIN_SUFFIX
#define NEURIPLO_FIXTURE_PLUGIN_SUFFIX ".so"
#endif

#define GOOD_ID "FIXTURE_GOOD"
#define SCRIPTED_ID "FIXTURE_SCRIPTED"
#define FIXTURE_DIR NEURIPLO_FIXTURE_PLUGIN_DIR
#define ELEMENTS 4

/* ------------------------------------------------------------------------ */
/* Minimal harness                                                           */
/* ------------------------------------------------------------------------ */

static int g_failures = 0;

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stdout, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);                                   \
            ++g_failures;                                                                                              \
        }                                                                                                              \
    } while (0)

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stdout, "%s:%d: REQUIRE failed: %s\n", __FILE__, __LINE__, #cond);                                 \
            ++g_failures;                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

static int status_is(neuriplo_status_t actual, neuriplo_status_t expected, const char* expr, const char* file,
                     int line) {
    if (actual == expected) {
        return 1;
    }
    fprintf(stdout, "%s:%d: %s returned %d (%s), expected %d (%s); last error: '%s'\n", file, line, expr, (int)actual,
            neuriplo_status_string(actual), (int)expected, neuriplo_status_string(expected), neuriplo_last_error());
    ++g_failures;
    return 0;
}

#define CHECK_STATUS(expr, expected) (void)status_is((expr), (expected), #expr, __FILE__, __LINE__)
#define REQUIRE_STATUS(expr, expected)                                                                                 \
    do {                                                                                                               \
        if (!status_is((expr), (expected), #expr, __FILE__, __LINE__)) {                                               \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

static int contains(const char* haystack, const char* needle) {
    return haystack != NULL && needle != NULL && strstr(haystack, needle) != NULL;
}

static int starts_with(const char* text, const char* prefix) {
    return text != NULL && strncmp(text, prefix, strlen(prefix)) == 0;
}

static int last_error_nonempty(void) {
    const char* message = neuriplo_last_error();
    return message != NULL && message[0] != '\0';
}

static int last_error_empty(void) {
    const char* message = neuriplo_last_error();
    return message != NULL && message[0] == '\0';
}

/* A non-NULL value no API may leave behind in an out-pointer after failure. */
static char g_sentinel_storage;
#define SENTINEL_PTR ((void*)&g_sentinel_storage)

/* ------------------------------------------------------------------------ */
/* Fixture helpers                                                           */
/* ------------------------------------------------------------------------ */

static void config_init(neuriplo_engine_config_t* config, const char* backend_id, const char* model_path,
                        const char* plugin_dir) {
    memset(config, 0, sizeof(*config));
    config->struct_size = (uint32_t)sizeof(*config);
    config->backend_id = backend_id;
    config->model_path = model_path;
    config->plugin_dir = plugin_dir;
}

static neuriplo_status_t create_fixture(const char* backend_id, const char* model_path, neuriplo_engine_t** out) {
    neuriplo_engine_config_t config;
    config_init(&config, backend_id, model_path, FIXTURE_DIR);
    return neuriplo_engine_create(&config, out);
}

static void fill_input(float values[ELEMENTS], float base) {
    int i;
    for (i = 0; i < ELEMENTS; ++i) {
        values[i] = base + (float)i;
    }
}

static neuriplo_status_t infer_values(neuriplo_engine_t* engine, const float values[ELEMENTS],
                                      neuriplo_result_t** out) {
    neuriplo_input_view_t input;
    input.data = values;
    input.size_bytes = ELEMENTS * sizeof(float);
    return neuriplo_infer(engine, &input, 1, out);
}

/* 1 if `result` holds exactly one FLOAT32 [1,4] tensor equal to 2 * in. No
 * harness calls: also used from worker threads. */
static int result_is_doubled(const neuriplo_result_t* result, const float in[ELEMENTS]) {
    size_t count = 0;
    const neuriplo_tensor_view_t* view = NULL;
    const float* out;
    int i;
    if (neuriplo_result_output_count(result, &count) != NEURIPLO_STATUS_OK || count != 1) {
        return 0;
    }
    if (neuriplo_result_output(result, 0, &view) != NEURIPLO_STATUS_OK || view == NULL) {
        return 0;
    }
    if (view->dtype != NEURIPLO_TENSOR_DTYPE_FLOAT32 || view->size_bytes != ELEMENTS * sizeof(float) ||
        view->element_count != ELEMENTS || view->ndim != 2 || view->shape == NULL || view->shape[0] != 1 ||
        view->shape[1] != ELEMENTS || view->data == NULL) {
        return 0;
    }
    out = (const float*)view->data;
    for (i = 0; i < ELEMENTS; ++i) {
        if (out[i] != 2.0f * in[i]) {
            return 0;
        }
    }
    return 1;
}

typedef struct fixture_counts {
    size_t handed;
    size_t released;
    size_t live;
    size_t overlapping;
} fixture_counts;

typedef void (*fixture_counters_fn)(size_t*, size_t*, size_t*);
typedef void (*fixture_overlaps_fn)(size_t*);

/* Reads a fixture module's own bookkeeping (test-only exports). The module
 * must already be loaded by the library; this never loads a second copy. */
static int read_fixture_counts(const char* module, fixture_counts* counts) {
    char path[4096];
    void* counters_symbol = NULL;
    void* overlaps_symbol = NULL;
    fixture_counters_fn counters_fn = NULL;
    fixture_overlaps_fn overlaps_fn = NULL;
    memset(counts, 0, sizeof(*counts));
    snprintf(path, sizeof(path), "%s/libneuriplo_backend_fixture_%s%s", FIXTURE_DIR, module,
             NEURIPLO_FIXTURE_PLUGIN_SUFFIX);
#ifdef _WIN32
    {
        char full[4096];
        HMODULE handle;
        if (_fullpath(full, path, sizeof(full)) == NULL) {
            return 0;
        }
        handle = GetModuleHandleA(full);
        if (handle == NULL) {
            return 0;
        }
        {
            FARPROC counters_proc = GetProcAddress(handle, "neuriplo_fixture_counters");
            FARPROC overlaps_proc = GetProcAddress(handle, "neuriplo_fixture_overlaps");
            memcpy(&counters_fn, &counters_proc, sizeof(counters_fn));
            memcpy(&overlaps_fn, &overlaps_proc, sizeof(overlaps_fn));
        }
        (void)counters_symbol;
        (void)overlaps_symbol;
    }
#else
    {
        char full[PATH_MAX];
        void* handle;
        if (realpath(path, full) == NULL) {
            return 0;
        }
        handle = dlopen(full, RTLD_NOW | RTLD_NOLOAD);
        if (handle == NULL) {
            return 0;
        }
        counters_symbol = dlsym(handle, "neuriplo_fixture_counters");
        overlaps_symbol = dlsym(handle, "neuriplo_fixture_overlaps");
        /* ISO C has no object-to-function pointer conversion; copy the bits. */
        memcpy(&counters_fn, &counters_symbol, sizeof(counters_fn));
        memcpy(&overlaps_fn, &overlaps_symbol, sizeof(overlaps_fn));
        dlclose(handle);
    }
#endif
    if (counters_fn == NULL || overlaps_fn == NULL) {
        return 0;
    }
    counters_fn(&counts->handed, &counts->released, &counts->live);
    overlaps_fn(&counts->overlapping);
    return 1;
}

/* ------------------------------------------------------------------------ */
/* Threads                                                                   */
/* ------------------------------------------------------------------------ */

typedef void (*thread_body)(void*);

typedef struct thread_start_block {
    thread_body body;
    void* arg;
} thread_start_block;

#ifdef _WIN32
typedef HANDLE test_thread;

static unsigned __stdcall thread_trampoline(void* raw) {
    thread_start_block block = *(thread_start_block*)raw;
    free(raw);
    block.body(block.arg);
    return 0;
}

static int thread_start(test_thread* thread, thread_body body, void* arg) {
    thread_start_block* block = (thread_start_block*)malloc(sizeof(*block));
    uintptr_t handle;
    if (block == NULL) {
        return 0;
    }
    block->body = body;
    block->arg = arg;
    handle = _beginthreadex(NULL, 0, thread_trampoline, block, 0, NULL);
    if (handle == 0) {
        free(block);
        return 0;
    }
    *thread = (HANDLE)handle;
    return 1;
}

static void thread_join(test_thread thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}
#else
typedef pthread_t test_thread;

static void* thread_trampoline(void* raw) {
    thread_start_block block = *(thread_start_block*)raw;
    free(raw);
    block.body(block.arg);
    return NULL;
}

static int thread_start(test_thread* thread, thread_body body, void* arg) {
    thread_start_block* block = (thread_start_block*)malloc(sizeof(*block));
    if (block == NULL) {
        return 0;
    }
    block->body = body;
    block->arg = arg;
    if (pthread_create(thread, NULL, thread_trampoline, block) != 0) {
        free(block);
        return 0;
    }
    return 1;
}

static void thread_join(test_thread thread) { pthread_join(thread, NULL); }
#endif

/* ------------------------------------------------------------------------ */
/* stderr capture (for "no callback means unchanged behaviour")              */
/* ------------------------------------------------------------------------ */

typedef struct stderr_capture {
    FILE* file;
    int saved_fd;
} stderr_capture;

static int capture_stderr_begin(stderr_capture* capture) {
    fflush(stderr);
#ifdef _WIN32
    {
        char path[MAX_PATH];
        char directory[MAX_PATH];
        DWORD length = GetTempPathA(MAX_PATH, directory);
        if (length == 0 || length > MAX_PATH) {
            return 0;
        }
        snprintf(path, sizeof(path), "%sneuriplo_capi_%lu.log", directory, (unsigned long)GetCurrentProcessId());
        capture->file = fopen(path, "w+b");
    }
    if (capture->file == NULL) {
        return 0;
    }
    capture->saved_fd = _dup(_fileno(stderr));
    if (capture->saved_fd < 0 || _dup2(_fileno(capture->file), _fileno(stderr)) != 0) {
        return 0;
    }
#else
    capture->file = tmpfile();
    if (capture->file == NULL) {
        return 0;
    }
    capture->saved_fd = dup(fileno(stderr));
    if (capture->saved_fd < 0 || dup2(fileno(capture->file), fileno(stderr)) < 0) {
        return 0;
    }
#endif
    return 1;
}

/* Restores stderr and returns what was written meanwhile (caller frees). */
static char* capture_stderr_end(stderr_capture* capture) {
    long size;
    char* text;
    fflush(stderr);
#ifdef _WIN32
    _dup2(capture->saved_fd, _fileno(stderr));
    _close(capture->saved_fd);
#else
    dup2(capture->saved_fd, fileno(stderr));
    close(capture->saved_fd);
#endif
    fseek(capture->file, 0, SEEK_END);
    size = ftell(capture->file);
    if (size < 0) {
        size = 0;
    }
    fseek(capture->file, 0, SEEK_SET);
    text = (char*)calloc((size_t)size + 1, 1);
    if (text != NULL && size > 0) {
        size_t got = fread(text, 1, (size_t)size, capture->file);
        text[got] = '\0';
    }
    fclose(capture->file);
    return text;
}

/* ------------------------------------------------------------------------ */
/* Log capture                                                               */
/* ------------------------------------------------------------------------ */

#define LOG_CAPACITY 256
#define LOG_MESSAGE_SIZE 512

typedef struct log_record {
    neuriplo_log_level_t level;
    char message[LOG_MESSAGE_SIZE];
} log_record;

typedef struct log_sink {
    size_t count; /* every delivery, including ones past capacity */
    int bad_user_data;
    int trailing_newline;
    log_record records[LOG_CAPACITY];
} log_sink;

static log_sink g_log;

static void NEURIPLO_CALL on_log(neuriplo_log_level_t level, const char* message, void* user_data) {
    size_t length;
    if (user_data != &g_log) {
        g_log.bad_user_data = 1;
    }
    if (message != NULL) {
        length = strlen(message);
        if (length > 0 && message[length - 1] == '\n') {
            g_log.trailing_newline = 1;
        }
    }
    if (g_log.count < LOG_CAPACITY) {
        log_record* record = &g_log.records[g_log.count];
        record->level = level;
        snprintf(record->message, sizeof(record->message), "%s", message != NULL ? message : "(null)");
    }
    ++g_log.count;
}

/* 1 if a delivered record at `level` or above contains `needle`. */
static int log_has(neuriplo_log_level_t min_level, const char* needle) {
    size_t i;
    size_t stored = g_log.count < LOG_CAPACITY ? g_log.count : LOG_CAPACITY;
    for (i = 0; i < stored; ++i) {
        if ((int)g_log.records[i].level >= (int)min_level && contains(g_log.records[i].message, needle)) {
            return 1;
        }
    }
    return 0;
}

/* ======================================================================== */
/* Lifecycle [V-1] [V-2]                                                     */
/* ======================================================================== */

static void case_Lifecycle_ApiVersion(void) {
    CHECK(NEURIPLO_C_API_VERSION == 1u);
    CHECK(neuriplo_api_version() == NEURIPLO_C_API_VERSION);
}

static void case_Lifecycle_StatusStrings(void) {
    static const neuriplo_status_t kStatuses[] = {
        NEURIPLO_STATUS_OK,         NEURIPLO_STATUS_INVALID_ARGUMENT, NEURIPLO_STATUS_BACKEND_NOT_FOUND,
        NEURIPLO_STATUS_MODEL_LOAD, NEURIPLO_STATUS_INFERENCE,        NEURIPLO_STATUS_OUT_OF_MEMORY,
        NEURIPLO_STATUS_INTERNAL,   NEURIPLO_STATUS_UNIMPLEMENTED,
    };
    const size_t n = sizeof(kStatuses) / sizeof(kStatuses[0]);
    size_t i;
    size_t j;
    const char* unknown;
    neuriplo_engine_t* engine = NULL;

    for (i = 0; i < n; ++i) {
        const char* name = neuriplo_status_string(kStatuses[i]);
        REQUIRE(name != NULL);
        CHECK(name[0] != '\0');
        for (j = 0; j < i; ++j) {
            CHECK(strcmp(name, neuriplo_status_string(kStatuses[j])) != 0);
        }
    }
    unknown = neuriplo_status_string((neuriplo_status_t)1000);
    REQUIRE(unknown != NULL);
    CHECK(unknown[0] != '\0');

    /* Infallible functions never touch the last error. */
    CHECK_STATUS(neuriplo_engine_create(NULL, &engine), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(last_error_nonempty());
    (void)neuriplo_status_string(NEURIPLO_STATUS_OK);
    (void)neuriplo_api_version();
    neuriplo_engine_destroy(NULL);
    CHECK(last_error_nonempty());
}

static void case_Lifecycle_CreateGoodFixture(void) {
    neuriplo_engine_t* engine = (neuriplo_engine_t*)SENTINEL_PTR;
    const char* id = NULL;
    fixture_counts counts;

    REQUIRE_STATUS(create_fixture(GOOD_ID, "ok", &engine), NEURIPLO_STATUS_OK);
    REQUIRE(engine != NULL && engine != (neuriplo_engine_t*)SENTINEL_PTR);
    CHECK(last_error_empty());

    CHECK_STATUS(neuriplo_engine_backend_id(engine, &id), NEURIPLO_STATUS_OK);
    CHECK(id != NULL && strcmp(id, GOOD_ID) == 0);

    REQUIRE(read_fixture_counts("good", &counts));
    CHECK(counts.live == 1);

    neuriplo_engine_destroy(engine);
    REQUIRE(read_fixture_counts("good", &counts));
    CHECK(counts.live == 0);
}

static void case_Lifecycle_ConformingConfigVariants(void) {
    /* Edges that are valid and must not be over-rejected. */
    static const int64_t kDims[2] = {1, ELEMENTS};
    neuriplo_dims_t input_size;
    neuriplo_engine_config_t config;
    neuriplo_engine_t* engine = NULL;
    fixture_counts counts;

    /* batch_size 0 (= default), use_gpu set, explicit input_sizes. */
    config_init(&config, GOOD_ID, "ok", FIXTURE_DIR);
    config.batch_size = 0;
    config.use_gpu = 1;
    input_size.dims = kDims;
    input_size.ndim = 2;
    config.input_sizes = &input_size;
    config.n_input_sizes = 1;
    CHECK_STATUS(neuriplo_engine_create(&config, &engine), NEURIPLO_STATUS_OK);
    neuriplo_engine_destroy(engine);
    engine = NULL;

    /* An empty model path is passed through; the fixture accepts it. */
    config_init(&config, GOOD_ID, "", FIXTURE_DIR);
    CHECK_STATUS(neuriplo_engine_create(&config, &engine), NEURIPLO_STATUS_OK);
    neuriplo_engine_destroy(engine);
    engine = NULL;

    /* A zero-rank input size may carry NULL dims. */
    config_init(&config, GOOD_ID, "ok", FIXTURE_DIR);
    input_size.dims = NULL;
    input_size.ndim = 0;
    config.input_sizes = &input_size;
    config.n_input_sizes = 1;
    CHECK_STATUS(neuriplo_engine_create(&config, &engine), NEURIPLO_STATUS_OK);
    neuriplo_engine_destroy(engine);

    REQUIRE(read_fixture_counts("good", &counts));
    CHECK(counts.live == 0);
}

static void case_Lifecycle_DefaultBackendBadModel(void) {
    /* The compiled-in default backend needs a real model, which this suite
     * does not have: it is exercised on its failure path, which still proves
     * the default is resolved and reached (MODEL_LOAD, not BACKEND_NOT_FOUND). */
    static const char* const kModel = "/nonexistent/neuriplo-capi-test/model.onnx";
    neuriplo_engine_config_t config;
    neuriplo_engine_t* engine = (neuriplo_engine_t*)SENTINEL_PTR;

    config_init(&config, NULL, kModel, NULL);
    CHECK_STATUS(neuriplo_engine_create(&config, &engine), NEURIPLO_STATUS_MODEL_LOAD);
    CHECK(engine == NULL);
    CHECK(contains(neuriplo_last_error(), kModel));
    CHECK(contains(neuriplo_last_error(), NEURIPLO_DEFAULT_BACKEND));

    engine = (neuriplo_engine_t*)SENTINEL_PTR;
    config_init(&config, "", kModel, "");
    CHECK_STATUS(neuriplo_engine_create(&config, &engine), NEURIPLO_STATUS_MODEL_LOAD);
    CHECK(engine == NULL);

    engine = (neuriplo_engine_t*)SENTINEL_PTR;
    config_init(&config, NEURIPLO_DEFAULT_BACKEND, kModel, NULL);
    CHECK_STATUS(neuriplo_engine_create(&config, &engine), NEURIPLO_STATUS_MODEL_LOAD);
    CHECK(engine == NULL);
    CHECK(contains(neuriplo_last_error(), NEURIPLO_DEFAULT_BACKEND));
    CHECK(contains(neuriplo_last_error(), kModel));
}

static void case_Lifecycle_UnknownBackend(void) {
    neuriplo_engine_t* engine = (neuriplo_engine_t*)SENTINEL_PTR;

    CHECK_STATUS(create_fixture("NO_SUCH_BACKEND", "ok", &engine), NEURIPLO_STATUS_BACKEND_NOT_FOUND);
    CHECK(engine == NULL);
    CHECK(contains(neuriplo_last_error(), "NO_SUCH_BACKEND"));
    CHECK(contains(neuriplo_last_error(), GOOD_ID));
    CHECK(contains(neuriplo_last_error(), SCRIPTED_ID));
    CHECK(contains(neuriplo_last_error(), NEURIPLO_DEFAULT_BACKEND));

    /* A plugin the host rejected is not available either. */
    engine = (neuriplo_engine_t*)SENTINEL_PTR;
    CHECK_STATUS(create_fixture("FIXTURE_WRONG_ABI", "ok", &engine), NEURIPLO_STATUS_BACKEND_NOT_FOUND);
    CHECK(engine == NULL);
    CHECK(contains(neuriplo_last_error(), "FIXTURE_WRONG_ABI"));
}

static int list_contains(const neuriplo_backend_list_t* list, const char* wanted, size_t* occurrences) {
    size_t count = 0;
    size_t i;
    *occurrences = 0;
    if (neuriplo_backend_list_count(list, &count) != NEURIPLO_STATUS_OK) {
        return 0;
    }
    for (i = 0; i < count; ++i) {
        const char* id = NULL;
        if (neuriplo_backend_list_get(list, i, &id) != NEURIPLO_STATUS_OK || id == NULL) {
            return 0;
        }
        if (strcmp(id, wanted) == 0) {
            ++*occurrences;
        }
    }
    return *occurrences > 0;
}

static void case_Lifecycle_AvailableBackends(void) {
    static const char* const kRejected[] = {"FIXTURE_WRONG_ABI", "FIXTURE_NULL_ENTRY", "FIXTURE_INCOMPLETE",
                                            "FIXTURE_MISSING_ENTRY"};
    neuriplo_backend_list_t* list = (neuriplo_backend_list_t*)SENTINEL_PTR;
    size_t count = 0;
    size_t occurrences = 0;
    size_t i;
    const char* id = (const char*)SENTINEL_PTR;

    REQUIRE_STATUS(neuriplo_available_backends(FIXTURE_DIR, &list), NEURIPLO_STATUS_OK);
    REQUIRE(list != NULL && list != (neuriplo_backend_list_t*)SENTINEL_PTR);
    CHECK(last_error_empty());

    REQUIRE_STATUS(neuriplo_backend_list_count(list, &count), NEURIPLO_STATUS_OK);
    CHECK(count >= 3);

    /* Compiled-in backends come first. */
    CHECK_STATUS(neuriplo_backend_list_get(list, 0, &id), NEURIPLO_STATUS_OK);
    CHECK(id != NULL && strcmp(id, NEURIPLO_DEFAULT_BACKEND) == 0);

    CHECK(list_contains(list, NEURIPLO_DEFAULT_BACKEND, &occurrences) && occurrences == 1);
    CHECK(list_contains(list, GOOD_ID, &occurrences) && occurrences == 1);
    CHECK(list_contains(list, SCRIPTED_ID, &occurrences) && occurrences == 1);
    for (i = 0; i < sizeof(kRejected) / sizeof(kRejected[0]); ++i) {
        CHECK(!list_contains(list, kRejected[i], &occurrences));
    }

    /* Index and argument checks. */
    id = (const char*)SENTINEL_PTR;
    CHECK_STATUS(neuriplo_backend_list_get(list, count, &id), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(id == NULL);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_backend_list_get"));
    CHECK_STATUS(neuriplo_backend_list_get(list, 0, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK_STATUS(neuriplo_backend_list_get(NULL, 0, &id), NEURIPLO_STATUS_INVALID_ARGUMENT);
    count = 99;
    CHECK_STATUS(neuriplo_backend_list_count(NULL, &count), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(count == 0);
    CHECK_STATUS(neuriplo_backend_list_count(list, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK_STATUS(neuriplo_available_backends(FIXTURE_DIR, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    neuriplo_backend_list_release(list);
    neuriplo_backend_list_release(NULL);

    /* A second plugin with an id already provided is rejected, so the id is
     * still listed once. */
    list = NULL;
    REQUIRE_STATUS(neuriplo_available_backends(NEURIPLO_FIXTURE_DUPLICATE_DIR, &list), NEURIPLO_STATUS_OK);
    CHECK(list_contains(list, GOOD_ID, &occurrences) && occurrences == 1);
    neuriplo_backend_list_release(list);

    /* No plugin directory: compiled-in backends only (plus what is already
     * loaded in this process). */
    list = NULL;
    REQUIRE_STATUS(neuriplo_available_backends(NULL, &list), NEURIPLO_STATUS_OK);
    CHECK(list_contains(list, NEURIPLO_DEFAULT_BACKEND, &occurrences));
    neuriplo_backend_list_release(list);
}

/* ======================================================================== */
/* StructSize [V-3]                                                          */
/* ======================================================================== */

/* A caller built against a hypothetical newer header: v1 config plus fields. */
typedef struct future_config {
    neuriplo_engine_config_t v1;
    uint64_t appended[4];
} future_config;

static void case_StructSize_TooSmall(void) {
    neuriplo_engine_config_t config;
    neuriplo_engine_t* engine = (neuriplo_engine_t*)SENTINEL_PTR;
    static const uint32_t kSizes[] = {0u, 4u, (uint32_t)offsetof(neuriplo_engine_config_t, plugin_dir),
                                      (uint32_t)sizeof(neuriplo_engine_config_t) - 1u};
    size_t i;

    for (i = 0; i < sizeof(kSizes) / sizeof(kSizes[0]); ++i) {
        config_init(&config, GOOD_ID, "ok", FIXTURE_DIR);
        config.struct_size = kSizes[i];
        engine = (neuriplo_engine_t*)SENTINEL_PTR;
        CHECK_STATUS(neuriplo_engine_create(&config, &engine), NEURIPLO_STATUS_INVALID_ARGUMENT);
        CHECK(engine == NULL);
        CHECK(starts_with(neuriplo_last_error(), "neuriplo_engine_create"));
    }
}

static void case_StructSize_LargerZeroedTrailing(void) {
    future_config config;
    neuriplo_engine_t* engine = NULL;
    const char* id = NULL;

    memset(&config, 0, sizeof(config));
    config_init(&config.v1, GOOD_ID, "ok", FIXTURE_DIR);
    config.v1.struct_size = (uint32_t)sizeof(config);
    REQUIRE_STATUS(neuriplo_engine_create(&config.v1, &engine), NEURIPLO_STATUS_OK);
    CHECK_STATUS(neuriplo_engine_backend_id(engine, &id), NEURIPLO_STATUS_OK);
    CHECK(id != NULL && strcmp(id, GOOD_ID) == 0);
    neuriplo_engine_destroy(engine);
}

static void case_StructSize_NonZeroTrailing(void) {
    future_config config;
    neuriplo_engine_t* engine = (neuriplo_engine_t*)SENTINEL_PTR;

    memset(&config, 0, sizeof(config));
    config_init(&config.v1, GOOD_ID, "ok", FIXTURE_DIR);
    config.v1.struct_size = (uint32_t)sizeof(config);
    config.appended[2] = 1u; /* a field this library cannot honour */
    CHECK_STATUS(neuriplo_engine_create(&config.v1, &engine), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(engine == NULL);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_engine_create"));
}

/* ======================================================================== */
/* Error [V-6], create side                                                  */
/* ======================================================================== */

static void case_Error_CreateFailure(void) {
    neuriplo_engine_t* engine = (neuriplo_engine_t*)SENTINEL_PTR;
    fixture_counts counts;

    CHECK_STATUS(create_fixture(SCRIPTED_ID, "create_fail", &engine), NEURIPLO_STATUS_MODEL_LOAD);
    CHECK(engine == NULL);
    CHECK(contains(neuriplo_last_error(), SCRIPTED_ID));
    CHECK(contains(neuriplo_last_error(), "create_fail"));
    REQUIRE(read_fixture_counts("scripted", &counts));
    CHECK(counts.live == 0);
}

static void case_Error_MetadataRejected(void) {
    static const char* const kModes[] = {"metadata_fail",     "meta_null_inputs", "meta_null_name",
                                         "meta_null_shape",   "meta_huge_ndim",   "meta_bad_output_shape",
                                         "meta_unknown_dtype"};
    size_t i;
    fixture_counts counts;

    for (i = 0; i < sizeof(kModes) / sizeof(kModes[0]); ++i) {
        neuriplo_engine_t* engine = (neuriplo_engine_t*)SENTINEL_PTR;
        CHECK_STATUS(create_fixture(SCRIPTED_ID, kModes[i], &engine), NEURIPLO_STATUS_MODEL_LOAD);
        CHECK(engine == NULL);
        CHECK(contains(neuriplo_last_error(), kModes[i]));
    }
    /* Every instance the host rejected was destroyed. */
    REQUIRE(read_fixture_counts("scripted", &counts));
    CHECK(counts.live == 0);
}

static void case_Error_NullArguments(void) {
    neuriplo_engine_config_t config;
    neuriplo_engine_t* engine = (neuriplo_engine_t*)SENTINEL_PTR;
    neuriplo_dims_t bad_size;
    const char* id = (const char*)SENTINEL_PTR;

    CHECK_STATUS(neuriplo_engine_create(NULL, &engine), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(engine == NULL);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_engine_create"));

    config_init(&config, GOOD_ID, "ok", FIXTURE_DIR);
    CHECK_STATUS(neuriplo_engine_create(&config, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(last_error_nonempty());

    engine = (neuriplo_engine_t*)SENTINEL_PTR;
    config_init(&config, GOOD_ID, NULL, FIXTURE_DIR);
    CHECK_STATUS(neuriplo_engine_create(&config, &engine), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(engine == NULL);

    engine = (neuriplo_engine_t*)SENTINEL_PTR;
    config_init(&config, GOOD_ID, "ok", FIXTURE_DIR);
    config.input_sizes = NULL;
    config.n_input_sizes = 1;
    CHECK_STATUS(neuriplo_engine_create(&config, &engine), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(engine == NULL);

    engine = (neuriplo_engine_t*)SENTINEL_PTR;
    config_init(&config, GOOD_ID, "ok", FIXTURE_DIR);
    bad_size.dims = NULL;
    bad_size.ndim = 2;
    config.input_sizes = &bad_size;
    config.n_input_sizes = 1;
    CHECK_STATUS(neuriplo_engine_create(&config, &engine), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(engine == NULL);

    CHECK_STATUS(neuriplo_engine_backend_id(NULL, &id), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(id == NULL);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_engine_backend_id"));

    /* Infallible releases accept NULL. */
    neuriplo_engine_destroy(NULL);
    neuriplo_backend_list_release(NULL);
    neuriplo_result_release(NULL);
}

static void case_Error_SuccessClearsLastError(void) {
    neuriplo_backend_list_t* list = NULL;
    neuriplo_engine_t* engine = NULL;

    CHECK(last_error_empty()); /* nothing has failed on this thread yet */
    CHECK_STATUS(create_fixture("NO_SUCH_BACKEND", "ok", &engine), NEURIPLO_STATUS_BACKEND_NOT_FOUND);
    CHECK(last_error_nonempty());
    REQUIRE_STATUS(neuriplo_available_backends(FIXTURE_DIR, &list), NEURIPLO_STATUS_OK);
    CHECK(last_error_empty());
    neuriplo_backend_list_release(list);
}

/* ======================================================================== */
/* LastError [V-7]                                                           */
/* ======================================================================== */

#define LAST_ERROR_THREADS 4
#define LAST_ERROR_ITERATIONS 50

typedef struct last_error_worker {
    char backend_id[64];
    int fresh_thread_empty;
    int wrong_status;
    int foreign_message;
} last_error_worker;

static void last_error_body(void* raw) {
    last_error_worker* worker = (last_error_worker*)raw;
    int i;
    worker->fresh_thread_empty = last_error_empty();
    for (i = 0; i < LAST_ERROR_ITERATIONS; ++i) {
        neuriplo_engine_t* engine = NULL;
        if (create_fixture(worker->backend_id, "ok", &engine) != NEURIPLO_STATUS_BACKEND_NOT_FOUND) {
            ++worker->wrong_status;
        }
        if (!contains(neuriplo_last_error(), worker->backend_id)) {
            ++worker->foreign_message;
        }
    }
}

static void case_LastError_PerThread(void) {
    last_error_worker workers[LAST_ERROR_THREADS];
    test_thread threads[LAST_ERROR_THREADS];
    neuriplo_engine_t* engine = NULL;
    char main_message[1024];
    int started[LAST_ERROR_THREADS];
    int i;

    CHECK_STATUS(create_fixture("MAIN_THREAD_MISSING", "ok", &engine), NEURIPLO_STATUS_BACKEND_NOT_FOUND);
    REQUIRE(contains(neuriplo_last_error(), "MAIN_THREAD_MISSING"));
    snprintf(main_message, sizeof(main_message), "%s", neuriplo_last_error());

    for (i = 0; i < LAST_ERROR_THREADS; ++i) {
        memset(&workers[i], 0, sizeof(workers[i]));
        snprintf(workers[i].backend_id, sizeof(workers[i].backend_id), "WORKER_%d_MISSING", i);
        started[i] = thread_start(&threads[i], last_error_body, &workers[i]);
        CHECK(started[i]);
    }
    for (i = 0; i < LAST_ERROR_THREADS; ++i) {
        if (started[i]) {
            thread_join(threads[i]);
            CHECK(workers[i].fresh_thread_empty);
            CHECK(workers[i].wrong_status == 0);
            CHECK(workers[i].foreign_message == 0);
        }
    }

    /* The workers' failures did not touch this thread's message. */
    CHECK(strcmp(neuriplo_last_error(), main_message) == 0);
    CHECK(!contains(neuriplo_last_error(), "WORKER_"));
}

/* ======================================================================== */
/* Metadata [V-4]                                                            */
/* ======================================================================== */

static void check_fixture_layer(const neuriplo_tensor_info_t* info, const char* name) {
    REQUIRE(info != NULL);
    CHECK(info->struct_size == sizeof(neuriplo_tensor_info_t));
    CHECK(info->name != NULL && strcmp(info->name, name) == 0);
    CHECK(info->dtype == NEURIPLO_TENSOR_DTYPE_FLOAT32);
    REQUIRE(info->ndim == 2 && info->shape != NULL);
    CHECK(info->shape[0] == 1 && info->shape[1] == ELEMENTS);
    CHECK(info->batch_size == 1);
}

static void case_Metadata_Fixture(void) {
    neuriplo_engine_t* engine = NULL;
    const neuriplo_tensor_info_t* info = NULL;
    size_t count = 0;

    REQUIRE_STATUS(create_fixture(GOOD_ID, "ok", &engine), NEURIPLO_STATUS_OK);
    CHECK_STATUS(neuriplo_engine_input_count(engine, &count), NEURIPLO_STATUS_OK);
    CHECK(count == 1);
    CHECK_STATUS(neuriplo_engine_output_count(engine, &count), NEURIPLO_STATUS_OK);
    CHECK(count == 1);
    CHECK_STATUS(neuriplo_engine_input(engine, 0, &info), NEURIPLO_STATUS_OK);
    check_fixture_layer(info, "input");
    info = NULL;
    CHECK_STATUS(neuriplo_engine_output(engine, 0, &info), NEURIPLO_STATUS_OK);
    check_fixture_layer(info, "output");
    CHECK(last_error_empty());
    neuriplo_engine_destroy(engine);
}

static void case_Metadata_ViewsStableAcrossInfer(void) {
    neuriplo_engine_t* engine = NULL;
    const neuriplo_tensor_info_t* input_before = NULL;
    const neuriplo_tensor_info_t* output_before = NULL;
    const neuriplo_tensor_info_t* input_after = NULL;
    const neuriplo_tensor_info_t* output_after = NULL;
    const char* id_before = NULL;
    const char* id_after = NULL;
    float in[ELEMENTS];
    int i;

    REQUIRE_STATUS(create_fixture(GOOD_ID, "ok", &engine), NEURIPLO_STATUS_OK);
    REQUIRE_STATUS(neuriplo_engine_input(engine, 0, &input_before), NEURIPLO_STATUS_OK);
    REQUIRE_STATUS(neuriplo_engine_output(engine, 0, &output_before), NEURIPLO_STATUS_OK);
    REQUIRE_STATUS(neuriplo_engine_backend_id(engine, &id_before), NEURIPLO_STATUS_OK);

    for (i = 0; i < 3; ++i) {
        neuriplo_result_t* result = NULL;
        fill_input(in, (float)i);
        CHECK_STATUS(infer_values(engine, in, &result), NEURIPLO_STATUS_OK);
        neuriplo_result_release(result);
    }

    /* The views taken before inference are still the engine's, unchanged. */
    check_fixture_layer(input_before, "input");
    check_fixture_layer(output_before, "output");
    CHECK(id_before != NULL && strcmp(id_before, GOOD_ID) == 0);

    REQUIRE_STATUS(neuriplo_engine_input(engine, 0, &input_after), NEURIPLO_STATUS_OK);
    REQUIRE_STATUS(neuriplo_engine_output(engine, 0, &output_after), NEURIPLO_STATUS_OK);
    REQUIRE_STATUS(neuriplo_engine_backend_id(engine, &id_after), NEURIPLO_STATUS_OK);
    CHECK(input_after == input_before);
    CHECK(output_after == output_before);
    CHECK(id_after == id_before);
    neuriplo_engine_destroy(engine);
}

static void case_Metadata_InvalidArguments(void) {
    neuriplo_engine_t* engine = NULL;
    const neuriplo_tensor_info_t* info = (const neuriplo_tensor_info_t*)SENTINEL_PTR;
    size_t count = 99;

    CHECK_STATUS(neuriplo_engine_input_count(NULL, &count), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(count == 0);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_engine_input_count"));
    count = 99;
    CHECK_STATUS(neuriplo_engine_output_count(NULL, &count), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(count == 0);
    CHECK_STATUS(neuriplo_engine_input(NULL, 0, &info), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(info == NULL);

    REQUIRE_STATUS(create_fixture(GOOD_ID, "ok", &engine), NEURIPLO_STATUS_OK);
    CHECK_STATUS(neuriplo_engine_input_count(engine, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK_STATUS(neuriplo_engine_output_count(engine, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK_STATUS(neuriplo_engine_input(engine, 0, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK_STATUS(neuriplo_engine_output(engine, 0, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    info = (const neuriplo_tensor_info_t*)SENTINEL_PTR;
    CHECK_STATUS(neuriplo_engine_input(engine, 1, &info), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(info == NULL);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_engine_input"));
    info = (const neuriplo_tensor_info_t*)SENTINEL_PTR;
    CHECK_STATUS(neuriplo_engine_output(engine, 1, &info), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(info == NULL);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_engine_output"));
    neuriplo_engine_destroy(engine);
}

/* ======================================================================== */
/* Infer [V-5]                                                               */
/* ======================================================================== */

static void case_Infer_DoublesInput(void) {
    static const float kIn[ELEMENTS] = {1.0f, 2.0f, 3.0f, 4.0f};
    static const float kExpected[ELEMENTS] = {2.0f, 4.0f, 6.0f, 8.0f};
    neuriplo_engine_t* engine = NULL;
    neuriplo_result_t* result = (neuriplo_result_t*)SENTINEL_PTR;
    const neuriplo_tensor_view_t* view = NULL;
    size_t count = 0;
    fixture_counts counts;

    REQUIRE_STATUS(create_fixture(GOOD_ID, "ok", &engine), NEURIPLO_STATUS_OK);
    REQUIRE_STATUS(infer_values(engine, kIn, &result), NEURIPLO_STATUS_OK);
    REQUIRE(result != NULL && result != (neuriplo_result_t*)SENTINEL_PTR);
    CHECK(last_error_empty());

    CHECK_STATUS(neuriplo_result_output_count(result, &count), NEURIPLO_STATUS_OK);
    CHECK(count == 1);
    REQUIRE_STATUS(neuriplo_result_output(result, 0, &view), NEURIPLO_STATUS_OK);
    REQUIRE(view != NULL);
    CHECK(view->struct_size == sizeof(neuriplo_tensor_view_t));
    CHECK(view->dtype == NEURIPLO_TENSOR_DTYPE_FLOAT32);
    CHECK(view->size_bytes == sizeof(kExpected));
    CHECK(view->element_count == ELEMENTS);
    REQUIRE(view->ndim == 2 && view->shape != NULL);
    CHECK(view->shape[0] == 1 && view->shape[1] == ELEMENTS);
    REQUIRE(view->data != NULL);
    CHECK(memcmp(view->data, kExpected, sizeof(kExpected)) == 0);

    neuriplo_result_release(result);
    neuriplo_engine_destroy(engine);
    REQUIRE(read_fixture_counts("good", &counts));
    CHECK(counts.handed == 1 && counts.released == 1);
    CHECK(counts.live == 0);
}

static void case_Infer_EmptyOutput(void) {
    /* Conforming: a zero-element output (e.g. no detections). */
    static const float kIn[ELEMENTS] = {1.0f, 2.0f, 3.0f, 4.0f};
    neuriplo_engine_t* engine = NULL;
    neuriplo_result_t* result = NULL;
    const neuriplo_tensor_view_t* view = NULL;
    size_t count = 0;
    fixture_counts counts;

    REQUIRE_STATUS(create_fixture(SCRIPTED_ID, "out_empty", &engine), NEURIPLO_STATUS_OK);
    REQUIRE_STATUS(infer_values(engine, kIn, &result), NEURIPLO_STATUS_OK);
    CHECK_STATUS(neuriplo_result_output_count(result, &count), NEURIPLO_STATUS_OK);
    CHECK(count == 1);
    REQUIRE_STATUS(neuriplo_result_output(result, 0, &view), NEURIPLO_STATUS_OK);
    REQUIRE(view != NULL);
    CHECK(view->dtype == NEURIPLO_TENSOR_DTYPE_FLOAT32);
    CHECK(view->size_bytes == 0);
    CHECK(view->element_count == 0);
    REQUIRE(view->ndim == 2 && view->shape != NULL);
    CHECK(view->shape[0] == 0 && view->shape[1] == ELEMENTS);
    /* view->data may be NULL or not; it is never read for 0 bytes. */
    neuriplo_result_release(result);
    neuriplo_engine_destroy(engine);
    REQUIRE(read_fixture_counts("scripted", &counts));
    CHECK(counts.handed == counts.released);
}

static void case_Infer_ResultOutlivesEngine(void) {
    static const float kIn[ELEMENTS] = {5.0f, 6.0f, 7.0f, 8.0f};
    neuriplo_engine_t* engine = NULL;
    neuriplo_result_t* result = NULL;
    fixture_counts counts;

    REQUIRE_STATUS(create_fixture(GOOD_ID, "ok", &engine), NEURIPLO_STATUS_OK);
    REQUIRE_STATUS(infer_values(engine, kIn, &result), NEURIPLO_STATUS_OK);
    neuriplo_engine_destroy(engine);
    REQUIRE(read_fixture_counts("good", &counts));
    CHECK(counts.live == 0);

    /* The result owns its outputs: still readable after the engine is gone. */
    CHECK(result_is_doubled(result, kIn));
    neuriplo_result_release(result);
}

static void case_Infer_Repeated1000(void) {
    /* Release-exactly-once over many iterations; run under ASan/LSan for the
     * library side ([V-5]). */
    neuriplo_engine_t* engine = NULL;
    float in[ELEMENTS];
    int failures = 0;
    int i;
    fixture_counts counts;

    REQUIRE_STATUS(create_fixture(GOOD_ID, "ok", &engine), NEURIPLO_STATUS_OK);
    for (i = 0; i < 1000; ++i) {
        neuriplo_result_t* result = NULL;
        fill_input(in, (float)i);
        if (infer_values(engine, in, &result) != NEURIPLO_STATUS_OK || !result_is_doubled(result, in)) {
            ++failures;
        }
        neuriplo_result_release(result);
    }
    CHECK(failures == 0);
    neuriplo_engine_destroy(engine);
    REQUIRE(read_fixture_counts("good", &counts));
    CHECK(counts.handed == 1000);
    CHECK(counts.released == 1000);
    CHECK(counts.live == 0);
}

/* ======================================================================== */
/* InferError [V-6], inference side                                          */
/* ======================================================================== */

static void case_InferError_BackendFailure(void) {
    static const float kIn[ELEMENTS] = {1.0f, 2.0f, 3.0f, 4.0f};
    neuriplo_engine_t* engine = NULL;
    neuriplo_result_t* result = (neuriplo_result_t*)SENTINEL_PTR;

    REQUIRE_STATUS(create_fixture(SCRIPTED_ID, "infer_fail", &engine), NEURIPLO_STATUS_OK);
    CHECK_STATUS(infer_values(engine, kIn, &result), NEURIPLO_STATUS_INFERENCE);
    CHECK(result == NULL);
    CHECK(contains(neuriplo_last_error(), "fixture infer failure"));
    neuriplo_engine_destroy(engine);
}

static void case_InferError_MalformedOutputs(void) {
    static const char* const kModes[] = {"out_null_tensors", "out_null_data", "out_null_shape",   "out_huge_ndim",
                                         "out_size_short",   "out_size_long", "out_negative_dim", "out_unknown_dtype"};
    static const float kIn[ELEMENTS] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t i;
    fixture_counts counts;

    for (i = 0; i < sizeof(kModes) / sizeof(kModes[0]); ++i) {
        neuriplo_engine_t* engine = NULL;
        neuriplo_result_t* result = (neuriplo_result_t*)SENTINEL_PTR;
        CHECK_STATUS(create_fixture(SCRIPTED_ID, kModes[i], &engine), NEURIPLO_STATUS_OK);
        if (engine == NULL) {
            continue;
        }
        CHECK_STATUS(infer_values(engine, kIn, &result), NEURIPLO_STATUS_INFERENCE);
        CHECK(result == NULL);
        CHECK(last_error_nonempty());
        neuriplo_engine_destroy(engine);
    }
    REQUIRE(read_fixture_counts("scripted", &counts));
    CHECK(counts.handed == counts.released);
    CHECK(counts.live == 0);
}

static void case_InferError_WrongInputs(void) {
    /* Whether inputs fit the model is the backend's call: INFERENCE, and the
     * engine stays usable afterwards. */
    static const float kIn[ELEMENTS] = {1.0f, 2.0f, 3.0f, 4.0f};
    neuriplo_engine_t* engine = NULL;
    neuriplo_result_t* result = NULL;
    neuriplo_input_view_t inputs[2];

    REQUIRE_STATUS(create_fixture(GOOD_ID, "ok", &engine), NEURIPLO_STATUS_OK);

    inputs[0].data = kIn;
    inputs[0].size_bytes = 3 * sizeof(float);
    CHECK_STATUS(neuriplo_infer(engine, inputs, 1, &result), NEURIPLO_STATUS_INFERENCE);
    CHECK(result == NULL);

    inputs[0].size_bytes = sizeof(kIn);
    inputs[1] = inputs[0];
    CHECK_STATUS(neuriplo_infer(engine, inputs, 2, &result), NEURIPLO_STATUS_INFERENCE);
    CHECK(result == NULL);

    /* No inputs at all, and an empty input with NULL data, are well-formed
     * arguments -- the backend rejects them, the C layer must not. */
    CHECK_STATUS(neuriplo_infer(engine, NULL, 0, &result), NEURIPLO_STATUS_INFERENCE);
    CHECK(result == NULL);
    inputs[0].data = NULL;
    inputs[0].size_bytes = 0;
    CHECK_STATUS(neuriplo_infer(engine, inputs, 1, &result), NEURIPLO_STATUS_INFERENCE);
    CHECK(result == NULL);

    REQUIRE_STATUS(infer_values(engine, kIn, &result), NEURIPLO_STATUS_OK);
    CHECK(result_is_doubled(result, kIn));
    neuriplo_result_release(result);
    neuriplo_engine_destroy(engine);
}

static void case_InferError_InvalidArguments(void) {
    static const float kIn[ELEMENTS] = {1.0f, 2.0f, 3.0f, 4.0f};
    neuriplo_engine_t* engine = NULL;
    neuriplo_result_t* result = (neuriplo_result_t*)SENTINEL_PTR;
    const neuriplo_tensor_view_t* view = (const neuriplo_tensor_view_t*)SENTINEL_PTR;
    neuriplo_input_view_t input;
    size_t count = 99;
    fixture_counts counts;

    input.data = kIn;
    input.size_bytes = sizeof(kIn);
    CHECK_STATUS(neuriplo_infer(NULL, &input, 1, &result), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(result == NULL);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_infer"));

    REQUIRE_STATUS(create_fixture(GOOD_ID, "ok", &engine), NEURIPLO_STATUS_OK);
    CHECK_STATUS(neuriplo_infer(engine, &input, 1, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    result = (neuriplo_result_t*)SENTINEL_PTR;
    CHECK_STATUS(neuriplo_infer(engine, NULL, 1, &result), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(result == NULL);
    input.data = NULL;
    result = (neuriplo_result_t*)SENTINEL_PTR;
    CHECK_STATUS(neuriplo_infer(engine, &input, 1, &result), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(result == NULL);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_infer"));

    CHECK_STATUS(neuriplo_result_output_count(NULL, &count), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(count == 0);
    CHECK_STATUS(neuriplo_result_output(NULL, 0, &view), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(view == NULL);

    result = NULL;
    REQUIRE_STATUS(infer_values(engine, kIn, &result), NEURIPLO_STATUS_OK);
    CHECK_STATUS(neuriplo_result_output_count(result, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK_STATUS(neuriplo_result_output(result, 0, NULL), NEURIPLO_STATUS_INVALID_ARGUMENT);
    view = (const neuriplo_tensor_view_t*)SENTINEL_PTR;
    CHECK_STATUS(neuriplo_result_output(result, 1, &view), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(view == NULL);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_result_output"));
    neuriplo_result_release(result);
    neuriplo_engine_destroy(engine);

    /* None of the rejected calls reached the plugin; the one that did was
     * released exactly once. */
    REQUIRE(read_fixture_counts("good", &counts));
    CHECK(counts.handed == 1 && counts.released == 1);
}

/* ======================================================================== */
/* Thread [V-8]                                                              */
/* ======================================================================== */

#define THREAD_ENGINES 4
#define THREADS_PER_ENGINE 4
#define THREAD_ITERATIONS 250

typedef struct infer_worker {
    neuriplo_engine_t* engine;
    int seed;
    int iterations;
    int failures;
} infer_worker;

static void infer_worker_body(void* raw) {
    infer_worker* worker = (infer_worker*)raw;
    float in[ELEMENTS];
    int i;
    for (i = 0; i < worker->iterations; ++i) {
        neuriplo_result_t* result = NULL;
        fill_input(in, (float)(worker->seed * 10000 + i));
        if (infer_values(worker->engine, in, &result) != NEURIPLO_STATUS_OK || !result_is_doubled(result, in)) {
            ++worker->failures;
        }
        neuriplo_result_release(result);
    }
}

static void case_Thread_EnginesTimesThreads(void) {
    neuriplo_engine_t* engines[THREAD_ENGINES];
    infer_worker workers[THREAD_ENGINES * THREADS_PER_ENGINE];
    test_thread threads[THREAD_ENGINES * THREADS_PER_ENGINE];
    int started[THREAD_ENGINES * THREADS_PER_ENGINE];
    int e;
    int t;
    fixture_counts counts;

    for (e = 0; e < THREAD_ENGINES; ++e) {
        engines[e] = NULL;
        CHECK_STATUS(create_fixture(GOOD_ID, "ok", &engines[e]), NEURIPLO_STATUS_OK);
    }
    for (e = 0; e < THREAD_ENGINES; ++e) {
        REQUIRE(engines[e] != NULL);
    }
    for (t = 0; t < THREAD_ENGINES * THREADS_PER_ENGINE; ++t) {
        workers[t].engine = engines[t % THREAD_ENGINES];
        workers[t].seed = t;
        workers[t].iterations = THREAD_ITERATIONS;
        workers[t].failures = 0;
        started[t] = thread_start(&threads[t], infer_worker_body, &workers[t]);
        CHECK(started[t]);
    }
    for (t = 0; t < THREAD_ENGINES * THREADS_PER_ENGINE; ++t) {
        if (started[t]) {
            thread_join(threads[t]);
            CHECK(workers[t].failures == 0);
        }
    }
    REQUIRE(read_fixture_counts("good", &counts));
    CHECK(counts.live == THREAD_ENGINES);
    CHECK(counts.handed == (size_t)THREAD_ENGINES * THREADS_PER_ENGINE * THREAD_ITERATIONS);
    CHECK(counts.released == counts.handed);
    for (e = 0; e < THREAD_ENGINES; ++e) {
        neuriplo_engine_destroy(engines[e]);
    }
    REQUIRE(read_fixture_counts("good", &counts));
    CHECK(counts.live == 0);
}

#define SERIAL_THREADS 8
#define SERIAL_ITERATIONS 15

static void case_Thread_SameEngineSerialised(void) {
    /* [D-7]: calls on one engine are serialised. The "slow" fixture mode holds
     * each call open and counts calls that overlap on the same instance. */
    neuriplo_engine_t* engine = NULL;
    infer_worker workers[SERIAL_THREADS];
    test_thread threads[SERIAL_THREADS];
    int started[SERIAL_THREADS];
    int t;
    fixture_counts counts;

    REQUIRE_STATUS(create_fixture(SCRIPTED_ID, "slow", &engine), NEURIPLO_STATUS_OK);
    for (t = 0; t < SERIAL_THREADS; ++t) {
        workers[t].engine = engine;
        workers[t].seed = t;
        workers[t].iterations = SERIAL_ITERATIONS;
        workers[t].failures = 0;
        started[t] = thread_start(&threads[t], infer_worker_body, &workers[t]);
        CHECK(started[t]);
    }
    for (t = 0; t < SERIAL_THREADS; ++t) {
        if (started[t]) {
            thread_join(threads[t]);
            CHECK(workers[t].failures == 0);
        }
    }
    neuriplo_engine_destroy(engine);
    REQUIRE(read_fixture_counts("scripted", &counts));
    CHECK(counts.handed == (size_t)SERIAL_THREADS * SERIAL_ITERATIONS);
    CHECK(counts.overlapping == 0);
}

#define LIFECYCLE_THREADS 8
#define LIFECYCLE_ITERATIONS 20

typedef struct lifecycle_worker {
    int seed;
    int failures;
} lifecycle_worker;

static void lifecycle_worker_body(void* raw) {
    lifecycle_worker* worker = (lifecycle_worker*)raw;
    float in[ELEMENTS];
    int i;
    for (i = 0; i < LIFECYCLE_ITERATIONS; ++i) {
        neuriplo_backend_list_t* list = NULL;
        neuriplo_engine_t* engine = NULL;
        neuriplo_result_t* result = NULL;
        size_t occurrences = 0;
        if (neuriplo_available_backends(FIXTURE_DIR, &list) != NEURIPLO_STATUS_OK ||
            !list_contains(list, GOOD_ID, &occurrences) || occurrences != 1) {
            ++worker->failures;
        }
        neuriplo_backend_list_release(list);
        if (create_fixture(GOOD_ID, "ok", &engine) != NEURIPLO_STATUS_OK) {
            ++worker->failures;
            continue;
        }
        fill_input(in, (float)(worker->seed * 1000 + i));
        if (infer_values(engine, in, &result) != NEURIPLO_STATUS_OK || !result_is_doubled(result, in)) {
            ++worker->failures;
        }
        neuriplo_result_release(result);
        neuriplo_engine_destroy(engine);
    }
}

static void case_Thread_ConcurrentCreateAndList(void) {
    /* Creation, listing, and the first plugin scan race each other. */
    lifecycle_worker workers[LIFECYCLE_THREADS];
    test_thread threads[LIFECYCLE_THREADS];
    int started[LIFECYCLE_THREADS];
    int t;
    fixture_counts counts;

    for (t = 0; t < LIFECYCLE_THREADS; ++t) {
        workers[t].seed = t;
        workers[t].failures = 0;
        started[t] = thread_start(&threads[t], lifecycle_worker_body, &workers[t]);
        CHECK(started[t]);
    }
    for (t = 0; t < LIFECYCLE_THREADS; ++t) {
        if (started[t]) {
            thread_join(threads[t]);
            CHECK(workers[t].failures == 0);
        }
    }
    REQUIRE(read_fixture_counts("good", &counts));
    CHECK(counts.live == 0);
    CHECK(counts.handed == (size_t)LIFECYCLE_THREADS * LIFECYCLE_ITERATIONS);
    CHECK(counts.released == counts.handed);
}

/* ======================================================================== */
/* Log [V-9]                                                                 */
/* ======================================================================== */

static void case_Log_CallbackReceivesRejection(void) {
    neuriplo_backend_list_t* list = NULL;
    stderr_capture capture;
    char* stderr_text;

    memset(&g_log, 0, sizeof(g_log));
    REQUIRE_STATUS(neuriplo_set_log_callback(on_log, NEURIPLO_LOG_LEVEL_INFO, &g_log), NEURIPLO_STATUS_OK);

    REQUIRE(capture_stderr_begin(&capture));
    CHECK_STATUS(neuriplo_available_backends(FIXTURE_DIR, &list), NEURIPLO_STATUS_OK);
    stderr_text = capture_stderr_end(&capture);
    neuriplo_backend_list_release(list);

    /* Loader rejections arrive at WARNING or above, naming the plugin. */
    CHECK(log_has(NEURIPLO_LOG_LEVEL_WARNING, "skipping plugin"));
    CHECK(log_has(NEURIPLO_LOG_LEVEL_WARNING, "fixture_wrong_abi"));
    CHECK(!g_log.bad_user_data);
    CHECK(!g_log.trailing_newline);
    /* The callback is additional: the default output is not silenced. */
    CHECK(contains(stderr_text, "skipping plugin"));
    free(stderr_text);

    CHECK_STATUS(neuriplo_set_log_callback(NULL, NEURIPLO_LOG_LEVEL_INFO, NULL), NEURIPLO_STATUS_OK);
}

static void case_Log_RemovedCallbackSilent(void) {
    neuriplo_backend_list_t* list = NULL;
    neuriplo_engine_t* engine = NULL;
    stderr_capture capture;
    char* stderr_text;
    size_t delivered;

    memset(&g_log, 0, sizeof(g_log));
    REQUIRE_STATUS(neuriplo_set_log_callback(on_log, NEURIPLO_LOG_LEVEL_INFO, &g_log), NEURIPLO_STATUS_OK);
    CHECK_STATUS(neuriplo_available_backends(FIXTURE_DIR, &list), NEURIPLO_STATUS_OK);
    neuriplo_backend_list_release(list);
    list = NULL;
    CHECK(g_log.count > 0);

    REQUIRE_STATUS(neuriplo_set_log_callback(NULL, NEURIPLO_LOG_LEVEL_INFO, NULL), NEURIPLO_STATUS_OK);
    delivered = g_log.count;

    /* Rejected plugins are re-attempted (and re-logged) on every scan, and a
     * plugin create failure logs an error: messages are produced, but none may
     * reach the removed callback. stderr proves they were produced. */
    REQUIRE(capture_stderr_begin(&capture));
    CHECK_STATUS(neuriplo_available_backends(FIXTURE_DIR, &list), NEURIPLO_STATUS_OK);
    CHECK_STATUS(create_fixture(SCRIPTED_ID, "create_fail", &engine), NEURIPLO_STATUS_MODEL_LOAD);
    stderr_text = capture_stderr_end(&capture);
    neuriplo_backend_list_release(list);

    CHECK(contains(stderr_text, "skipping plugin"));
    CHECK(g_log.count == delivered);
    free(stderr_text);
}

static void case_Log_NoCallbackStderrUnchanged(void) {
    neuriplo_backend_list_t* list = NULL;
    stderr_capture capture;
    char* stderr_text;

    REQUIRE(capture_stderr_begin(&capture));
    CHECK_STATUS(neuriplo_available_backends(FIXTURE_DIR, &list), NEURIPLO_STATUS_OK);
    stderr_text = capture_stderr_end(&capture);
    neuriplo_backend_list_release(list);

    CHECK(contains(stderr_text, "skipping plugin"));
    CHECK(contains(stderr_text, "fixture_wrong_abi"));
    free(stderr_text);
}

static void case_Log_MinLevelFilters(void) {
    neuriplo_backend_list_t* list = NULL;
    neuriplo_engine_t* engine = NULL;
    size_t i;

    CHECK_STATUS(neuriplo_set_log_callback(on_log, (neuriplo_log_level_t)7, &g_log), NEURIPLO_STATUS_INVALID_ARGUMENT);
    CHECK(starts_with(neuriplo_last_error(), "neuriplo_set_log_callback"));
    /* Removal ignores the level. */
    CHECK_STATUS(neuriplo_set_log_callback(NULL, (neuriplo_log_level_t)7, NULL), NEURIPLO_STATUS_OK);

    memset(&g_log, 0, sizeof(g_log));
    REQUIRE_STATUS(neuriplo_set_log_callback(on_log, NEURIPLO_LOG_LEVEL_ERROR, &g_log), NEURIPLO_STATUS_OK);
    /* Loader rejections are warnings: filtered out. */
    CHECK_STATUS(neuriplo_available_backends(FIXTURE_DIR, &list), NEURIPLO_STATUS_OK);
    neuriplo_backend_list_release(list);
    /* A plugin create failure is an error: delivered. */
    CHECK_STATUS(create_fixture(SCRIPTED_ID, "create_fail", &engine), NEURIPLO_STATUS_MODEL_LOAD);
    CHECK_STATUS(neuriplo_set_log_callback(NULL, NEURIPLO_LOG_LEVEL_INFO, NULL), NEURIPLO_STATUS_OK);

    CHECK(log_has(NEURIPLO_LOG_LEVEL_ERROR, "fixture create failure"));
    for (i = 0; i < g_log.count && i < LOG_CAPACITY; ++i) {
        CHECK(g_log.records[i].level == NEURIPLO_LOG_LEVEL_ERROR);
    }
}

/* ======================================================================== */
/* Case table -- CMake registers every CAPI_CASE line as ctest CApi.<G>.<N>  */
/* ======================================================================== */

typedef struct capi_case {
    const char* name;
    void (*run)(void);
} capi_case;

#define CAPI_CASE(group, name)                                                                                         \
    { #group "." #name, case_##group##_##name }

static const capi_case kCases[] = {
    CAPI_CASE(Lifecycle, ApiVersion),
    CAPI_CASE(Lifecycle, StatusStrings),
    CAPI_CASE(Lifecycle, CreateGoodFixture),
    CAPI_CASE(Lifecycle, ConformingConfigVariants),
    CAPI_CASE(Lifecycle, DefaultBackendBadModel),
    CAPI_CASE(Lifecycle, UnknownBackend),
    CAPI_CASE(Lifecycle, AvailableBackends),
    CAPI_CASE(StructSize, TooSmall),
    CAPI_CASE(StructSize, LargerZeroedTrailing),
    CAPI_CASE(StructSize, NonZeroTrailing),
    CAPI_CASE(Error, CreateFailure),
    CAPI_CASE(Error, MetadataRejected),
    CAPI_CASE(Error, NullArguments),
    CAPI_CASE(Error, SuccessClearsLastError),
    CAPI_CASE(LastError, PerThread),
    CAPI_CASE(Metadata, Fixture),
    CAPI_CASE(Metadata, ViewsStableAcrossInfer),
    CAPI_CASE(Metadata, InvalidArguments),
    CAPI_CASE(Infer, DoublesInput),
    CAPI_CASE(Infer, EmptyOutput),
    CAPI_CASE(Infer, ResultOutlivesEngine),
    CAPI_CASE(Infer, Repeated1000),
    CAPI_CASE(InferError, BackendFailure),
    CAPI_CASE(InferError, MalformedOutputs),
    CAPI_CASE(InferError, WrongInputs),
    CAPI_CASE(InferError, InvalidArguments),
    CAPI_CASE(Thread, EnginesTimesThreads),
    CAPI_CASE(Thread, SameEngineSerialised),
    CAPI_CASE(Thread, ConcurrentCreateAndList),
    CAPI_CASE(Log, CallbackReceivesRejection),
    CAPI_CASE(Log, RemovedCallbackSilent),
    CAPI_CASE(Log, NoCallbackStderrUnchanged),
    CAPI_CASE(Log, MinLevelFilters),
};

int main(int argc, char** argv) {
    const size_t n = sizeof(kCases) / sizeof(kCases[0]);
    size_t i;

    if (argc == 2 && strcmp(argv[1], "--list") == 0) {
        for (i = 0; i < n; ++i) {
            printf("%s\n", kCases[i].name);
        }
        return 0;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s <Group.Name> | --list\n", argv[0]);
        return 2;
    }
    for (i = 0; i < n; ++i) {
        if (strcmp(argv[1], kCases[i].name) == 0) {
            kCases[i].run();
            if (g_failures != 0) {
                printf("[  FAILED  ] %s (%d failed checks)\n", kCases[i].name, g_failures);
                return 1;
            }
            printf("[       OK ] %s\n", kCases[i].name);
            return 0;
        }
    }
    fprintf(stderr, "unknown case '%s' (see --list)\n", argv[1]);
    return 2;
}
