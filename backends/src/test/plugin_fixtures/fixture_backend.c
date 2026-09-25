/* First-party, dependency-free plugin fixtures for PluginAbiContractTest.
 *
 * One source, compiled into several libneuriplo_backend_fixture_* modules by
 * backends/src/test/CMakeLists.txt. Two compile-time knobs select what a given
 * module is:
 *
 *   FIXTURE_BACKEND_ID   the backend_id it advertises
 *   FIXTURE_API_DEFECT   a load-time defect in the exported api table:
 *                          0 none, 1 wrong abi_version, 2 entry returns NULL,
 *                          3 incomplete table (infer == NULL),
 *                          4 no entry symbol exported at all
 *
 * Call-time behaviour is chosen per backend instance by the model_path passed
 * to create (see the MODE_* table below), so one loadable module covers every
 * malformed-metadata and malformed-output case without a library per case.
 *
 * The conforming behaviour: one FP32 input "input" [1,4], one FP32 output
 * "output" [1,4], output[i] = 2 * input[i].
 *
 * Owned by the specifier; read-only to every implementer (see
 * specs/2026-09-22-plugin-abi-loader-hardening/orchestration.md). */

#include "neuriplo/plugin_abi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CMake always defines FIXTURE_BACKEND_ID; the fallback only lets static
 * analysers that do not see the build's definitions parse this file. */
#ifndef FIXTURE_BACKEND_ID
#define FIXTURE_BACKEND_ID "FIXTURE_UNCONFIGURED"
#endif
#ifndef FIXTURE_API_DEFECT
#define FIXTURE_API_DEFECT 0
#endif

#ifdef _WIN32
#define FIXTURE_EXPORT __declspec(dllexport)
#else
#define FIXTURE_EXPORT __attribute__((visibility("default")))
#endif

enum fixture_mode {
    MODE_OK = 0,
    MODE_CREATE_FAIL,
    MODE_METADATA_FAIL,
    MODE_INFER_FAIL,
    MODE_META_NULL_INPUTS,
    MODE_META_NULL_NAME,
    MODE_META_NULL_SHAPE,
    MODE_META_HUGE_NDIM,
    MODE_META_BAD_OUTPUT_SHAPE,
    MODE_META_UNKNOWN_DTYPE,
    MODE_OUT_NULL_TENSORS,
    MODE_OUT_NULL_DATA,
    MODE_OUT_NULL_SHAPE,
    MODE_OUT_HUGE_NDIM,
    MODE_OUT_SIZE_SHORT,
    MODE_OUT_SIZE_LONG,
    MODE_OUT_NEGATIVE_DIM,
    MODE_OUT_UNKNOWN_DTYPE,
    MODE_OUT_EMPTY,
    MODE_COUNT
};

static const char* const kModeNames[MODE_COUNT] = {
    "ok",
    "create_fail",
    "metadata_fail",
    "infer_fail",
    "meta_null_inputs",
    "meta_null_name",
    "meta_null_shape",
    "meta_huge_ndim",
    "meta_bad_output_shape",
    "meta_unknown_dtype",
    "out_null_tensors",
    "out_null_data",
    "out_null_shape",
    "out_huge_ndim",
    "out_size_short",
    "out_size_long",
    "out_negative_dim",
    "out_unknown_dtype",
    "out_empty",
};

#define FIXTURE_ELEMENTS 4
#define FIXTURE_HUGE_NDIM ((size_t)1 << 20)

static const int64_t kShape[2] = {1, FIXTURE_ELEMENTS};

/* Counters the test reads back through neuriplo_fixture_counters: how many
 * output arrays infer handed to the host, and how many release_outputs gave
 * back. Every successful infer must be released exactly once. */
static size_t g_outputs_handed = 0;
static size_t g_outputs_released = 0;
static size_t g_live_instances = 0;

struct neuriplo_backend_t {
    enum fixture_mode mode;
    neuriplo_layer_info_t inputs[1];
    neuriplo_layer_info_t outputs[1];
};

/* Everything infer returns lives in one allocation, released as a unit. */
typedef struct fixture_outputs {
    neuriplo_output_tensor_t tensor;
    int64_t shape[2];
    float values[FIXTURE_ELEMENTS + 1]; /* +1 so out_size_long stays in bounds */
} fixture_outputs;

static void write_error(char* error, size_t error_size, const char* message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

static enum fixture_mode parse_mode(const char* model_path) {
    int i;
    if (model_path == NULL) {
        return MODE_OK;
    }
    for (i = 0; i < MODE_COUNT; ++i) {
        if (strcmp(model_path, kModeNames[i]) == 0) {
            return (enum fixture_mode)i;
        }
    }
    return MODE_OK;
}

static neuriplo_backend_t* fixture_create(const neuriplo_engine_options_t* options,
                                          const neuriplo_host_services_t* host, char* error, size_t error_size) {
    neuriplo_backend_t* backend;
    enum fixture_mode mode = parse_mode(options != NULL ? options->model_path : NULL);
    (void)host;
    if (mode == MODE_CREATE_FAIL) {
        write_error(error, error_size, "fixture create failure");
        return NULL;
    }
    backend = (neuriplo_backend_t*)calloc(1, sizeof(*backend));
    if (backend == NULL) {
        write_error(error, error_size, "fixture out of memory");
        return NULL;
    }
    backend->mode = mode;
    backend->inputs[0].name = "input";
    backend->inputs[0].shape = kShape;
    backend->inputs[0].ndim = 2;
    backend->inputs[0].batch_size = 1;
    backend->inputs[0].element_type = NEURIPLO_DTYPE_FP32;
    backend->outputs[0] = backend->inputs[0];
    backend->outputs[0].name = "output";
    ++g_live_instances;
    return backend;
}

static void fixture_destroy(neuriplo_backend_t* backend) {
    if (backend != NULL) {
        --g_live_instances;
        free(backend);
    }
}

static int fixture_get_metadata(neuriplo_backend_t* backend, neuriplo_metadata_t* out_metadata) {
    if (backend->mode == MODE_METADATA_FAIL) {
        return -1;
    }
    out_metadata->inputs = backend->inputs;
    out_metadata->n_inputs = 1;
    out_metadata->outputs = backend->outputs;
    out_metadata->n_outputs = 1;
    switch (backend->mode) {
    case MODE_META_NULL_INPUTS:
        out_metadata->inputs = NULL;
        break;
    case MODE_META_NULL_NAME:
        backend->inputs[0].name = NULL;
        break;
    case MODE_META_NULL_SHAPE:
        backend->inputs[0].shape = NULL;
        break;
    case MODE_META_HUGE_NDIM:
        backend->inputs[0].ndim = FIXTURE_HUGE_NDIM;
        break;
    case MODE_META_BAD_OUTPUT_SHAPE:
        backend->outputs[0].shape = NULL;
        break;
    case MODE_META_UNKNOWN_DTYPE:
        backend->inputs[0].element_type = (neuriplo_dtype_t)99;
        break;
    default:
        break;
    }
    return 0;
}

static int fixture_infer(neuriplo_backend_t* backend, const neuriplo_input_buffer_t* inputs, size_t n_inputs,
                         neuriplo_output_tensor_t** out_tensors, size_t* out_count, char* error, size_t error_size) {
    fixture_outputs* outputs;
    const float* in;
    size_t i;

    if (backend->mode == MODE_INFER_FAIL) {
        write_error(error, error_size, "fixture infer failure");
        return -1;
    }
    if (n_inputs != 1 || inputs == NULL || inputs[0].data == NULL ||
        inputs[0].size_bytes != FIXTURE_ELEMENTS * sizeof(float)) {
        write_error(error, error_size, "fixture expects one FP32 [1,4] input");
        return -1;
    }
    if (backend->mode == MODE_OUT_NULL_TENSORS) {
        /* Claims success and one tensor, hands back nothing. Nothing was
         * allocated, so nothing is counted as handed out. */
        *out_tensors = NULL;
        *out_count = 1;
        return 0;
    }

    outputs = (fixture_outputs*)calloc(1, sizeof(*outputs));
    if (outputs == NULL) {
        write_error(error, error_size, "fixture out of memory");
        return -1;
    }
    in = (const float*)inputs[0].data;
    for (i = 0; i < FIXTURE_ELEMENTS; ++i) {
        outputs->values[i] = 2.0f * in[i];
    }
    outputs->shape[0] = kShape[0];
    outputs->shape[1] = kShape[1];
    outputs->tensor.dtype = NEURIPLO_DTYPE_FP32;
    outputs->tensor.data = outputs->values;
    outputs->tensor.size_bytes = FIXTURE_ELEMENTS * sizeof(float);
    outputs->tensor.shape = outputs->shape;
    outputs->tensor.ndim = 2;

    switch (backend->mode) {
    case MODE_OUT_NULL_DATA:
        outputs->tensor.data = NULL;
        break;
    case MODE_OUT_NULL_SHAPE:
        outputs->tensor.shape = NULL;
        break;
    case MODE_OUT_HUGE_NDIM:
        outputs->tensor.ndim = FIXTURE_HUGE_NDIM;
        break;
    case MODE_OUT_SIZE_SHORT:
        outputs->tensor.size_bytes = (FIXTURE_ELEMENTS - 1) * sizeof(float);
        break;
    case MODE_OUT_SIZE_LONG:
        outputs->tensor.size_bytes = (FIXTURE_ELEMENTS + 1) * sizeof(float);
        break;
    case MODE_OUT_NEGATIVE_DIM:
        outputs->shape[0] = -1;
        break;
    case MODE_OUT_UNKNOWN_DTYPE:
        outputs->tensor.dtype = (neuriplo_dtype_t)42;
        break;
    case MODE_OUT_EMPTY:
        /* Conforming: a zero-element tensor (e.g. no detections) may carry a
         * NULL data pointer, as std::vector<uint8_t>{}.data() does. */
        outputs->shape[0] = 0;
        outputs->tensor.data = NULL;
        outputs->tensor.size_bytes = 0;
        break;
    default:
        break;
    }

    ++g_outputs_handed;
    *out_tensors = &outputs->tensor;
    *out_count = 1;
    return 0;
}

static void fixture_release_outputs(neuriplo_backend_t* backend, neuriplo_output_tensor_t* tensors, size_t count) {
    (void)backend;
    (void)count;
    if (tensors != NULL) {
        ++g_outputs_released;
        /* tensor is the first member, so this is the fixture_outputs block. */
        free(tensors);
    }
}

/* Test-only introspection; not part of the plugin ABI. */
FIXTURE_EXPORT void neuriplo_fixture_counters(size_t* handed, size_t* released, size_t* live_instances) {
    *handed = g_outputs_handed;
    *released = g_outputs_released;
    *live_instances = g_live_instances;
}

static const neuriplo_plugin_api_v1* fixture_api(void) {
#if FIXTURE_API_DEFECT == 3
    (void)fixture_infer; /* left out of the table on purpose */
#endif
    static const neuriplo_plugin_api_v1 api = {
#if FIXTURE_API_DEFECT == 1
        NEURIPLO_PLUGIN_ABI_VERSION + 1u,
#else
        NEURIPLO_PLUGIN_ABI_VERSION,
#endif
        FIXTURE_BACKEND_ID,
        "neuriplo contract fixture " FIXTURE_BACKEND_ID,
        0,
        fixture_create,
        fixture_destroy,
        fixture_get_metadata,
#if FIXTURE_API_DEFECT == 3
        NULL,
#else
        fixture_infer,
#endif
        fixture_release_outputs,
    };
    return &api;
}

#if FIXTURE_API_DEFECT == 4
/* Misspelled on purpose: the module opens, but the host's lookup of
 * NEURIPLO_PLUGIN_ENTRY_SYMBOL must fail. */
FIXTURE_EXPORT const neuriplo_plugin_api_v1* neuriplo_plugin_get_api_v0(void) { return fixture_api(); }
#else
/* plugin_abi.h already declares the entry point without dllexport, and MSVC
 * rejects a definition that adds it (C2375), so export it through the
 * linker instead. */
#ifdef _MSC_VER
#pragma comment(linker, "/EXPORT:neuriplo_plugin_get_api_v1")
#define FIXTURE_ENTRY_EXPORT
#else
#define FIXTURE_ENTRY_EXPORT FIXTURE_EXPORT
#endif
FIXTURE_ENTRY_EXPORT const neuriplo_plugin_api_v1* neuriplo_plugin_get_api_v1(void) {
#if FIXTURE_API_DEFECT == 2
    (void)fixture_api;
    return NULL;
#else
    return fixture_api();
#endif
}
#endif
