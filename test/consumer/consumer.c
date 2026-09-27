/* Consumer proof, C99 ([T-14]): the only neuriplo header is neuriplo_c.h.
 * Usage: consumer_c <plugin_dir>. Creates FIXTURE_GOOD, infers [1,2,3,4],
 * expects [2,4,6,8]. Prints "OK FIXTURE_GOOD 2 4 6 8" and exits 0 on success. */
#include <neuriplo/neuriplo_c.h>

#include <stdio.h>
#include <string.h>

static int report(const char* what, neuriplo_status_t status) {
    fprintf(stderr, "consumer_c: %s failed: %s (%d): %s\n", what, neuriplo_status_string(status), (int)status,
            neuriplo_last_error());
    return 1;
}

int main(int argc, char** argv) {
    neuriplo_engine_config_t config;
    neuriplo_engine_t* engine = NULL;
    neuriplo_result_t* result = NULL;
    neuriplo_input_view_t input;
    const neuriplo_tensor_view_t* view = NULL;
    const char* backend = NULL;
    const float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float* out;
    size_t count = 0;
    size_t i;
    neuriplo_status_t status;
    int rc = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <plugin_dir>\n", argv[0]);
        return 2;
    }

    memset(&config, 0, sizeof(config));
    config.struct_size = (uint32_t)sizeof(config);
    config.backend_id = "FIXTURE_GOOD";
    config.model_path = "ok";
    config.plugin_dir = argv[1];
    status = neuriplo_engine_create(&config, &engine);
    if (status != NEURIPLO_STATUS_OK) {
        return report("neuriplo_engine_create", status);
    }

    status = neuriplo_engine_backend_id(engine, &backend);
    if (status != NEURIPLO_STATUS_OK) {
        rc = report("neuriplo_engine_backend_id", status);
        goto done;
    }

    input.data = in;
    input.size_bytes = sizeof(in);
    status = neuriplo_infer(engine, &input, 1, &result);
    if (status != NEURIPLO_STATUS_OK) {
        rc = report("neuriplo_infer", status);
        goto done;
    }
    status = neuriplo_result_output_count(result, &count);
    if (status != NEURIPLO_STATUS_OK || count != 1) {
        rc = report("neuriplo_result_output_count", status);
        goto done;
    }
    status = neuriplo_result_output(result, 0, &view);
    if (status != NEURIPLO_STATUS_OK) {
        rc = report("neuriplo_result_output", status);
        goto done;
    }
    if (view->dtype != NEURIPLO_TENSOR_DTYPE_FLOAT32 || view->element_count != 4) {
        fprintf(stderr, "consumer_c: unexpected output dtype %d / element_count %lu\n", (int)view->dtype,
                (unsigned long)view->element_count);
        goto done;
    }
    out = (const float*)view->data;
    for (i = 0; i < 4; ++i) {
        if (out[i] != 2.0f * in[i]) {
            fprintf(stderr, "consumer_c: element %lu is %g, expected %g\n", (unsigned long)i, (double)out[i],
                    (double)(2.0f * in[i]));
            goto done;
        }
    }
    printf("OK %s %g %g %g %g\n", backend, (double)out[0], (double)out[1], (double)out[2], (double)out[3]);
    rc = 0;

done:
    neuriplo_result_release(result);
    neuriplo_engine_destroy(engine);
    return rc;
}
