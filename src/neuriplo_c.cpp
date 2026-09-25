// Consumer C ABI (include/neuriplo/neuriplo_c.h) -- Group 0 stubs.
//
// Every status-returning function returns NEURIPLO_STATUS_UNIMPLEMENTED and
// every infallible one returns an inert value, so the acceptance suite builds
// and fails loudly before implementation (specs/2026-09-25-consumer-c-abi,
// validation [M-4]). Groups 1-3 replace these bodies; the signatures below are
// the contract and must stay exactly as declared (NEURIPLO_CALL and
// NEURIPLO_NOEXCEPT repeated, NEURIPLO_C_API not repeated).

#include "neuriplo/neuriplo_c.h"

uint32_t NEURIPLO_CALL neuriplo_api_version(void) NEURIPLO_NOEXCEPT { return 0; }

const char* NEURIPLO_CALL neuriplo_status_string(neuriplo_status_t /*status*/) NEURIPLO_NOEXCEPT { return ""; }

const char* NEURIPLO_CALL neuriplo_last_error(void) NEURIPLO_NOEXCEPT { return ""; }

neuriplo_status_t NEURIPLO_CALL neuriplo_set_log_callback(neuriplo_log_callback_t /*callback*/,
                                                          neuriplo_log_level_t /*min_level*/,
                                                          void* /*user_data*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_available_backends(const char* /*plugin_dir*/,
                                                            neuriplo_backend_list_t** /*out_list*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_backend_list_count(const neuriplo_backend_list_t* /*list*/,
                                                            size_t* /*out_count*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

neuriplo_status_t NEURIPLO_CALL neuriplo_backend_list_get(const neuriplo_backend_list_t* /*list*/, size_t /*index*/,
                                                          const char** /*out_id*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

void NEURIPLO_CALL neuriplo_backend_list_release(neuriplo_backend_list_t* /*list*/) NEURIPLO_NOEXCEPT {}

neuriplo_status_t NEURIPLO_CALL neuriplo_engine_create(const neuriplo_engine_config_t* /*config*/,
                                                       neuriplo_engine_t** /*out_engine*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
}

void NEURIPLO_CALL neuriplo_engine_destroy(neuriplo_engine_t* /*engine*/) NEURIPLO_NOEXCEPT {}

neuriplo_status_t NEURIPLO_CALL neuriplo_engine_backend_id(const neuriplo_engine_t* /*engine*/,
                                                           const char** /*out_id*/) NEURIPLO_NOEXCEPT {
    return NEURIPLO_STATUS_UNIMPLEMENTED;
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
