#pragma once

// Plugin-side adapter: wraps an existing IBackendRuntimeFactory /
// InferenceInterface pair behind the C plugin ABI so backends need no
// per-backend rewrite to become plugins. Used only inside plugin shared
// libraries via NEURIPLO_DEFINE_PLUGIN; the host never includes this header.

#include "IBackendRuntimeFactory.hpp"
#include "InferenceInterface.hpp"
#include "neuriplo/plugin_abi.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace neuriplo_plugin {

inline void write_error(char* error, size_t error_size, const std::string& message) {
    if (error == nullptr || error_size == 0) {
        return;
    }
    const size_t length = message.size() < error_size - 1 ? message.size() : error_size - 1;
    std::memcpy(error, message.data(), length);
    error[length] = '\0';
}

// Owned storage backing the views handed across the ABI. Views stay valid
// until the next call that repopulates them, matching the ABI contract.
struct BackendHandle {
    std::unique_ptr<InferenceInterface> backend;
    neuriplo_host_services_t host{};

    // get_metadata storage
    std::vector<std::string> layer_names;
    std::vector<std::vector<int64_t>> layer_shapes;
    std::vector<neuriplo_layer_info_t> input_infos;
    std::vector<neuriplo_layer_info_t> output_infos;

    // infer storage
    std::vector<std::vector<uint8_t>> output_bytes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<neuriplo_output_tensor_t> output_tensors;

    void log(neuriplo_log_severity_t severity, const std::string& message) const {
        if (host.log != nullptr) {
            host.log(severity, message.c_str(), host.log_user_data);
        }
    }
};

// TensorDtype values mirror the ABI enum, so the conversion is a cast.
static_assert(static_cast<int>(TensorDtype::FP32) == NEURIPLO_DTYPE_FP32, "dtype enums must mirror the ABI");
static_assert(static_cast<int>(TensorDtype::INT32) == NEURIPLO_DTYPE_INT32, "dtype enums must mirror the ABI");
static_assert(static_cast<int>(TensorDtype::INT64) == NEURIPLO_DTYPE_INT64, "dtype enums must mirror the ABI");
static_assert(static_cast<int>(TensorDtype::UINT8) == NEURIPLO_DTYPE_UINT8, "dtype enums must mirror the ABI");

inline neuriplo_dtype_t to_abi_dtype(TensorDtype dtype) { return static_cast<neuriplo_dtype_t>(dtype); }

template <typename Factory>
neuriplo_backend_t* plugin_create(const neuriplo_engine_options_t* options, const neuriplo_host_services_t* host,
                                  char* error, size_t error_size) {
    if (options == nullptr || options->model_path == nullptr) {
        write_error(error, error_size, "engine options with a model_path are required");
        return nullptr;
    }

    try {
        std::vector<std::vector<int64_t>> input_sizes;
        if (options->input_sizes != nullptr) {
            input_sizes.reserve(options->n_input_sizes);
            for (size_t i = 0; i < options->n_input_sizes; ++i) {
                const neuriplo_shape_t& shape = options->input_sizes[i];
                input_sizes.emplace_back(shape.dims, shape.dims + shape.ndim);
            }
        }

        Factory factory;
        auto backend =
            factory.create_backend(options->model_path, options->use_gpu != 0, options->batch_size, input_sizes);
        if (!backend) {
            write_error(error, error_size, "backend factory returned null");
            return nullptr;
        }

        backend->load();
        if (backend->state() == BackendState::Failed) {
            write_error(error, error_size, std::string("backend failed to load model: ") + options->model_path);
            return nullptr;
        }

        auto handle = std::make_unique<BackendHandle>();
        handle->backend = std::move(backend);
        if (host != nullptr && host->struct_size >= sizeof(neuriplo_host_services_t)) {
            handle->host = *host;
        }
        return reinterpret_cast<neuriplo_backend_t*>(handle.release());
    } catch (const std::exception& e) {
        write_error(error, error_size, e.what());
        return nullptr;
    } catch (...) {
        write_error(error, error_size, "unknown error while creating backend");
        return nullptr;
    }
}

inline void plugin_destroy(neuriplo_backend_t* backend) { delete reinterpret_cast<BackendHandle*>(backend); }

inline int plugin_get_metadata(neuriplo_backend_t* backend, neuriplo_metadata_t* out_metadata) {
    if (backend == nullptr || out_metadata == nullptr) {
        return 1;
    }
    auto* handle = reinterpret_cast<BackendHandle*>(backend);
    try {
        const InferenceMetadata metadata = handle->backend->get_inference_metadata();

        handle->layer_names.clear();
        handle->layer_shapes.clear();
        handle->input_infos.clear();
        handle->output_infos.clear();

        // Reserve so the views built below are never invalidated by growth.
        const size_t total_layers = metadata.getInputs().size() + metadata.getOutputs().size();
        handle->layer_names.reserve(total_layers);
        handle->layer_shapes.reserve(total_layers);

        auto append = [&](const LayerInfo& layer, std::vector<neuriplo_layer_info_t>& infos) {
            handle->layer_names.push_back(layer.name);
            handle->layer_shapes.push_back(layer.shape);
            neuriplo_layer_info_t info{};
            info.name = handle->layer_names.back().c_str();
            info.shape = handle->layer_shapes.back().data();
            info.ndim = handle->layer_shapes.back().size();
            info.batch_size = layer.batch_size;
            infos.push_back(info);
        };
        for (const LayerInfo& layer : metadata.getInputs()) {
            append(layer, handle->input_infos);
        }
        for (const LayerInfo& layer : metadata.getOutputs()) {
            append(layer, handle->output_infos);
        }

        out_metadata->inputs = handle->input_infos.data();
        out_metadata->n_inputs = handle->input_infos.size();
        out_metadata->outputs = handle->output_infos.data();
        out_metadata->n_outputs = handle->output_infos.size();
        return 0;
    } catch (const std::exception& e) {
        handle->log(NEURIPLO_LOG_ERROR, std::string("get_metadata failed: ") + e.what());
        return 1;
    }
}

inline int plugin_infer(neuriplo_backend_t* backend, const neuriplo_input_buffer_t* inputs, size_t n_inputs,
                        neuriplo_output_tensor_t** out_tensors, size_t* out_count, char* error, size_t error_size) {
    if (backend == nullptr || out_tensors == nullptr || out_count == nullptr) {
        write_error(error, error_size, "invalid arguments to infer");
        return 1;
    }
    auto* handle = reinterpret_cast<BackendHandle*>(backend);
    try {
        std::vector<std::vector<uint8_t>> input_tensors;
        input_tensors.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            input_tensors.emplace_back(inputs[i].data, inputs[i].data + inputs[i].size_bytes);
        }

        std::vector<RawOutputTensor> outputs = handle->backend->get_infer_results_raw(input_tensors);

        handle->output_bytes.clear();
        handle->output_shapes.clear();
        handle->output_tensors.clear();
        handle->output_bytes.reserve(outputs.size());
        handle->output_shapes.reserve(outputs.size());
        handle->output_tensors.reserve(outputs.size());

        for (RawOutputTensor& output : outputs) {
            handle->output_bytes.push_back(std::move(output.bytes));
            handle->output_shapes.push_back(std::move(output.shape));

            neuriplo_output_tensor_t tensor{};
            tensor.dtype = to_abi_dtype(output.dtype);
            tensor.data = handle->output_bytes.back().data();
            tensor.size_bytes = handle->output_bytes.back().size();
            tensor.shape = handle->output_shapes.back().data();
            tensor.ndim = handle->output_shapes.back().size();
            handle->output_tensors.push_back(tensor);
        }

        *out_tensors = handle->output_tensors.data();
        *out_count = handle->output_tensors.size();
        return 0;
    } catch (const std::exception& e) {
        write_error(error, error_size, e.what());
        return 1;
    } catch (...) {
        write_error(error, error_size, "unknown error during inference");
        return 1;
    }
}

inline void plugin_release_outputs(neuriplo_backend_t* backend, neuriplo_output_tensor_t* tensors, size_t count) {
    (void)tensors;
    (void)count;
    if (backend == nullptr) {
        return;
    }
    auto* handle = reinterpret_cast<BackendHandle*>(backend);
    handle->output_bytes.clear();
    handle->output_shapes.clear();
    handle->output_tensors.clear();
}

} // namespace neuriplo_plugin

// Generates the exported entry point for a plugin wrapping FactoryClass.
#define NEURIPLO_DEFINE_PLUGIN(FactoryClass, backend_id_str, display_name_str, force_gpu_flag)                         \
    extern "C" __attribute__((visibility("default"))) const neuriplo_plugin_api_v1* neuriplo_plugin_get_api_v1(void) { \
        static const neuriplo_plugin_api_v1 api = {NEURIPLO_PLUGIN_ABI_VERSION,                                        \
                                                   backend_id_str,                                                     \
                                                   display_name_str,                                                   \
                                                   force_gpu_flag,                                                     \
                                                   &neuriplo_plugin::plugin_create<FactoryClass>,                      \
                                                   &neuriplo_plugin::plugin_destroy,                                   \
                                                   &neuriplo_plugin::plugin_get_metadata,                              \
                                                   &neuriplo_plugin::plugin_infer,                                     \
                                                   &neuriplo_plugin::plugin_release_outputs};                          \
        return &api;                                                                                                   \
    }
