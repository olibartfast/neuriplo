#include "NertInfer.hpp"

#include "nert/constant_folding.hpp"
#include "nert/device.hpp"
#include "nert/model_loader.hpp"
#include "nert/shapes.hpp"

#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

// The backend boundary owns the global exception types; every engine failure
// crosses it translated so no nert::-typed exception escapes the adapter.
nert::Graph load_graph_or_throw(const std::string& model_path) {
    try {
        return nert::LoadGraphFromFile(model_path);
    } catch (const nert::ModelLoadException& ex) {
        throw ::ModelLoadException(ex.what());
    }
}

// Resolve one concrete, positive shape per declared graph input: a caller size
// wins when supplied, otherwise the model's declared dims are used.
nert::ShapeMap build_input_shapes(const nert::Graph& graph, const std::vector<std::vector<int64_t>>& input_sizes) {
    nert::ShapeMap shapes;
    for (size_t i = 0; i < graph.inputs.size(); ++i) {
        const std::string& name = graph.inputs[i].first;
        std::vector<int64_t> dims =
            (i < input_sizes.size() && !input_sizes[i].empty()) ? input_sizes[i] : graph.inputs[i].second.dims;
        for (int64_t dim : dims) {
            if (dim == -1) {
                throw ::ModelLoadException("NERT input '" + name +
                                           "' has a symbolic dimension; pass its full shape in input_sizes");
            }
            if (dim <= 0) {
                throw ::ModelLoadException("NERT input '" + name + "' resolves to a non-positive dimension");
            }
        }
        shapes[name] = std::move(dims);
    }
    return shapes;
}

// The backend metadata tag for an engine graph I/O dtype.
TensorDataType to_tensor_data_type(nert::DataType dtype, const std::string& name) {
    switch (dtype) {
    case nert::DataType::Float32:
        return TensorDataType::Float32;
    case nert::DataType::Int64:
        return TensorDataType::Int64;
    case nert::DataType::Bool:
        return TensorDataType::Bool;
    case nert::DataType::Unknown:
        break;
    }
    throw ::ModelLoadException("NERT tensor '" + name + "' has no supported element type");
}

// The resolved dtype of graph tensor `name`, falling back to its declaration.
nert::DataType resolved_dtype(const nert::Model& model, const std::string& name, nert::DataType declared) {
    const auto resolved = model.shapes().tensors.find(name);
    return resolved != model.shapes().tensors.end() ? resolved->second.dtype : declared;
}

size_t element_count(const std::vector<int64_t>& dims) {
    size_t count = 1;
    for (int64_t dim : dims) {
        count *= static_cast<size_t>(dim);
    }
    return count;
}

} // namespace

NertInfer::NertInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                     const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
#ifdef NEURIPLO_NERT_CUDA
    const nert::Device& device = use_gpu ? nert::CudaDevice() : nert::CpuDevice();
#else
    if (use_gpu) {
        throw ::InferenceException("NERT runs on CPU only in this phase");
    }
    const nert::Device& device = nert::CpuDevice();
#endif

    graph_ = load_graph_or_throw(model_path);
    nert::FoldConstants(graph_, device);

    const nert::ShapeMap shapes = build_input_shapes(graph_, input_sizes);
    try {
        model_ = std::make_unique<nert::Model>(graph_, shapes, device);
    } catch (const nert::ModelLoadException& ex) {
        throw ::ModelLoadException(ex.what());
    }

    for (const auto& input : graph_.inputs) {
        const auto resolved = model_->shapes().tensors.find(input.first);
        const std::vector<int64_t>& dims =
            resolved != model_->shapes().tensors.end() ? resolved->second.dims : input.second.dims;
        inference_metadata_.addInput(
            input.first, dims, batch_size,
            to_tensor_data_type(resolved_dtype(*model_, input.first, input.second.dtype), input.first));
    }
    for (const auto& output : graph_.outputs) {
        const auto resolved = model_->shapes().tensors.find(output.first);
        const std::vector<int64_t>& dims =
            resolved != model_->shapes().tensors.end() ? resolved->second.dims : output.second.dims;
        inference_metadata_.addOutput(
            output.first, dims, batch_size,
            to_tensor_data_type(resolved_dtype(*model_, output.first, output.second.dtype), output.first));
    }

    state_ = BackendState::Ready;
}

nert::HostTensorMap NertInfer::run_engine(const std::vector<std::vector<uint8_t>>& input_tensors) {
    validate_input(input_tensors);

    const auto& inputs = inference_metadata_.getInputs();
    nert::HostTensorMap engine_inputs;
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        const LayerInfo& info = inputs[i];
        nert::HostTensor host;
        host.dtype = model_->shapes().tensors.at(info.name).dtype;
        host.dims = info.shape;
        const size_t expected_bytes = element_count(info.shape) * nert::HostElementSize(host.dtype);
        if (input_tensors[i].size() != expected_bytes) {
            throw ::InferenceExecutionException("NERT input '" + info.name + "' expects " +
                                                std::to_string(expected_bytes) + " bytes, got " +
                                                std::to_string(input_tensors[i].size()));
        }
        host.bytes = input_tensors[i];
        engine_inputs.emplace(info.name, std::move(host));
    }

    try {
        return model_->RunTyped(engine_inputs);
    } catch (const nert::InferenceException& ex) {
        throw ::InferenceExecutionException(ex.what());
    }
}

namespace {

const nert::HostTensor& find_output(const nert::HostTensorMap& result, const std::string& name) {
    const auto output = result.find(name);
    if (output == result.end()) {
        throw ::InferenceExecutionException("NERT engine produced no output named '" + name + "'");
    }
    return output->second;
}

// Widen one engine output into TensorElements: float32 as float, int64 as
// int64_t, and bool as uint8_t (0 or 1), per the HostTensorConverter rule.
template <typename T, typename Element>
void append_elements(const nert::HostTensor& host, std::vector<TensorElement>& elements) {
    const size_t count = host.bytes.size() / sizeof(T);
    elements.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        T value;
        std::memcpy(&value, host.bytes.data() + i * sizeof(T), sizeof(T));
        elements.emplace_back(static_cast<Element>(value));
    }
}

TensorDtype to_raw_dtype(nert::DataType dtype) {
    switch (dtype) {
    case nert::DataType::Float32:
        return TensorDtype::FP32;
    case nert::DataType::Int64:
        return TensorDtype::INT64;
    case nert::DataType::Bool:
        return TensorDtype::UINT8;
    case nert::DataType::Unknown:
        break;
    }
    throw ::InferenceExecutionException("NERT output has no supported element type");
}

} // namespace

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
NertInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    const nert::HostTensorMap result = run_engine(input_tensors);

    std::vector<std::vector<TensorElement>> output_tensors;
    std::vector<std::vector<int64_t>> shapes;
    for (const LayerInfo& info : inference_metadata_.getOutputs()) {
        const nert::HostTensor& output = find_output(result, info.name);

        std::vector<TensorElement> elements;
        switch (output.dtype) {
        case nert::DataType::Float32:
            append_elements<float, float>(output, elements);
            break;
        case nert::DataType::Int64:
            append_elements<int64_t, int64_t>(output, elements);
            break;
        case nert::DataType::Bool:
            append_elements<uint8_t, uint8_t>(output, elements);
            break;
        case nert::DataType::Unknown:
            throw ::InferenceExecutionException("NERT output '" + info.name + "' has no supported element type");
        }
        output_tensors.push_back(std::move(elements));
        shapes.push_back(info.shape);
    }

    return std::make_tuple(std::move(output_tensors), std::move(shapes));
}

std::vector<RawOutputTensor> NertInfer::get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) {
    nert::HostTensorMap result = run_engine(input_tensors);

    std::vector<RawOutputTensor> outputs;
    outputs.reserve(inference_metadata_.getOutputs().size());
    for (const LayerInfo& info : inference_metadata_.getOutputs()) {
        const auto output = result.find(info.name);
        if (output == result.end()) {
            throw ::InferenceExecutionException("NERT engine produced no output named '" + info.name + "'");
        }

        RawOutputTensor raw;
        raw.dtype = to_raw_dtype(output->second.dtype);
        raw.shape = info.shape;
        raw.bytes = std::move(output->second.bytes);
        outputs.push_back(std::move(raw));
    }

    return outputs;
}
