#include "NativeInfer.hpp"

#include "engine/Device.hpp"
#include "engine/ModelLoader.hpp"
#include "engine/Shapes.hpp"

#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

// The backend boundary owns the global exception types; every engine failure
// crosses it translated so no engine::-typed exception escapes the adapter.
engine::Graph load_graph_or_throw(const std::string& model_path) {
    try {
        return engine::LoadGraphFromFile(model_path);
    } catch (const engine::ModelLoadException& ex) {
        throw ::ModelLoadException(ex.what());
    }
}

// Resolve one concrete, positive shape per declared graph input: a caller size
// wins when supplied, otherwise the model's declared dims are used.
engine::ShapeMap build_input_shapes(const engine::Graph& graph, const std::vector<std::vector<int64_t>>& input_sizes) {
    engine::ShapeMap shapes;
    for (size_t i = 0; i < graph.inputs.size(); ++i) {
        const std::string& name = graph.inputs[i].first;
        std::vector<int64_t> dims =
            (i < input_sizes.size() && !input_sizes[i].empty()) ? input_sizes[i] : graph.inputs[i].second.dims;
        for (int64_t dim : dims) {
            if (dim <= 0) {
                throw ::ModelLoadException("NATIVE input '" + name + "' resolves to a non-positive dimension");
            }
        }
        shapes[name] = std::move(dims);
    }
    return shapes;
}

size_t element_count(const std::vector<int64_t>& dims) {
    size_t count = 1;
    for (int64_t dim : dims) {
        count *= static_cast<size_t>(dim);
    }
    return count;
}

} // namespace

NativeInfer::NativeInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                         const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
    if (use_gpu) {
        throw ::InferenceException("NATIVE runs on CPU only in this phase");
    }

    graph_ = load_graph_or_throw(model_path);

    const engine::ShapeMap shapes = build_input_shapes(graph_, input_sizes);
    try {
        model_ = std::make_unique<engine::Model>(graph_, shapes, engine::CpuDevice());
    } catch (const engine::ModelLoadException& ex) {
        throw ::ModelLoadException(ex.what());
    }

    for (const auto& input : graph_.inputs) {
        const auto resolved = model_->shapes().tensors.find(input.first);
        const std::vector<int64_t>& dims =
            resolved != model_->shapes().tensors.end() ? resolved->second.dims : input.second.dims;
        inference_metadata_.addInput(input.first, dims, batch_size, TensorDataType::Float32);
    }
    for (const auto& output : graph_.outputs) {
        const auto resolved = model_->shapes().tensors.find(output.first);
        const std::vector<int64_t>& dims =
            resolved != model_->shapes().tensors.end() ? resolved->second.dims : output.second.dims;
        inference_metadata_.addOutput(output.first, dims, batch_size, TensorDataType::Float32);
    }

    state_ = BackendState::Ready;
}

engine::InferenceResult NativeInfer::run_engine(const std::vector<std::vector<uint8_t>>& input_tensors) {
    validate_input(input_tensors);

    const auto& inputs = inference_metadata_.getInputs();
    std::map<std::string, std::vector<float>> engine_inputs;
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        const LayerInfo& info = inputs[i];
        const size_t elements = element_count(info.shape);
        const size_t expected_bytes = elements * sizeof(float);
        if (input_tensors[i].size() != expected_bytes) {
            throw ::InferenceExecutionException("NATIVE input '" + info.name + "' expects " +
                                                std::to_string(expected_bytes) + " bytes, got " +
                                                std::to_string(input_tensors[i].size()));
        }

        std::vector<float> data(elements);
        std::memcpy(data.data(), input_tensors[i].data(), expected_bytes);
        engine_inputs[info.name] = std::move(data);
    }

    try {
        return model_->Run(engine_inputs);
    } catch (const engine::InferenceException& ex) {
        throw ::InferenceExecutionException(ex.what());
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
NativeInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    const engine::InferenceResult result = run_engine(input_tensors);

    std::vector<std::vector<TensorElement>> output_tensors;
    std::vector<std::vector<int64_t>> shapes;
    for (const LayerInfo& info : inference_metadata_.getOutputs()) {
        const auto output = result.outputs.find(info.name);
        if (output == result.outputs.end()) {
            throw ::InferenceExecutionException("NATIVE engine produced no output named '" + info.name + "'");
        }

        std::vector<TensorElement> elements;
        elements.reserve(output->second.size());
        for (float value : output->second) {
            elements.emplace_back(value);
        }
        output_tensors.push_back(std::move(elements));
        shapes.push_back(info.shape);
    }

    return std::make_tuple(std::move(output_tensors), std::move(shapes));
}

std::vector<RawOutputTensor>
NativeInfer::get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) {
    const engine::InferenceResult result = run_engine(input_tensors);

    std::vector<RawOutputTensor> outputs;
    outputs.reserve(inference_metadata_.getOutputs().size());
    for (const LayerInfo& info : inference_metadata_.getOutputs()) {
        const auto output = result.outputs.find(info.name);
        if (output == result.outputs.end()) {
            throw ::InferenceExecutionException("NATIVE engine produced no output named '" + info.name + "'");
        }

        RawOutputTensor raw;
        raw.dtype = TensorDtype::FP32;
        raw.shape = info.shape;
        raw.bytes.resize(output->second.size() * sizeof(float));
        std::memcpy(raw.bytes.data(), output->second.data(), raw.bytes.size());
        outputs.push_back(std::move(raw));
    }

    return outputs;
}
