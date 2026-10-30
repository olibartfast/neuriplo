#include "MIGraphXInfer.hpp"

#include <cstring>
#include <stdexcept>

namespace {
TensorDataType datatype(migraphx_shape_datatype_t type) {
    switch (type) {
    case migraphx_shape_float_type:
        return TensorDataType::Float32;
    case migraphx_shape_int32_type:
        return TensorDataType::Int32;
    case migraphx_shape_int64_type:
        return TensorDataType::Int64;
    case migraphx_shape_uint8_type:
        return TensorDataType::UInt8;
    case migraphx_shape_int8_type:
        return TensorDataType::Int8;
    case migraphx_shape_bool_type:
        return TensorDataType::Bool;
    default:
        throw ModelLoadException("MIGraphX tensor datatype is unsupported");
    }
}

template <class T, class Stored = T>
void append_values(std::vector<TensorElement>& result, const migraphx::argument& arg) {
    const auto shape = arg.get_shape();
    const auto lens = shape.lengths();
    const auto strides = shape.strides();
    const auto* data = reinterpret_cast<const uint8_t*>(arg.data());
    for (size_t i = 0; i < shape.elements(); ++i) {
        size_t index = i;
        size_t offset = 0;
        for (size_t dim = lens.size(); dim > 0; --dim) {
            offset += (index % lens[dim - 1]) * strides[dim - 1];
            index /= lens[dim - 1];
        }
        T value;
        std::memcpy(&value, data + offset * sizeof(T), sizeof(T));
        result.emplace_back(static_cast<Stored>(value));
    }
}
} // namespace

MIGraphXInfer::MIGraphXInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                             const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}, use_gpu_{use_gpu} {
    migraphx::onnx_options onnx_opts;
    onnx_opts.set_default_dim_value(static_cast<unsigned>(batch_size));
    try {
        program_ = migraphx::parse_onnx(model_path.c_str(), onnx_opts);
        // Capture model inputs before GPU compilation adds internal output parameters.
        const auto model_parameters = program_.get_parameter_shapes();
        for (const auto& name : model_parameters.names())
            input_names_.push_back(name);

        migraphx::compile_options options;
        // The public interface supplies host buffers and returns host values.
        options.set_offload_copy(true);
        LOG(INFO) << "MIGraphX: compiling for " << (use_gpu_ ? "GPU (hip)" : "CPU (ref)");
        program_.compile(migraphx::target(use_gpu_ ? "gpu" : "ref"), options);
        gpu_available_ = use_gpu_;

        const auto parameters = program_.get_parameter_shapes();
        for (const auto& name : input_names_) {
            auto shape = parameters[name.c_str()];
            if (shape.dynamic() || !shape.standard())
                throw ModelLoadException("MIGraphX requires static contiguous input shapes");
            const auto lens = shape.lengths();
            inference_metadata_.addInput(name, {lens.begin(), lens.end()}, batch_size, datatype(shape.type()));
            input_shapes_.push_back(shape);
        }
        const auto outputs = program_.get_output_shapes();
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto shape = outputs[i];
            if (shape.dynamic())
                throw ModelLoadException("MIGraphX dynamic output shapes are unsupported");
            const auto lens = shape.lengths();
            inference_metadata_.addOutput("output_" + std::to_string(i), {lens.begin(), lens.end()}, batch_size,
                                          datatype(shape.type()));
        }
        state_ = BackendState::Ready;
    } catch (const std::exception& e) {
        state_ = BackendState::Failed;
        throw ModelLoadException(std::string("MIGraphX model initialization failed: ") + e.what());
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
MIGraphXInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    validate_input(input_tensors);
    if (input_tensors.size() != input_shapes_.size())
        throw InferenceExecutionException("MIGraphX input count mismatch");
    migraphx::program_parameters params;
    for (size_t i = 0; i < input_shapes_.size(); ++i) {
        const auto& shape = input_shapes_[i];
        if (input_tensors[i].size() != shape.bytes())
            throw InferenceExecutionException("MIGraphX input byte count mismatch: " + input_names_[i]);
        params.add(input_names_[i].c_str(), migraphx::argument(shape, const_cast<uint8_t*>(input_tensors[i].data())));
    }
    start_timer();
    const auto results = program_.eval(params);
    end_timer();
    ++total_inferences_;

    std::vector<std::vector<TensorElement>> output_tensors;
    std::vector<std::vector<int64_t>> shapes;
    for (const auto& res : results) {
        const auto shape = res.get_shape();
        const auto lens = shape.lengths();
        std::vector<TensorElement> values;
        values.reserve(shape.elements());
        switch (shape.type()) {
        case migraphx_shape_float_type:
            append_values<float>(values, res);
            break;
        case migraphx_shape_int32_type:
            append_values<int32_t>(values, res);
            break;
        case migraphx_shape_int64_type:
            append_values<int64_t>(values, res);
            break;
        case migraphx_shape_uint8_type:
            append_values<uint8_t>(values, res);
            break;
        case migraphx_shape_int8_type:
            append_values<int8_t, int32_t>(values, res);
            break;
        case migraphx_shape_bool_type:
            append_values<uint8_t>(values, res);
            break;
        default:
            throw InferenceExecutionException("MIGraphX output datatype is unsupported");
        }
        output_tensors.push_back(std::move(values));
        shapes.emplace_back(lens.begin(), lens.end());
    }
    return {std::move(output_tensors), std::move(shapes)};
}
