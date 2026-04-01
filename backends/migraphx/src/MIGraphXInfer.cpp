#include "MIGraphXInfer.hpp"
#include <numeric>
#include <stdexcept>

MIGraphXInfer::MIGraphXInfer(const std::string& model_path, bool use_gpu,
                             size_t batch_size,
                             const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}
    , use_gpu_{use_gpu}
{
    // Parse ONNX — fix the batch dimension before compiling
    migraphx::onnx_options onnx_opts;
    onnx_opts.set_default_dim_value(static_cast<unsigned>(batch_size));

    try {
        program_ = migraphx::parse_onnx(model_path.c_str(), onnx_opts);
    } catch (const std::exception& e) {
        throw ModelLoadException(std::string("MIGraphX parse_onnx failed: ") + e.what());
    }

    // Compile for the requested target
    if (use_gpu_) {
        LOG(INFO) << "MIGraphX: compiling for GPU (hip)";
        program_.compile(migraphx::target("gpu"));
        gpu_available_ = true;
    } else {
        LOG(INFO) << "MIGraphX: compiling for CPU (ref)";
        program_.compile(migraphx::target("ref"));
    }

    // Collect input metadata from compiled parameter shapes
    auto param_shapes = program_.get_parameter_shapes();
    for (const auto& name : param_shapes.names()) {
        const auto& shape = param_shapes[name];
        auto lens = shape.lengths();                        // std::vector<size_t>
        std::vector<int64_t> dims(lens.begin(), lens.end());
        inference_metadata_.addInput(name, dims, batch_size);
        input_names_.push_back(name);
        LOG(INFO) << "MIGraphX input: " << name;
    }

    // Discover output shapes with a dummy eval using zero-filled buffers
    {
        migraphx::program_parameters dummy_params;
        for (const auto& name : input_names_) {
            const auto& meta = inference_metadata_.getInputs();
            auto it = std::find_if(meta.begin(), meta.end(),
                                   [&](const LayerInfo& l){ return l.name == name; });
            std::vector<std::size_t> lens(it->shape.begin(), it->shape.end());
            migraphx::shape s{migraphx_shape_float_type, lens};
            std::size_t n = s.elements();
            dummy_buf_.emplace_back(n * sizeof(float), 0);
            dummy_params.add(name.c_str(),
                             migraphx::argument(s, dummy_buf_.back().data()));
        }
        auto results = program_.eval(dummy_params);
        for (size_t i = 0; i < results.size(); ++i) {
            auto lens = results[i].get_shape().lengths();
            std::vector<int64_t> dims(lens.begin(), lens.end());
            std::string out_name = "output_" + std::to_string(i);
            inference_metadata_.addOutput(out_name, dims, batch_size);
            output_names_.push_back(out_name);
            LOG(INFO) << "MIGraphX output[" << i << "] detected";
        }
        dummy_buf_.clear();
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
MIGraphXInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors)
{
    validate_input(input_tensors);

    const auto& inputs = inference_metadata_.getInputs();
    if (input_tensors.size() != inputs.size()) {
        throw std::runtime_error(
            "Input count mismatch: expected " + std::to_string(inputs.size()) +
            ", got " + std::to_string(input_tensors.size()));
    }

    start_timer();

    migraphx::program_parameters params;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& meta = inputs[i];
        std::vector<std::size_t> lens(meta.shape.begin(), meta.shape.end());
        migraphx::shape s{migraphx_shape_float_type, lens};
        // argument wraps existing buffer — no copy
        params.add(input_names_[i].c_str(),
                   migraphx::argument(s,
                       const_cast<void*>(static_cast<const void*>(input_tensors[i].data()))));
    }

    auto results = program_.eval(params);

    end_timer();
    ++total_inferences_;

    std::vector<std::vector<TensorElement>> output_tensors;
    std::vector<std::vector<int64_t>> shapes;

    for (const auto& res : results) {
        auto lens = res.get_shape().lengths();
        std::vector<int64_t> shape(lens.begin(), lens.end());

        std::size_t num_elements = res.get_shape().elements();
        const float* data = reinterpret_cast<const float*>(res.data());

        std::vector<TensorElement> tensor_data;
        tensor_data.reserve(num_elements);
        for (std::size_t j = 0; j < num_elements; ++j)
            tensor_data.emplace_back(data[j]);

        output_tensors.push_back(std::move(tensor_data));
        shapes.push_back(std::move(shape));
    }

    return {std::move(output_tensors), std::move(shapes)};
}
