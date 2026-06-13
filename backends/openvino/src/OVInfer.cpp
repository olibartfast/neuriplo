#include "OVInfer.hpp"

#include <filesystem>
#include <numeric>
#include <sstream>

// Helper function to print ov::Shape and ov::PartialShape
template <typename ShapeType> std::string OVInfer::print_shape(const ShapeType& shape) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if constexpr (std::is_same<ShapeType, ov::PartialShape>::value) { // Note the 'constexpr'
            if (shape[i].is_dynamic()) {
                ss << "?";
            } else {
                ss << shape[i].get_length();
            }
        } else { // It's ov::Shape
            ss << shape[i];
        }

        if (i < shape.size() - 1) {
            ss << ",";
        }
    }
    ss << "]";
    return ss.str();
}

TensorDataType OVInfer::inputTensorDataType(ov::element::Type type) {
    switch (type) {
    case ov::element::f32:
        return TensorDataType::Float32;
    case ov::element::i32:
        return TensorDataType::Int32;
    case ov::element::i64:
        return TensorDataType::Int64;
    case ov::element::u8:
        return TensorDataType::UInt8;
    case ov::element::i8:
        return TensorDataType::Int8;
    case ov::element::boolean:
        return TensorDataType::Bool;
    default:
        throw std::runtime_error("Unsupported OpenVINO input tensor element type for metadata datatype: " +
                                 type.get_type_name());
    }
}

TensorDataType OVInfer::outputTensorDataType(ov::element::Type type) {
    switch (type) {
    case ov::element::f32:
        return TensorDataType::Float32;
    case ov::element::i32:
        return TensorDataType::Int32;
    case ov::element::i64:
        return TensorDataType::Int64;
    case ov::element::u8:
        return TensorDataType::UInt8;
    default:
        throw std::runtime_error("Unsupported OpenVINO output tensor element type for metadata datatype: " +
                                 type.get_type_name());
    }
}

OVInfer::OVInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                 const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
    std::filesystem::path fs_path(model_path);

    try {
        if (fs_path.extension() == ".onnx") {
            model_ = core_.read_model(model_path);
        } else {
            // Handle both .xml and .bin paths
            std::string model_config;
            if (model_path.find(".xml") != std::string::npos) {
                model_config = model_path;
            } else if (model_path.find(".bin") != std::string::npos) {
                model_config = model_path.substr(0, model_path.find(".bin")) + ".xml";
            } else {
                // Assume it's a base name, add .xml extension
                model_config = model_path + ".xml";
            }

            if (!std::filesystem::exists(model_config)) {
                throw std::runtime_error("XML file not found: " + model_config);
            }

            model_ = core_.read_model(model_config);
        }

        // --- Handle dynamic shapes before compiling the model ---
        std::map<ov::Output<ov::Node>, ov::PartialShape> all_shapes; // Store dynamic shapes for all inputs

        LOG(INFO) << "Input Node Name/Shape (" << model_->inputs().size() << "):";
        for (size_t i = 0; i < model_->inputs().size(); ++i) {
            auto input = model_->input(i);
            std::string name = input.get_any_name();
            ov::PartialShape partial_shape = input.get_partial_shape();

            // Non-batch dynamic dims require explicit input_sizes; batch-only
            // dynamic models (common after ovc) can compile with batch_size alone.
            bool non_batch_dynamic = false;
            for (size_t j = 1; j < partial_shape.size(); ++j) {
                if (partial_shape[j].is_dynamic()) {
                    non_batch_dynamic = true;
                    break;
                }
            }

            if (non_batch_dynamic || (!input_sizes.empty() && i < input_sizes.size())) {
                if (input_sizes.empty() || i >= input_sizes.size()) {
                    if (non_batch_dynamic) {
                        throw std::runtime_error("Dynamic shapes found but no input sizes provided for input '" + name +
                                                 "'");
                    }
                } else {
                    const auto& provided_shape = input_sizes[i];

                    if (non_batch_dynamic) {
                        if (provided_shape.size() != partial_shape.size() - 1) {
                            throw std::runtime_error("Provided shape size mismatch for input '" + name +
                                                     "'. Expected " + std::to_string(partial_shape.size() - 1) +
                                                     " dimensions, got " + std::to_string(provided_shape.size()));
                        }

                        for (size_t j = 1; j < partial_shape.size(); ++j) {
                            partial_shape[j] = provided_shape[j - 1];
                        }
                    } else {
                        if (provided_shape.size() != partial_shape.size() - 1) {
                            throw std::runtime_error("Provided shape size mismatch for input '" + name +
                                                     "'. Expected " + std::to_string(partial_shape.size() - 1) +
                                                     " dimensions, got " + std::to_string(provided_shape.size()));
                        }

                        for (size_t j = 1; j < partial_shape.size(); ++j) {
                            partial_shape[j] = provided_shape[j - 1];
                        }
                    }
                }
            }

            if (partial_shape[0].is_dynamic()) {
                partial_shape[0] = batch_size;
            } else {
                partial_shape[0] = batch_size;
            }

            // Store the potentially updated shape
            all_shapes[input] = partial_shape;
        }

        if (!all_shapes.empty()) {
            // Reshape the model with the gathered partial shapes
            model_->reshape(all_shapes);
        }

        // Set up device
        std::string device = use_gpu ? "GPU" : "CPU";
        LOG(INFO) << "Using device: " << device;

        try {
            compiled_model_ = core_.compile_model(model_, device);
        } catch (const ov::Exception& e) {
            if (use_gpu && device == "GPU") {
                LOG(WARNING) << "GPU not available, falling back to CPU: " << e.what();
                device = "CPU";
                compiled_model_ = core_.compile_model(model_, device);
            } else {
                throw; // Re-throw if it's not a GPU fallback case
            }
        }
        infer_request_ = compiled_model_.create_infer_request();

        // --- Process inputs after compilation ---
        for (size_t i = 0; i < model_->inputs().size(); ++i) {
            auto input = model_->input(i);
            std::string name = input.get_any_name();
            ov::Shape shape = input.get_shape();

            // Convert ov::Shape to std::vector<int64_t>
            std::vector<int64_t> shape_vec(shape.begin() + 1, shape.end());

            LOG(INFO) << "\t" << name << " : " << print_shape(shape);
            ov::element::Type input_type = input.get_element_type();
            inference_metadata_.addInput(name, shape_vec, batch_size, inputTensorDataType(input_type));

            LOG(INFO) << "\tData Type: " << input_type.get_type_name();
        }

        // Log network dimensions from first input when shape has spatial dims
        const auto& first_input_shape_vec = inference_metadata_.getInputs()[0].shape;
        if (first_input_shape_vec.size() >= 3) {
            const auto channels = static_cast<int>(first_input_shape_vec[0]);
            const auto network_height = static_cast<int>(first_input_shape_vec[1]);
            const auto network_width = static_cast<int>(first_input_shape_vec[2]);

            LOG(INFO) << "channels " << channels;
            LOG(INFO) << "width " << network_width;
            LOG(INFO) << "height " << network_height;
        }

        // Process outputs
        LOG(INFO) << "Output Node Name/Shape (" << model_->outputs().size() << "):";
        for (size_t i = 0; i < model_->outputs().size(); ++i) {
            auto output = model_->output(i);
            std::string name = output.get_any_name();

            ov::Shape shape = output.get_shape();
            ov::element::Type output_type = output.get_element_type();

            // Convert ov::Shape to std::vector<int64_t>
            std::vector<int64_t> shape_vec(shape.begin() + 1, shape.end());

            LOG(INFO) << "\t" << name << " : " << print_shape(shape);
            inference_metadata_.addOutput(name, shape_vec, batch_size, outputTensorDataType(output_type));
        }

        state_ = BackendState::Ready;
    } catch (const ov::Exception& e) {
        LOG(ERROR) << "Failed to load or process the OpenVINO model: " << e.what();
        state_ = BackendState::Failed;
        throw ModelLoadException(std::string("OpenVINO model load failed: ") + e.what());
    }
}

void OVInfer::bind_inputs_and_infer(const std::vector<std::vector<uint8_t>>& input_tensors) {
    const size_t num_inputs = model_->inputs().size();
    if (input_tensors.size() != num_inputs) {
        throw std::runtime_error("Input tensor count mismatch. Expected " + std::to_string(num_inputs) + ", got " +
                                 std::to_string(input_tensors.size()));
    }

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_port = compiled_model_.input(i);
        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(),
                                const_cast<uint8_t*>(input_tensors[i].data()));
        infer_request_.set_input_tensor(i, input_tensor);
    }

    infer_request_.infer();
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
OVInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    bind_inputs_and_infer(input_tensors);

    std::vector<std::vector<TensorElement>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    const size_t num_outputs = model_->outputs().size();
    outputs.reserve(num_outputs);
    shapes.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_tensor = infer_request_.get_output_tensor(i);
        const ov::element::Type output_type = output_tensor.get_element_type();
        const std::size_t output_size = output_tensor.get_size();

        std::vector<int64_t> output_shape(output_tensor.get_shape().begin(), output_tensor.get_shape().end());
        std::vector<TensorElement> output;
        output.reserve(output_size);

        switch (output_type) {
        case ov::element::f32: {
            const float* output_buffer = output_tensor.data<const float>();
            for (std::size_t j = 0; j < output_size; ++j) {
                output.push_back(output_buffer[j]);
            }
            break;
        }
        case ov::element::i32: {
            const int32_t* output_buffer = output_tensor.data<const int32_t>();
            for (std::size_t j = 0; j < output_size; ++j) {
                output.push_back(output_buffer[j]);
            }
            break;
        }
        case ov::element::i64: {
            const int64_t* output_buffer = output_tensor.data<const int64_t>();
            for (std::size_t j = 0; j < output_size; ++j) {
                output.push_back(output_buffer[j]);
            }
            break;
        }
        case ov::element::u8: {
            const uint8_t* output_buffer = output_tensor.data<const uint8_t>();
            for (std::size_t j = 0; j < output_size; ++j) {
                output.push_back(output_buffer[j]);
            }
            break;
        }
        default:
            LOG(ERROR) << "Unsupported output tensor type: " << output_type.get_type_name();
            state_ = BackendState::Failed;
            throw InferenceExecutionException("Unsupported output tensor type for OpenVINO: " +
                                              output_type.get_type_name());
        }

        outputs.emplace_back(std::move(output));
        shapes.emplace_back(std::move(output_shape));
    }

    return std::make_tuple(outputs, shapes);
}

std::vector<RawOutputTensor> OVInfer::get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) {
    bind_inputs_and_infer(input_tensors);

    std::vector<RawOutputTensor> raw_outputs;
    raw_outputs.reserve(model_->outputs().size());

    for (size_t i = 0; i < model_->outputs().size(); ++i) {
        auto output_tensor = infer_request_.get_output_tensor(i);
        const ov::element::Type output_type = output_tensor.get_element_type();
        const std::size_t output_size = output_tensor.get_size();

        RawOutputTensor raw;
        raw.shape.assign(output_tensor.get_shape().begin(), output_tensor.get_shape().end());

        auto copy_bytes = [&](const void* data, size_t element_size, TensorDtype dtype) {
            raw.dtype = dtype;
            const auto* bytes = static_cast<const uint8_t*>(data);
            raw.bytes.assign(bytes, bytes + output_size * element_size);
        };

        switch (output_type) {
        case ov::element::f32:
            copy_bytes(output_tensor.data<const float>(), sizeof(float), TensorDtype::FP32);
            break;
        case ov::element::i32:
            copy_bytes(output_tensor.data<const int32_t>(), sizeof(int32_t), TensorDtype::INT32);
            break;
        case ov::element::i64:
            copy_bytes(output_tensor.data<const int64_t>(), sizeof(int64_t), TensorDtype::INT64);
            break;
        case ov::element::u8:
            copy_bytes(output_tensor.data<const uint8_t>(), sizeof(uint8_t), TensorDtype::UINT8);
            break;
        default:
            LOG(ERROR) << "Unsupported output tensor type: " << output_type.get_type_name();
            state_ = BackendState::Failed;
            throw InferenceExecutionException("Unsupported output tensor type for OpenVINO: " +
                                              output_type.get_type_name());
        }

        raw_outputs.push_back(std::move(raw));
    }

    return raw_outputs;
}
