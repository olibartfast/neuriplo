#include "DALIInfer.hpp"

#include <cstdlib>
#include <cstring>
#include <dali/c_api.h>
#include <dali/operators.h>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace {

// Process-global, exactly once before any pipeline is created. Both calls are
// required: daliInitialize() brings up the backend, daliInitOperators()
// registers the operator schemas from libdali_operators.so, which nothing loads
// implicitly. Without the second call deserialization succeeds and the first run
// fails with `No schema found for operator "decoders__Image"`.
void ensure_dali_initialized() {
    static std::once_flag once;
    std::call_once(once, [] {
        daliInitialize();
        daliInitOperators();
    });
}

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw ModelLoadException("cannot open serialized DALI pipeline: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Guards against DALI's unsigned-wrapped "not declared" sentinel sizing a vector.
constexpr size_t kMaxTensorRank = 8;

// DALI has thirteen output types; this backend carries the ones neuriplo's
// TensorDtype can represent exactly. Anything else is rejected rather than
// reported as FP32, which would mis-size the buffer and silently corrupt the
// tensor (FP16 being the realistic case for a half-precision pipeline).
struct DaliTypeMapping {
    TensorDtype dtype;
    size_t size;
};

DaliTypeMapping map_dali_type(dali_data_type_t type) {
    switch (type) {
    case DALI_UINT8:
        return {TensorDtype::UINT8, 1};
    case DALI_INT32:
        return {TensorDtype::INT32, 4};
    case DALI_INT64:
        return {TensorDtype::INT64, 8};
    case DALI_FLOAT:
        return {TensorDtype::FP32, 4};
    default:
        break;
    }
    throw InferenceExecutionException("DALI output type " + std::to_string(static_cast<int>(type)) +
                                      " is not supported by this backend; the pipeline must emit "
                                      "UINT8, INT32, INT64, or FLOAT");
}

bool is_supported_dali_type(dali_data_type_t type) {
    return type == DALI_UINT8 || type == DALI_INT32 || type == DALI_INT64 || type == DALI_FLOAT;
}

size_t dali_type_size(dali_data_type_t type) {
    switch (type) {
    case DALI_UINT8:
    case DALI_INT8:
    case DALI_BOOL:
        return 1;
    case DALI_UINT16:
    case DALI_INT16:
    case DALI_FLOAT16:
        return 2;
    case DALI_UINT32:
    case DALI_INT32:
    case DALI_FLOAT:
        return 4;
    case DALI_UINT64:
    case DALI_INT64:
    case DALI_FLOAT64:
        return 8;
    default:
        break;
    }
    throw InferenceExecutionException("unsupported DALI external input type: " +
                                      std::to_string(static_cast<int>(type)));
}

// daliShapeAt returns memory the caller owns (see dali/c_api.h).
struct DaliShapeDeleter {
    void operator()(int64_t* shape) const noexcept { std::free(shape); }
};
using DaliShape = std::unique_ptr<int64_t, DaliShapeDeleter>;

int64_t element_count(const std::vector<int64_t>& shape) {
    int64_t count = 1;
    for (const auto dim : shape) {
        const int64_t extent = dim > 0 ? dim : 1;
        if (count > std::numeric_limits<int64_t>::max() / extent) {
            throw InferenceException("declared shape overflows the element count");
        }
        count *= extent;
    }
    return count;
}

} // namespace

struct DALIInfer::Impl {
    daliPipelineHandle handle{};
    bool output_shared{false};
    std::vector<int64_t> declared_output_shape;
    std::vector<std::vector<int64_t>> last_output_shapes;
    std::vector<dali_data_type_t> declared_output_dtypes;
    std::vector<size_t> declared_output_ndims;
    std::vector<dali_data_type_t> last_output_dtypes;
    unsigned output_count{0};

    // Names of every external source, in declaration order. A preprocessing
    // pipeline has one (the encoded image); a postprocessing pipeline has one
    // per model output it consumes, so the backend cannot assume either shape.
    std::vector<std::string> external_inputs;
    std::vector<dali_data_type_t> external_input_types;
    // Output names: DALI identifies outputs positionally, so a caller that
    // needs them addressable by name (an ensemble graph mapping an envelope)
    // supplies them through the model_path "outnames=" suffix.
    std::vector<std::string> output_names;

    Impl(const std::string& path_spec, const std::vector<std::vector<int64_t>>& input_sizes) {
        ensure_dali_initialized();

        // "pipeline.dali|plugin=libfoo.so": pipelines built on custom operators
        // (GPU postprocessing) need their plugin loaded before deserialization,
        // or the operator schema is unknown.
        // model_path may carry suffixes: "pipeline.dali|plugin=libfoo.so|out=3x640x640".
        // input_sizes keeps its usual meaning (per-input shapes); the declared
        // output shape is separate because DALI reports output shapes only
        // after a run and the caller needs one to advertise metadata at load.
        std::string path = path_spec;
        std::string plugin;
        {
            size_t cursor = path.find('|');
            const std::string spec = cursor == std::string::npos ? "" : path.substr(cursor + 1);
            if (cursor != std::string::npos) {
                path = path.substr(0, cursor);
            }
            size_t start = 0;
            while (start < spec.size()) {
                const size_t end = spec.find('|', start);
                const std::string field = spec.substr(start, end - start);
                if (field.rfind("plugin=", 0) == 0) {
                    plugin = field.substr(7);
                } else if (field.rfind("outnames=", 0) == 0) {
                    std::string names = field.substr(9);
                    size_t pos = 0;
                    while (pos <= names.size()) {
                        const size_t next = names.find(',', pos);
                        output_names.push_back(names.substr(pos, next - pos));
                        if (next == std::string::npos) {
                            break;
                        }
                        pos = next + 1;
                    }
                } else if (field.rfind("out=", 0) == 0) {
                    std::string dims = field.substr(4);
                    size_t pos = 0;
                    while (pos < dims.size()) {
                        const size_t next = dims.find('x', pos);
                        declared_output_shape.push_back(std::stoll(dims.substr(pos, next - pos)));
                        if (next == std::string::npos) {
                            break;
                        }
                        pos = next + 1;
                    }
                }
                if (end == std::string::npos) {
                    break;
                }
                start = end + 1;
            }
        }
        if (!plugin.empty()) {
            try {
                daliLoadLibrary(plugin.c_str());
            } catch (...) {
                throw ModelLoadException("could not load DALI operator plugin: " + plugin);
            }
        }

        const std::string serialized = read_file(path);
        if (serialized.empty()) {
            throw ModelLoadException("serialized DALI pipeline is empty: " + path);
        }
        if (daliIsDeserializable(serialized.c_str(), static_cast<int>(serialized.size())) != 0) {
            throw ModelLoadException("not a serialized DALI pipeline: " + path);
        }
        daliDeserializeDefault(&handle, serialized.c_str(), static_cast<int>(serialized.size()));

        try {
            const int max_batch_size = daliGetMaxBatchSize(&handle);
            if (max_batch_size != 1) {
                throw ModelLoadException("serialized DALI pipeline has max batch size " +
                                         std::to_string(max_batch_size) + "; only max batch size 1 is supported");
            }

            const int num_inputs = daliGetNumExternalInput(&handle);
            if (num_inputs < 1) {
                throw ModelLoadException("DALI pipeline declares no external source: " + path);
            }
            for (int i = 0; i < num_inputs; ++i) {
                const char* name = daliGetExternalInputName(&handle, i);
                if (name == nullptr) {
                    throw ModelLoadException("DALI pipeline has an unnamed external source: " + path);
                }
                external_inputs.emplace_back(name);
                external_input_types.push_back(daliGetExternalInputType(&handle, name));
            }

            output_count = daliGetNumOutput(&handle);
            if (output_count < 1) {
                throw ModelLoadException("DALI pipeline has no outputs: " + path);
            }
            if (!output_names.empty() && output_names.size() != output_count) {
                throw ModelLoadException("outnames= supplies " + std::to_string(output_names.size()) +
                                         " names for a pipeline with " + std::to_string(output_count) + " outputs");
            }
            declared_output_dtypes.reserve(output_count);
            declared_output_ndims.reserve(output_count);
            for (unsigned index = 0; index < output_count; ++index) {
                const auto dtype = daliGetDeclaredOutputDtype(&handle, static_cast<int>(index));
                if (dtype != DALI_NO_TYPE && !is_supported_dali_type(dtype)) {
                    throw ModelLoadException("DALI output " + std::to_string(index) + " declares unsupported type " +
                                             std::to_string(static_cast<int>(dtype)));
                }
                declared_output_dtypes.push_back(dtype);
                declared_output_ndims.push_back(daliGetDeclaredOutputNdim(&handle, static_cast<int>(index)));
            }

            (void)input_sizes;
        } catch (...) {
            try {
                if (handle != nullptr) {
                    daliDeletePipeline(&handle);
                }
            } catch (...) {
                // Preserve the original DALI or metadata exception.
            }
            throw;
        }
    }

    ~Impl() {
        if (output_shared) {
            daliOutputRelease(&handle);
        }
        daliDeletePipeline(&handle);
    }

    struct Output {
        std::vector<uint8_t> bytes;
        std::vector<int64_t> shape;
        TensorDtype dtype = TensorDtype::FP32;
    };

    // Feeds each declared external source from the corresponding request
    // tensor, runs the pipeline, and copies every output to host.
    //
    // A preprocessing pipeline takes one encoded image and emits the model
    // tensor plus the source dimensions. A postprocessing pipeline takes the
    // model's outputs and emits a decoded result envelope. The backend does not
    // distinguish them: it feeds what the pipeline declares, in order.
    std::vector<Output> run(const std::vector<std::vector<uint8_t>>& inputs,
                            const std::vector<std::vector<int64_t>>& shapes) {
        if (inputs.size() != external_inputs.size()) {
            throw InferenceExecutionException("DALI pipeline declares " + std::to_string(external_inputs.size()) +
                                              " external inputs but received " + std::to_string(inputs.size()));
        }

        if (output_shared) {
            daliOutputRelease(&handle);
            output_shared = false;
        }

        for (size_t i = 0; i < external_inputs.size(); ++i) {
            if (inputs[i].empty()) {
                throw InferenceExecutionException("DALI backend received empty data for input '" + external_inputs[i] +
                                                  "'");
            }
            const auto type = external_input_types[i];
            const size_t element_size = dali_type_size(type);
            if (inputs[i].size() % element_size != 0) {
                throw InferenceExecutionException("DALI external input '" + external_inputs[i] +
                                                  "' byte count is not aligned to its datatype");
            }

            // DALI asserts on rank, so the sample shape must match the rank the
            // external source declares. A caller-declared shape is trimmed of
            // leading batch dimensions; a rank-1 source otherwise takes the flat
            // element count, which is the encoded-image case.
            const int declared_ndim = daliGetExternalInputNdim(&handle, external_inputs[i].c_str());
            std::vector<int64_t> sample_shape;
            if (i < shapes.size() && !shapes[i].empty()) {
                sample_shape = shapes[i];
                while (static_cast<int>(sample_shape.size()) > declared_ndim && !sample_shape.empty() &&
                       sample_shape.front() == 1) {
                    sample_shape.erase(sample_shape.begin());
                }
            }
            if (static_cast<int>(sample_shape.size()) != declared_ndim) {
                if (declared_ndim == 1) {
                    if (inputs[i].size() % element_size != 0) {
                        // Truncating to a whole element count would silently
                        // discard the trailing bytes; reject instead.
                        throw InferenceExecutionException("DALI external input '" + external_inputs[i] +
                                                          "' byte count is not aligned to its datatype");
                    }
                    sample_shape = {static_cast<int64_t>(inputs[i].size() / element_size)};
                } else {
                    throw InferenceExecutionException("DALI external input '" + external_inputs[i] + "' expects rank " +
                                                      std::to_string(declared_ndim) +
                                                      " but no matching shape was declared");
                }
            }

            // DALI reads product(sample_shape) * element_size bytes from the
            // pointer with no length argument, so a shape that outruns the
            // buffer is an out-of-bounds read inside the library with no
            // diagnostic. Fail here instead, where the names are still known.
            const auto count = static_cast<uint64_t>(element_count(sample_shape));
            if (count > std::numeric_limits<size_t>::max() ||
                static_cast<size_t>(count) > std::numeric_limits<size_t>::max() / element_size) {
                throw InferenceExecutionException("DALI external input '" + external_inputs[i] +
                                                  "' declared shape overflows its byte count");
            }
            const size_t required = static_cast<size_t>(count) * element_size;
            if (required > inputs[i].size()) {
                throw InferenceExecutionException("DALI external input '" + external_inputs[i] + "' needs " +
                                                  std::to_string(required) + " bytes for the declared shape but only " +
                                                  std::to_string(inputs[i].size()) + " were supplied");
            }

            daliSetExternalInput(&handle, external_inputs[i].c_str(), device_type_t::CPU, inputs[i].data(), type,
                                 sample_shape.data(), static_cast<int>(sample_shape.size()), nullptr,
                                 DALI_ext_force_copy);
        }

        daliRun(&handle);
        daliOutput(&handle);
        output_shared = true;

        const unsigned num_outputs = daliGetNumOutput(&handle);
        if (num_outputs != output_count) {
            throw InferenceExecutionException("DALI output count changed from " + std::to_string(output_count) +
                                              " to " + std::to_string(num_outputs));
        }
        std::vector<Output> outputs;
        outputs.reserve(num_outputs);
        for (unsigned index = 0; index < num_outputs; ++index) {
            outputs.push_back(read_output(static_cast<int>(index)));
        }
        std::vector<std::vector<int64_t>> runtime_shapes;
        std::vector<dali_data_type_t> runtime_dtypes;
        runtime_shapes.reserve(outputs.size());
        runtime_dtypes.reserve(outputs.size());
        for (unsigned index = 0; index < num_outputs; ++index) {
            runtime_shapes.push_back(outputs[index].shape);
            runtime_dtypes.push_back(daliTypeAt(&handle, static_cast<int>(index)));
        }
        last_output_shapes = std::move(runtime_shapes);
        last_output_dtypes = std::move(runtime_dtypes);
        return outputs;
    }

    Output read_output(int index) {
        // daliShapeAt returns the batch-inclusive shape. Derive the sample rank
        // from the successful runtime output, because the declared rank may be
        // absent or may not describe a dynamic output.
        const size_t ndim = daliMaxDimTensors(&handle, index);
        if (ndim == 0 || ndim > kMaxTensorRank) {
            throw InferenceExecutionException("DALI pipeline reported an implausible output rank: " +
                                              std::to_string(ndim));
        }
        const DaliShape dims(daliShapeAt(&handle, index));
        if (!dims) {
            throw InferenceExecutionException("DALI pipeline returned no output shape");
        }

        Output output;
        // The runtime type, not the declared one: a pipeline that declares no
        // output type still produces concretely typed data.
        const auto mapping = map_dali_type(daliTypeAt(&handle, index));
        output.dtype = mapping.dtype;
        const size_t bytes = daliTensorSize(&handle, index);

        // daliShapeAt returns the batch-inclusive shape. Use the runtime rank,
        // rather than a declaration that may be absent or stale, and retain the
        // batch dimension exactly once.
        output.shape.assign(dims.get(), dims.get() + ndim + 1);

        // daliOutputCopy takes no destination length, so a daliTensorSize that
        // under-reports what the pipeline actually writes corrupts the heap
        // silently and aborts in some unrelated later allocation. Pad the
        // destination and check the padding rather than trusting the number.
        constexpr size_t kRedzone = 64;
        constexpr uint8_t kRedzoneByte = 0xA5;
        output.bytes.assign(bytes + kRedzone, kRedzoneByte);
        // Destination is host memory, so DALI does the device-to-host copy. A
        // serving pipeline that owned the downstream device buffer would copy
        // straight into it and skip the host round trip entirely.
        daliOutputCopy(&handle, output.bytes.data(), index, device_type_t::CPU, 0, DALI_ext_force_sync);
        for (size_t i = bytes; i < bytes + kRedzone; ++i) {
            if (output.bytes[i] != kRedzoneByte) {
                throw InferenceExecutionException(
                    "DALI wrote past the end of output " + std::to_string(index) + ": daliTensorSize reported " +
                    std::to_string(bytes) + " bytes but the copy overran by at least " + std::to_string(i - bytes + 1));
            }
        }
        output.bytes.resize(bytes);
        return output;
    }
};

DALIInfer::DALIInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                     const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface(model_path, use_gpu, batch_size, input_sizes),
      impl_(std::make_unique<Impl>(model_path, input_sizes)), input_sizes_(input_sizes), input_shapes_(input_sizes) {
    if (batch_size != 1) {
        throw ModelLoadException("DALI backend currently supports batch size 1 only");
    }
}

DALIInfer::~DALIInfer() = default;

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
DALIInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    const auto raw = get_infer_results_raw(input_tensors);

    std::vector<std::vector<TensorElement>> data;
    std::vector<std::vector<int64_t>> shapes;
    data.reserve(raw.size());
    shapes.reserve(raw.size());

    for (const auto& tensor : raw) {
        std::vector<TensorElement> elements;
        if (tensor.dtype == TensorDtype::FP32) {
            const auto count = tensor.bytes.size() / sizeof(float);
            elements.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                float value = 0.0F;
                std::memcpy(&value, tensor.bytes.data() + i * sizeof(float), sizeof(float));
                elements.emplace_back(value);
            }
        } else {
            const size_t element_size = tensor_dtype_size(tensor.dtype);
            if (element_size == 0 || tensor.bytes.size() % element_size != 0) {
                throw InferenceExecutionException("DALI output byte count is not aligned to its datatype");
            }
            const size_t count = tensor.bytes.size() / element_size;
            elements.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                if (tensor.dtype == TensorDtype::INT32) {
                    int32_t value = 0;
                    std::memcpy(&value, tensor.bytes.data() + i * sizeof(value), sizeof(value));
                    elements.emplace_back(value);
                } else if (tensor.dtype == TensorDtype::INT64) {
                    int64_t value = 0;
                    std::memcpy(&value, tensor.bytes.data() + i * sizeof(value), sizeof(value));
                    elements.emplace_back(value);
                } else {
                    elements.emplace_back(tensor.bytes[i]);
                }
            }
        }
        data.push_back(std::move(elements));
        shapes.push_back(tensor.shape);
    }

    return {std::move(data), std::move(shapes)};
}

std::vector<RawOutputTensor> DALIInfer::get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) {
    auto produced = impl_->run(input_tensors, input_shapes_);
    std::vector<RawOutputTensor> outputs;
    outputs.reserve(produced.size());
    for (auto& item : produced) {
        RawOutputTensor tensor;
        tensor.bytes = std::move(item.bytes);
        tensor.shape = std::move(item.shape);
        tensor.dtype = item.dtype;
        outputs.push_back(std::move(tensor));
    }
    return outputs;
}

InferenceMetadata DALIInfer::get_inference_metadata() {
    InferenceMetadata metadata;
    for (size_t i = 0; i < impl_->external_inputs.size(); ++i) {
        const auto& name = impl_->external_inputs[i];
        const auto type = impl_->external_input_types[i];
        std::vector<int64_t> shape{1, -1};
        if (name != kEncodedInputName && i < input_sizes_.size()) {
            shape = input_sizes_[i];
        }
        const auto to_input_type = [](dali_data_type_t type) -> TensorDataType {
            switch (type) {
            case DALI_UINT8:
                return TensorDataType::UInt8;
            case DALI_INT8:
                return TensorDataType::Int8;
            case DALI_INT32:
                return TensorDataType::Int32;
            case DALI_INT64:
                return TensorDataType::Int64;
            case DALI_BOOL:
                return TensorDataType::Bool;
            case DALI_FLOAT:
                return TensorDataType::Float32;
            // Anything else the pipeline accepts is silently miscast under
            // any fallback; refuse to advertise it.
            default:
                throw InferenceException("DALI external input type " + std::to_string(static_cast<int>(type)) +
                                         " has no neuriplo metadata representation");
            }
        };
        metadata.addInput(name, shape, 1, to_input_type(type));
    }

    const bool has_runtime = !impl_->last_output_shapes.empty();
    const auto to_metadata_type = [](dali_data_type_t type) {
        switch (type) {
        case DALI_UINT8:
            return TensorDataType::UInt8;
        case DALI_INT32:
            return TensorDataType::Int32;
        case DALI_INT64:
            return TensorDataType::Int64;
        case DALI_FLOAT:
            return TensorDataType::Float32;
        default:
            throw InferenceException("DALI output metadata has no concrete datatype; run inference successfully "
                                     "before requesting metadata");
        }
    };

    for (unsigned i = 0; i < impl_->output_count; ++i) {
        dali_data_type_t dtype = impl_->declared_output_dtypes[i];
        if (has_runtime && i < impl_->last_output_dtypes.size()) {
            dtype = impl_->last_output_dtypes[i];
        } else if (dtype == DALI_NO_TYPE) {
            throw InferenceException("DALI output " + std::to_string(i) +
                                     " has undeclared datatype; run inference successfully before requesting metadata");
        }

        std::vector<int64_t> shape;
        if (has_runtime && i < impl_->last_output_shapes.size()) {
            shape = impl_->last_output_shapes[i];
        } else if (i == 0 && !impl_->declared_output_shape.empty()) {
            shape = impl_->declared_output_shape;
            // Runtime shapes carry a batch factor; make the declared hint
            // batch-inclusive the same way, whatever its rank.
            shape.insert(shape.begin(), 1);
        } else {
            const size_t declared_ndim = impl_->declared_output_ndims[i];
            if (declared_ndim != std::numeric_limits<size_t>::max() && declared_ndim <= kMaxTensorRank) {
                shape.assign(declared_ndim + 1, -1);
                shape.front() = 1;
            }
        }

        const std::string name = i < impl_->output_names.size() ? impl_->output_names[i] : "output" + std::to_string(i);
        metadata.addOutput(name, shape, 1, to_metadata_type(dtype));
    }
    return metadata;
}
