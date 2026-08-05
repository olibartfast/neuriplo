#include "DALIInfer.hpp"

#include <cstdlib>
#include <cstring>
#include <dali/c_api.h>
#include <dali/operators.h>
#include <fstream>
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

// daliShapeAt returns memory the caller owns (see dali/c_api.h).
struct DaliShapeDeleter {
    void operator()(int64_t* shape) const noexcept { std::free(shape); }
};
using DaliShape = std::unique_ptr<int64_t, DaliShapeDeleter>;

int64_t element_count(const std::vector<int64_t>& shape) {
    int64_t count = 1;
    for (const auto dim : shape) {
        count *= dim > 0 ? dim : 1;
    }
    return count;
}

} // namespace

struct DALIInfer::Impl {
    daliPipelineHandle handle{};
    bool output_shared{false};
    std::vector<int64_t> declared_output_shape;
    std::vector<int64_t> last_output_shape;
    dali_data_type_t output_dtype{DALI_FLOAT};

    Impl(const std::string& path, const std::vector<std::vector<int64_t>>& input_sizes) {
        ensure_dali_initialized();

        const std::string serialized = read_file(path);
        if (serialized.empty()) {
            throw ModelLoadException("serialized DALI pipeline is empty: " + path);
        }
        if (daliIsDeserializable(serialized.c_str(), static_cast<int>(serialized.size())) != 0) {
            throw ModelLoadException("not a serialized DALI pipeline: " + path);
        }
        daliDeserializeDefault(&handle, serialized.c_str(), static_cast<int>(serialized.size()));

        // Fail at load if the pipeline does not expose the external source this
        // backend feeds, rather than producing wrong tensors later.
        const int num_inputs = daliGetNumExternalInput(&handle);
        bool found = false;
        std::string names;
        for (int i = 0; i < num_inputs; ++i) {
            const char* name = daliGetExternalInputName(&handle, i);
            if (name == nullptr) {
                continue;
            }
            if (!names.empty()) {
                names += ", ";
            }
            names += name;
            if (std::string(name) == DALIInfer::kEncodedInputName) {
                found = true;
            }
        }
        if (!found) {
            daliDeletePipeline(&handle);
            throw ModelLoadException(std::string("DALI pipeline has no external source named '") +
                                     DALIInfer::kEncodedInputName + "' (found: " + (names.empty() ? "none" : names) +
                                     "): " + path);
        }

        if (daliGetNumOutput(&handle) < 1) {
            daliDeletePipeline(&handle);
            throw ModelLoadException("DALI pipeline has no outputs: " + path);
        }
        output_dtype = daliGetDeclaredOutputDtype(&handle, 0);

        // DALI reports output shapes only after a run, so the caller declares
        // the shape the downstream model expects.
        if (!input_sizes.empty() && !input_sizes[0].empty()) {
            declared_output_shape = input_sizes[0];
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

    // Feeds one encoded image, runs the pipeline, and copies every output to
    // host. A preprocessing pipeline emits more than the model tensor: the
    // original image shape travels alongside it, because postprocessing needs
    // the source dimensions to map boxes back and the decode is the only place
    // that knows them.
    std::vector<Output> run(const std::vector<uint8_t>& encoded) {
        if (encoded.empty()) {
            throw InferenceExecutionException("DALI backend received an empty encoded image");
        }

        if (output_shared) {
            daliOutputRelease(&handle);
            output_shared = false;
        }

        const int64_t shape = static_cast<int64_t>(encoded.size());
        daliSetExternalInput(&handle, DALIInfer::kEncodedInputName, device_type_t::CPU, encoded.data(), DALI_UINT8,
                             &shape, 1, nullptr, DALI_ext_force_copy);
        daliRun(&handle);
        daliOutput(&handle);
        output_shared = true;

        const unsigned num_outputs = daliGetNumOutput(&handle);
        std::vector<Output> outputs;
        outputs.reserve(num_outputs);
        for (unsigned index = 0; index < num_outputs; ++index) {
            outputs.push_back(read_output(static_cast<int>(index)));
        }
        last_output_shape = outputs.front().shape;
        return outputs;
    }

    Output read_output(int index) {
        // daliShapeAt returns the sample shape for a uniform batch. The
        // declared rank is a negative sentinel when the pipeline does not
        // declare one, which is unsigned here, so it is range-checked before it
        // sizes anything.
        const auto declared = static_cast<int64_t>(daliGetDeclaredOutputNdim(&handle, index));
        size_t ndim = (declared > 0 && declared <= static_cast<int64_t>(kMaxTensorRank))
                          ? static_cast<size_t>(declared)
                          : daliMaxDimTensors(&handle, index);
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

        // The declared rank is the *sample* rank while daliShapeAt returns the
        // batch-inclusive shape, so reading `ndim` entries drops the last axis.
        // Take the batch dimension too, then check the product against the byte
        // count rather than trusting either number on its own.
        output.shape.assign(dims.get(), dims.get() + ndim + 1);
        if (static_cast<size_t>(element_count(output.shape)) * mapping.size != bytes) {
            output.shape.assign(dims.get(), dims.get() + ndim);
        }

        output.bytes.resize(bytes);
        // Destination is host memory, so DALI does the device-to-host copy. A
        // serving pipeline that owned the downstream device buffer would copy
        // straight into it and skip the host round trip entirely.
        daliOutputCopy(&handle, output.bytes.data(), index, device_type_t::CPU, 0, DALI_ext_force_sync);
        return output;
    }
};

DALIInfer::DALIInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                     const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface(model_path, use_gpu, batch_size, input_sizes),
      impl_(std::make_unique<Impl>(model_path, input_sizes)) {
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
            elements.reserve(tensor.bytes.size());
            for (const auto byte : tensor.bytes) {
                elements.emplace_back(byte);
            }
        }
        data.push_back(std::move(elements));
        shapes.push_back(tensor.shape);
    }

    return {std::move(data), std::move(shapes)};
}

std::vector<RawOutputTensor> DALIInfer::get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) {
    if (input_tensors.size() != 1) {
        throw InferenceExecutionException("DALI backend expects exactly one input (the encoded image)");
    }

    auto produced = impl_->run(input_tensors[0]);
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
    // Variable-length encoded bytes: the extent is per-request.
    metadata.addInput(kEncodedInputName, {1, -1}, 1, TensorDataType::UInt8);

    auto shape = impl_->declared_output_shape;
    if (shape.empty()) {
        shape = impl_->last_output_shape;
    }
    if (shape.size() == 3) {
        shape.insert(shape.begin(), 1);
    }
    metadata.addOutput(kPreprocessedOutputName, shape, 1,
                       impl_->output_dtype == DALI_UINT8   ? TensorDataType::UInt8
                       : impl_->output_dtype == DALI_INT32 ? TensorDataType::Int32
                       : impl_->output_dtype == DALI_INT64 ? TensorDataType::Int64
                                                           : TensorDataType::Float32);
    // Source image dimensions, carried so a downstream postprocess step can map
    // results back onto the original frame.
    metadata.addOutput(kImageShapeOutputName, {1, 3}, 1, TensorDataType::Int32);
    return metadata;
}
