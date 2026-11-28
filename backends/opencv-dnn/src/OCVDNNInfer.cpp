#include "OCVDNNInfer.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <opencv2/core/version.hpp> // CV_VERSION, CV_VERSION_MAJOR
#include <stdexcept>
#include <string>

namespace {

#if CV_VERSION_MAJOR >= 5
TensorDataType input_datatype(int type) {
    switch (type) {
    case CV_32F:
        return TensorDataType::Float32;
    case CV_32S:
        return TensorDataType::Int32;
    case CV_8U:
        return TensorDataType::UInt8;
    case CV_64S:
        return TensorDataType::Int64;
    default:
        throw ModelLoadException("OpenCV DNN graph has an unsupported input datatype");
    }
}
#endif

int mat_type(TensorDataType type) {
    switch (type) {
    case TensorDataType::Float32:
        return CV_32F;
    case TensorDataType::Int32:
        return CV_32S;
    case TensorDataType::UInt8:
        return CV_8U;
#if CV_VERSION_MAJOR >= 5
    case TensorDataType::Int64:
        return CV_64S;
#endif
    default:
        throw InferenceExecutionException("OpenCV DNN input datatype is unsupported");
    }
}

// The Darknet importer is the one piece of the OpenCV DNN API this backend
// uses that 5.x removed -- along with the Caffe and Torch importers, leaving
// readNet() with *.pb, *.onnx and OpenVINO *.bin/*.xml. Everything else here is
// source-compatible across 4.6 and 5.x: readNet()'s added `engine` parameter is
// defaulted, and the Backend/Target enums, getUnconnectedOutLayers() and
// MatSize all kept their signatures.
cv::dnn::Net load_net(const std::string& model_path) {
    const size_t weights_pos = model_path.find(".weights");
    if (weights_pos == std::string::npos) {
#if CV_VERSION_MAJOR >= 5
        return cv::dnn::readNet(model_path, "", "", cv::dnn::ENGINE_AUTO);
#else
        return cv::dnn::readNet(model_path);
#endif
    }

#if CV_VERSION_MAJOR >= 5
    // readNet() would take a .weights file and fail with "can't load the
    // model", which points at the file rather than at the missing importer.
    throw std::runtime_error("OpenCV " CV_VERSION " has no Darknet importer, so " + model_path +
                             " cannot be loaded. Convert the model to ONNX, or build this backend "
                             "against OpenCV 4.x.");
#else
    const std::string config = model_path.substr(0, weights_pos) + ".cfg";
    if (!std::ifstream(config)) {
        throw std::runtime_error("Can't find the configuration file " + config + " for the model: " + model_path);
    }
    return cv::dnn::readNetFromDarknet(config, model_path);
#endif
}

} // namespace

OCVDNNInfer::OCVDNNInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                         const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
    LOG(INFO) << "Running using OpenCV DNN runtime: " << model_path;
#if CV_VERSION_MAJOR >= 5
    // readNet() defaults to ENGINE_AUTO, which resolves to the rewritten
    // ENGINE_OPENCV, and that engine does not support non-CPU backends yet.
    // setPreferableBackend(DNN_BACKEND_CUDA) below would therefore be accepted
    // and then ignored, so a caller that asked for the GPU would silently get
    // CPU inference and only notice as a latency regression. Refuse instead:
    // use_gpu=false asks for CPU explicitly, and OpenCV 4.x still honours CUDA.
    // Revisit once an OpenCV 5 engine supports non-CPU backends.
    if (use_gpu) {
        throw std::runtime_error("OpenCV " CV_VERSION " cannot run this backend on the GPU: its default DNN engine "
                                 "supports CPU only, so a CUDA request would silently run on the CPU. "
                                 "Pass use_gpu=false, or build this backend against OpenCV 4.x.");
    }
#endif

    net_ = load_net(model_path);
    if (net_.empty()) {
        throw std::runtime_error("Can't load the model: " + model_path);
    }

    // Both enums survive into 5.x, so this compiles either way.
    if (use_gpu && isCudaBuildEnabled()) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    outLayers_ = net_.getUnconnectedOutLayers();
    outLayerType_ = net_.getLayer(outLayers_[0])->type;
    outNames_ = net_.getUnconnectedOutLayersNames();

#if CV_VERSION_MAJOR >= 5
    if (const auto graph = net_.getMainGraph(); graph) {
        native_graph_inputs_ = true;
        const auto& inputs = graph->inputs();
        if (!input_sizes.empty() && input_sizes.size() != inputs.size()) {
            throw ModelLoadException("OpenCV DNN input-size count must match the graph input count");
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& data = net_.argData(inputs[i]);
            std::vector<int64_t> shape(data.shape.begin(), data.shape.end());
            if (!input_sizes.empty()) {
                auto requested = input_sizes[i];
                if (requested.size() + 1 == shape.size()) {
                    requested.insert(requested.begin(), static_cast<int64_t>(batch_size));
                }
                if (requested.size() != shape.size()) {
                    throw ModelLoadException("OpenCV DNN input-size rank does not match graph input " + data.name);
                }
                for (size_t d = 0; d < shape.size(); ++d) {
                    if (requested[d] <= 0 || (shape[d] > 0 && shape[d] != requested[d])) {
                        throw ModelLoadException("OpenCV DNN input size does not match graph input " + data.name);
                    }
                }
                shape = std::move(requested);
            } else if (!shape.empty() && shape[0] <= 0) {
                shape[0] = static_cast<int64_t>(batch_size);
            }
            if (shape.empty()) {
                throw ModelLoadException("OpenCV DNN scalar input is unsupported: " + data.name);
            }
            for (const auto dimension : shape) {
                if (dimension <= 0) {
                    throw ModelLoadException("OpenCV DNN dynamic input requires explicit input sizes: " + data.name);
                }
            }
            inference_metadata_.addInput(data.name, shape, batch_size, input_datatype(data.type));
        }
    }
#endif
    if (!native_graph_inputs_) {
        if (input_sizes.empty()) {
            throw std::runtime_error("With OpenCV DNN backend, input sizes must be specified");
        }
        for (size_t i = 0; i < input_sizes.size(); ++i) {
            inference_metadata_.addInput("input" + std::to_string(i + 1), input_sizes[i], batch_size);
        }
    }

    for (auto& outName : outNames_) {
        std::vector<int64_t> shape{-1, -1, -1};
        TensorDataType datatype = TensorDataType::Float32;
#if CV_VERSION_MAJOR >= 5
        if (native_graph_inputs_ && net_.haveArg(outName)) {
            const auto& data = net_.argData(net_.getArg(outName));
            shape.assign(data.shape.begin(), data.shape.end());
            datatype = data.type == CV_64F ? TensorDataType::Float32 : input_datatype(data.type);
        }
#endif
        inference_metadata_.addOutput(outName, shape, batch_size, datatype);
    }

    state_ = BackendState::Ready;
}

std::vector<cv::Mat> OCVDNNInfer::run_forward(const std::vector<std::vector<uint8_t>>& input_tensors) {

    const auto& metadata = inference_metadata_.getInputs();
    if (input_tensors.size() != metadata.size() || (!native_graph_inputs_ && input_tensors.size() != 1)) {
        throw InferenceExecutionException("OpenCV DNN input count does not match the model");
    }
    // Validate every buffer before changing the network's input state.
    std::vector<cv::Mat> blobs;
    blobs.reserve(metadata.size());
    for (size_t i = 0; i < metadata.size(); ++i) {
        std::vector<int64_t> shape = metadata[i].shape;
        if (!native_graph_inputs_) {
            const size_t batch = get_batch_size();
            if (batch == 0 || batch > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw InferenceExecutionException("OpenCV DNN input batch is invalid");
            }
            shape.insert(shape.begin(), static_cast<int64_t>(batch));
        }
        const int type = mat_type(metadata[i].datatype);
        const size_t element_bytes = CV_ELEM_SIZE(type);
        size_t expected_bytes = element_bytes;
        std::vector<int> mat_size;
        for (const auto dimension : shape) {
            if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) {
                throw InferenceExecutionException("OpenCV DNN input dimensions must be positive and fit in int");
            }
            const size_t size = static_cast<size_t>(dimension);
            if (expected_bytes > std::numeric_limits<size_t>::max() / size) {
                throw InferenceExecutionException("OpenCV DNN input byte size overflows size_t");
            }
            expected_bytes *= size;
            mat_size.push_back(static_cast<int>(dimension));
        }
        if (input_tensors[i].size() != expected_bytes) {
            throw InferenceExecutionException("OpenCV DNN input byte size mismatch: expected " +
                                              std::to_string(expected_bytes) + ", got " +
                                              std::to_string(input_tensors[i].size()));
        }
        blobs.emplace_back(static_cast<int>(mat_size.size()), mat_size.data(), type,
                           const_cast<uint8_t*>(input_tensors[i].data()));
    }
    for (size_t i = 0; i < blobs.size(); ++i) {
        net_.setInput(blobs[i], native_graph_inputs_ ? metadata[i].name : "");
    }
    std::vector<cv::Mat> outs;
    net_.forward(outs, outNames_);
    return outs;
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
OCVDNNInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {

    const std::vector<cv::Mat> outs = run_forward(input_tensors);

    std::vector<std::vector<TensorElement>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    outputs.reserve(outs.size());
    shapes.reserve(outs.size());

    for (const auto& output : outs) {
        // Extracting dimensions of the output tensor
        std::vector<int64_t> shape;
        shape.reserve(output.dims);
        for (int j = 0; j < output.dims; ++j) {
            shape.push_back(output.size[j]);
        }
        shapes.push_back(std::move(shape));

        // Extracting data
        std::vector<TensorElement> tensor_data;
        tensor_data.reserve(output.total());

        if (output.type() == CV_32F) {
            const float* data = output.ptr<float>();
            for (int j = 0; j < output.total(); ++j) {
                tensor_data.push_back(data[j]);
            }
        } else if (output.type() == CV_64F) {
            const double* data = output.ptr<double>();
            for (int j = 0; j < output.total(); ++j) {
                tensor_data.push_back(static_cast<float>(data[j]));
            }
        } else if (output.type() == CV_32S) {
            const auto* data = output.ptr<int32_t>();
            for (size_t j = 0; j < output.total(); ++j)
                tensor_data.emplace_back(data[j]);
#if CV_VERSION_MAJOR >= 5
        } else if (output.type() == CV_64S) {
            const auto* data = output.ptr<int64_t>();
            for (size_t j = 0; j < output.total(); ++j)
                tensor_data.emplace_back(data[j]);
#endif
        } else {
            throw std::runtime_error("Unsupported data type in OCVDNNInfer::get_infer_results");
        }

        outputs.push_back(std::move(tensor_data));
    }

    return std::make_tuple(outputs, shapes);
}

std::vector<RawOutputTensor>
OCVDNNInfer::get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) {

    const std::vector<cv::Mat> outs = run_forward(input_tensors);

    std::vector<RawOutputTensor> raw_outputs;
    raw_outputs.reserve(outs.size());

    for (const auto& output : outs) {
        // forward() allocates fresh, continuous Mats; clone defensively if not.
        const cv::Mat contiguous = output.isContinuous() ? output : output.clone();

        RawOutputTensor raw;
        raw.shape.reserve(contiguous.dims);
        for (int j = 0; j < contiguous.dims; ++j) {
            raw.shape.push_back(contiguous.size[j]);
        }

        raw.dtype = TensorDtype::FP32;
        const size_t num_elements = contiguous.total();
        raw.bytes.resize(num_elements * sizeof(float));
        auto* typed = reinterpret_cast<float*>(raw.bytes.data());

        if (contiguous.type() == CV_32F) {
            std::memcpy(raw.bytes.data(), contiguous.ptr<float>(), raw.bytes.size());
        } else if (contiguous.type() == CV_64F) {
            const double* data = contiguous.ptr<double>();
            for (size_t j = 0; j < num_elements; ++j) {
                typed[j] = static_cast<float>(data[j]);
            }
        } else if (contiguous.type() == CV_32S) {
            raw.dtype = TensorDtype::INT32;
            raw.bytes.resize(num_elements * sizeof(int32_t));
            std::memcpy(raw.bytes.data(), contiguous.data, raw.bytes.size());
#if CV_VERSION_MAJOR >= 5
        } else if (contiguous.type() == CV_64S) {
            raw.dtype = TensorDtype::INT64;
            raw.bytes.resize(num_elements * sizeof(int64_t));
            std::memcpy(raw.bytes.data(), contiguous.data, raw.bytes.size());
#endif
        } else {
            throw std::runtime_error("Unsupported data type in OCVDNNInfer::get_infer_results_raw");
        }

        raw_outputs.push_back(std::move(raw));
    }

    return raw_outputs;
}
