#include "OCVDNNInfer.hpp"

#include <cstring>
#include <fstream>
#include <opencv2/core/version.hpp> // CV_VERSION, CV_VERSION_MAJOR
#include <stdexcept>
#include <string>

namespace {

// The Darknet importer is the one piece of the OpenCV DNN API this backend
// uses that 5.x removed -- along with the Caffe and Torch importers, leaving
// readNet() with *.pb, *.onnx and OpenVINO *.bin/*.xml. Everything else here is
// source-compatible across 4.6 and 5.x: readNet()'s added `engine` parameter is
// defaulted, and the Backend/Target enums, getUnconnectedOutLayers() and
// MatSize all kept their signatures.
cv::dnn::Net load_net(const std::string& model_path) {
    const size_t weights_pos = model_path.find(".weights");
    if (weights_pos == std::string::npos) {
        return cv::dnn::readNet(model_path);
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
    net_ = load_net(model_path);
    if (net_.empty()) {
        throw std::runtime_error("Can't load the model: " + model_path);
    }

    // Both enums survive into 5.x, so this compiles either way. On 5.x it may
    // still fall back to CPU at runtime: readNet() defaults to ENGINE_AUTO,
    // which resolves to the rewritten ENGINE_OPENCV, and that engine does not
    // support non-CPU backends yet.
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

    if (input_sizes.empty()) {
        throw std::runtime_error("With OpenCV DNN backend, input sizes must be specified");
    }

    for (size_t i = 0; i < input_sizes.size(); i++) {
        std::vector<int64_t> shape = input_sizes[i];
        inference_metadata_.addInput("input" + std::to_string(i + 1), shape, batch_size);
    }

    for (auto& outName : outNames_) {
        std::vector<int64_t> shape{-1, -1, -1};
        inference_metadata_.addOutput(outName, shape, batch_size);
    }

    state_ = BackendState::Ready;
}

std::vector<cv::Mat> OCVDNNInfer::run_forward(const std::vector<std::vector<uint8_t>>& input_tensors) {

    // OpenCV DNN backend currently supports only single input models
    if (input_tensors.size() != 1) {
        throw std::runtime_error("OpenCV DNN backend currently supports only single input models, got " +
                                 std::to_string(input_tensors.size()) + " inputs");
    }

    const std::vector<uint8_t>& input_data = input_tensors[0];

    // Reconstruct cv::Mat from raw bytes
    // We assume the input is already a preprocessed blob (NCHW or similar) matching the model input
    const auto& shape_meta = inference_metadata_.getInputs()[0].shape;
    std::vector<int> mat_size;
    mat_size.push_back(static_cast<int>(get_batch_size())); // batch dimension (N in NCHW)
    for (auto s : shape_meta)
        mat_size.push_back(static_cast<int>(s));

    // validate size
    size_t expected_elements = 1;
    for (auto s : shape_meta)
        expected_elements *= s;
    if (input_data.size() != expected_elements * sizeof(float)) {
        // Fallback or warning?
        // OpenCV DNN usually works with Float32
        // If size mismatches, it might be uint8 image?
        // If we strictly follow "get_infer_results takes processed tensors", it should be float.
        // But if user passes an image, we can't easily handle blobFromImage without parameters (mean, scale).
        // We assume it's the blob.
        if (input_data.size() == expected_elements) {
            // Maybe it's uint8?
        }
    }

    cv::Mat blob(mat_size.size(), mat_size.data(), CV_32F, const_cast<uint8_t*>(input_data.data()));

    std::vector<cv::Mat> outs;
    net_.setInput(blob);
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
        } else {
            throw std::runtime_error("Unsupported data type in OCVDNNInfer::get_infer_results_raw");
        }

        raw_outputs.push_back(std::move(raw));
    }

    return raw_outputs;
}
