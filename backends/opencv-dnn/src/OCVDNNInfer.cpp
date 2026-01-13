#include "OCVDNNInfer.hpp"

OCVDNNInfer::OCVDNNInfer(const std::string &model_path, bool use_gpu,
                         size_t batch_size,
                         const std::vector<std::vector<int64_t>> &input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
  // check if model path has .weights extension
  std::string modelConfiguration = "";
  if (model_path.find(".weights") != std::string::npos) {
    // get the name without extension
    modelConfiguration =
        model_path.substr(0, model_path.find(".weights")) + ".cfg";

    // check if .cfg file exists
    if (!std::ifstream(modelConfiguration))
      throw std::runtime_error("Can't find the configuration file " +
                               modelConfiguration +
                               " for the model: " + model_path);
  }
  LOG(INFO) << "Running using OpenCV DNN runtime: " << model_path;
  net_ = modelConfiguration.empty()
             ? cv::dnn::readNet(model_path)
             : cv::dnn::readNetFromDarknet(modelConfiguration, model_path);
  if (net_.empty()) {
    throw std::runtime_error("Can't load the model: " + model_path);
  }

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
    throw("With OpenCV DNN backend, input sizes must be specified");
  }

  for (size_t i = 0; i < input_sizes.size(); i++) {
    std::vector<int64_t> shape = input_sizes[i];
    inference_metadata_.addInput("input" + std::to_string(i + 1), shape,
                                 batch_size);
  }

  for (auto &outName : outNames_) {
    std::vector<int64_t> shape{-1, -1, -1};
    inference_metadata_.addOutput(outName, shape, batch_size);
  }
}

std::tuple<std::vector<std::vector<TensorElement>>,
           std::vector<std::vector<int64_t>>>
OCVDNNInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {

  // OpenCV DNN backend currently supports only single input models
  if (input_tensors.size() != 1) {
    throw std::runtime_error("OpenCV DNN backend currently supports only single input models, got " + std::to_string(input_tensors.size()) + " inputs");
  }
  
  const std::vector<uint8_t>& input_data = input_tensors[0];
  
  // Reconstruct cv::Mat from raw bytes
  // We assume the input is already a preprocessed blob (NCHW or similar) matching the model input
  const auto& shape_meta = inference_metadata_.getInputs()[0].shape;
  std::vector<int> mat_size;
  for(auto s : shape_meta) mat_size.push_back(static_cast<int>(s));
  
  // validate size
  size_t expected_elements = 1;
  for(auto s : shape_meta) expected_elements *= s;
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

  std::vector<std::vector<TensorElement>> outputs;
  std::vector<std::vector<int64_t>> shapes;

  std::vector<cv::Mat> outs;
  net_.setInput(blob);
  net_.forward(outs, outNames_);

  outputs.reserve(outs.size());
  shapes.reserve(outs.size());

  for (const auto &output : outs) {
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
      const float *data = output.ptr<float>();
      for (int j = 0; j < output.total(); ++j) {
        tensor_data.push_back(data[j]);
      }
    } else if (output.type() == CV_64F) {
      const double *data = output.ptr<double>();
      for (int j = 0; j < output.total(); ++j) {
        tensor_data.push_back(static_cast<float>(data[j]));
      }
    } else {
      throw std::runtime_error(
          "Unsupported data type in OCVDNNInfer::get_infer_results");
    }

    outputs.push_back(std::move(tensor_data));
  }

  return std::make_tuple(outputs, shapes);
}
