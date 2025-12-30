#include "ORTInfer.hpp"
#include <algorithm>
#include <numeric>

ORTInfer::ORTInfer(const std::string &model_path, bool use_gpu,
                   size_t batch_size,
                   const std::vector<std::vector<int64_t>> &input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Onnx Runtime Inference");
  Ort::SessionOptions session_options;

  if (use_gpu) {
    std::vector<std::string> providers = Ort::GetAvailableProviders();
    LOG(INFO) << "Available providers:";
    bool is_found = false;
    for (const auto &p : providers) {
      LOG(INFO) << p;
      if (p.find("CUDA") != std::string::npos) {
        LOG(INFO) << "Using CUDA GPU";
        OrtCUDAProviderOptions cuda_options;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        is_found = true;
        break;
      }
    }
    if (!is_found) {
      LOG(INFO) << "CUDA GPU not available, falling back to CPU";
      session_options = Ort::SessionOptions();
    }
  } else {
    LOG(INFO) << "Using CPU";
    session_options = Ort::SessionOptions();
  }

  try {
    session_ = Ort::Session(env_, model_path.c_str(), session_options);
  } catch (const Ort::Exception &ex) {
    LOG(ERROR) << "Failed to load the ONNX model: " << ex.what();
    std::exit(1);
  }

  Ort::AllocatorWithDefaultOptions allocator;
  LOG(INFO) << "Input Node Name/Shape (" << session_.GetInputCount() << "):";

  // Process inputs
  for (std::size_t i = 0; i < session_.GetInputCount(); i++) {
    const std::string name = session_.GetInputNameAllocated(i, allocator).get();
    auto shapes =
        session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    auto input_type = session_.GetInputTypeInfo(i)
                          .GetTensorTypeAndShapeInfo()
                          .GetElementType();

    // Check if this input has dynamic dimensions
    bool has_dynamic = false;
    for (size_t j = 1; j < shapes.size(); j++) { // Skip batch dimension
      if (shapes[j] == -1) {
        has_dynamic = true;
        break;
      }
    }

    // Handle batch dimension first
    shapes[0] = shapes[0] == -1 ? batch_size : shapes[0];

    // Handle dimensions if dynamic or if input_sizes provided
    if (has_dynamic || (!input_sizes.empty() && i < input_sizes.size())) {
      if (input_sizes.empty() || i >= input_sizes.size()) {
        throw std::runtime_error(
            "Dynamic shapes found but no input sizes provided for input '" +
            name + "'");
      }

      const auto &provided_shape = input_sizes[i];

      if (has_dynamic) {
        // Check if provided shape has enough dimensions for dynamic inputs
        size_t dynamic_dim_count = 0;
        for (size_t j = 1; j < shapes.size(); j++) {
          if (shapes[j] == -1)
            dynamic_dim_count++;
        }

        if (provided_shape.size() < dynamic_dim_count) {
          throw std::runtime_error(
              "Not enough dimensions provided for dynamic shapes in input '" +
              name + "'");
        }

        // Apply provided dimensions to dynamic shapes
        size_t provided_idx = 0;
        for (size_t j = 1; j < shapes.size(); j++) {
          if (shapes[j] == -1) {
            if (provided_idx >= provided_shape.size()) {
              throw std::runtime_error("Insufficient input sizes provided for "
                                       "dynamic dimensions in input '" +
                                       name + "'");
            }
            shapes[j] = provided_shape[provided_idx++];
          }
        }
      } else {
        // Override fixed dimensions with provided dimensions (skip batch dimension)
        if (provided_shape.size() != shapes.size() - 1) {
          throw std::runtime_error(
              "Provided shape size mismatch for input '" + name + 
              "'. Expected " + std::to_string(shapes.size() - 1) + 
              " dimensions, got " + std::to_string(provided_shape.size()));
        }
        
        for (size_t j = 1; j < shapes.size(); j++) {
          shapes[j] = provided_shape[j - 1];
        }
      }
    }

    LOG(INFO) << "\t" << name << " : " << print_shape(shapes);
    inference_metadata_.addInput(name, shapes, batch_size);

    std::string input_type_str = getDataTypeString(input_type);
    LOG(INFO) << "\tData Type: " << input_type_str;
  }

  // Log network dimensions from first input
  const auto &first_input = inference_metadata_.getInputs()[0].shape;
  const auto channels = static_cast<int>(first_input[1]);
  const auto network_height = static_cast<int>(first_input[2]);
  const auto network_width = static_cast<int>(first_input[3]);

  LOG(INFO) << "channels " << channels;
  LOG(INFO) << "width " << network_width;
  LOG(INFO) << "height " << network_height;

  // Process outputs
  LOG(INFO) << "Output Node Name/Shape (" << session_.GetOutputCount() << "):";
  for (std::size_t i = 0; i < session_.GetOutputCount(); i++) {
    const std::string name =
        session_.GetOutputNameAllocated(i, allocator).get();
    auto shapes =
        session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    shapes[0] = shapes[0] == -1 ? batch_size : shapes[0];
    LOG(INFO) << "\t" << name << " : " << print_shape(shapes);
    inference_metadata_.addOutput(name, shapes, batch_size);
  }
}

std::string ORTInfer::getDataTypeString(ONNXTensorElementDataType type) {
  switch (type) {
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "Float";
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return "Int64";
  default:
    return "Unknown";
  }
}

std::string ORTInfer::print_shape(const std::vector<std::int64_t> &v) {
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

size_t ORTInfer::getSizeByDim(const std::vector<int64_t> &dims) {
  size_t size = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] == -1 || dims[i] == 0) {
      continue;
    }
    size *= dims[i];
  }
  return size;
}

std::tuple<std::vector<std::vector<TensorElement>>,
           std::vector<std::vector<int64_t>>>
ORTInfer::get_infer_results(const std::vector<cv::Mat> &input_blobs) {
  validate_input(input_blobs);
  
  // Process multiple input tensors
  std::vector<cv::Mat> processed_blobs;
  for (const auto& input_blob : input_blobs) {
    cv::Mat blob;
    if (input_blob.dims > 2) {
      blob = input_blob;
    } else {
      cv::dnn::blobFromImage(input_blob, blob, 1.0, cv::Size(),
                             cv::Scalar(), false, false);
    }
    processed_blobs.push_back(blob);
  }
  
  const auto &inputs = inference_metadata_.getInputs();
  const auto &outputs = inference_metadata_.getOutputs();

  std::vector<std::vector<TensorElement>> output_tensors;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::vector<float>> input_tensor_data(std::min(processed_blobs.size(), static_cast<size_t>(session_.GetInputCount())));
  std::vector<Ort::Value> in_ort_tensors;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<int64_t> orig_target_sizes;

  // Process user-provided input tensors
  size_t num_user_inputs = std::min(processed_blobs.size(), static_cast<size_t>(session_.GetInputCount()));
  for (size_t i = 0; i < num_user_inputs; ++i) {
    input_tensor_data[i] = blob2vec(processed_blobs[i]);
    in_ort_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_data[i].data(), getSizeByDim(inputs[i].shape),
        inputs[i].shape.data(), inputs[i].shape.size()));
  }

  // Handle models that need additional computed inputs (e.g., RTDETR orig_target_sizes)
  if (session_.GetInputCount() > num_user_inputs) {
    orig_target_sizes = {static_cast<int64_t>(processed_blobs[0].size[2]),
                         static_cast<int64_t>(processed_blobs[0].size[3])};
    in_ort_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, orig_target_sizes.data(), orig_target_sizes.size(),
        inputs[num_user_inputs].shape.data(), inputs[num_user_inputs].shape.size()));
  }

  // Run inference
  std::vector<const char *> input_names_char(inputs.size());
  std::transform(inputs.begin(), inputs.end(), input_names_char.begin(),
                 [](const LayerInfo &layer) { return layer.name.c_str(); });

  std::vector<const char *> output_names_char(outputs.size());
  std::transform(outputs.begin(), outputs.end(), output_names_char.begin(),
                 [](const LayerInfo &layer) { return layer.name.c_str(); });

  std::vector<Ort::Value> output_ort_tensors = session_.Run(
      Ort::RunOptions{nullptr}, input_names_char.data(), in_ort_tensors.data(),
      in_ort_tensors.size(), output_names_char.data(), outputs.size());

  // Process output tensors
  assert(output_ort_tensors.size() == outputs.size());

  for (const Ort::Value &output_tensor : output_ort_tensors) {
    const auto &shape_ref =
        output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> shape(shape_ref.begin(), shape_ref.end());

    size_t num_elements = 1;
    for (int64_t dim : shape) {
      num_elements *= dim;
    }

    std::vector<TensorElement> tensor_data;
    tensor_data.reserve(num_elements);

    // Retrieve tensor data
    const int onnx_type =
        output_tensor.GetTensorTypeAndShapeInfo().GetElementType();
    switch (onnx_type) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      const float *output_data_float = output_tensor.GetTensorData<float>();
      for (size_t i = 0; i < num_elements; ++i) {
        tensor_data.emplace_back(output_data_float[i]);
      }
      break;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      const int64_t *output_data_int64 = output_tensor.GetTensorData<int64_t>();
      for (size_t i = 0; i < num_elements; ++i) {
        tensor_data.emplace_back(output_data_int64[i]);
      }
      break;
    }
    default:
      LOG(ERROR) << "Unsupported tensor type: " << onnx_type;
      std::exit(1);
    }

    output_tensors.emplace_back(std::move(tensor_data));
    shapes.emplace_back(shape);
  }

  return std::make_tuple(output_tensors, shapes);
}