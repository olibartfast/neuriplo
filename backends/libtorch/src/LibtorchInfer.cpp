#include "LibtorchInfer.hpp"
#include <sstream>

std::string LibtorchInfer::print_shape(const std::vector<int64_t> &shape) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    ss << shape[i];
    if (i < shape.size() - 1) {
      ss << ", ";
    }
  }
  ss << ")";
  return ss.str();
}

LibtorchInfer::LibtorchInfer(
    const std::string &model_path, bool use_gpu, size_t batch_size,
    const std::vector<std::vector<int64_t>> &input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
  if (use_gpu && torch::cuda::is_available()) {
    device_ = torch::kCUDA;
    LOG(INFO) << "Using CUDA GPU";
  } else {
    device_ = torch::kCPU;
    LOG(INFO) << "Using CPU";
  }

  try {
    module_ = torch::jit::load(model_path, device_);
  } catch (const c10::Error &e) {
    LOG(ERROR) << "Failed to load the LibTorch model: " << e.what();
    std::exit(1);
  }

  // Process inputs
  LOG(INFO) << "Input Node Name/Shape:";
  auto method = module_.get_method("forward");
  auto graph = method.graph();
  auto inputs = graph->inputs();

  // Skip the first input as it's usually the self/module input
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto input = inputs[i];
    std::string name = input->debugName();
    auto type = input->type()->cast<c10::TensorType>();
    if (!type) {
      LOG(WARNING) << "Input " << name << " is not a tensor. Skipping.";
      continue;
    }

    auto shapes = type->sizes().concrete_sizes();
    bool has_dynamic = !shapes.has_value();
    
    if (has_dynamic || (!input_sizes.empty() && (i - 1) < input_sizes.size())) {
      if (input_sizes.empty() || (i - 1) >= input_sizes.size()) {
        throw std::runtime_error(
            "LibtorchInfer Initialitazion Error: Dynamic shapes found but no "
            "input sizes provided for input '" +
            name + "'");
      }
      
      if (has_dynamic) {
        // Use provided dimensions for dynamic shapes
        shapes = input_sizes[i - 1];
      } else {
        // Override fixed dimensions with provided dimensions
        auto fixed_shapes = *shapes;
        const auto& provided_shape = input_sizes[i - 1];
        
        if (provided_shape.size() != fixed_shapes.size() - 1) {
          throw std::runtime_error(
              "Provided shape size mismatch for input '" + name + 
              "'. Expected " + std::to_string(fixed_shapes.size() - 1) + 
              " dimensions, got " + std::to_string(provided_shape.size()));
        }
        
        for (size_t j = 1; j < fixed_shapes.size(); ++j) {
          fixed_shapes[j] = provided_shape[j - 1];
        }
        shapes = fixed_shapes;
      }
    }

    std::vector<int64_t> final_shape = *shapes;

    LOG(INFO) << "\t" << name << " : " << print_shape(final_shape);
    inference_metadata_.addInput(name, final_shape, batch_size);

    std::string input_type_str = "Unknown";
    c10::ScalarType s_type = c10::ScalarType::Float; // default
    if (type->scalarType().has_value()) {
        s_type = type->scalarType().value();
        input_type_str = toString(s_type);
    }
    input_types_.push_back(s_type);
    LOG(INFO) << "\tData Type: " << input_type_str;
  }

  // Log network dimensions from first input
  const auto &first_input = inference_metadata_.getInputs()[0].shape;
  for (size_t i = 0; i < first_input.size(); ++i) {
    LOG(INFO) << "Network Dimension " << i << ": " << first_input[i];
  }

  // Process outputs
  LOG(INFO) << "Output Node Name/Shape:";
  auto outputs = graph->outputs();

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto output = outputs[i];
    std::string name = output->debugName();
    auto output_type = output->type();

    LOG(INFO) << "Inspecting output " << i << ": " << name;
    LOG(INFO) << "  Type: " << output_type->str();

    // Try all possible casts with detailed logging
    auto tensor_type = output_type->cast<c10::TensorType>();
    auto tuple_type = output_type->cast<c10::TupleType>();
    auto list_type = output_type->cast<c10::ListType>();

    LOG(INFO) << "  Cast results:";
    LOG(INFO) << "    TensorType: " << (tensor_type ? "YES" : "NO");
    LOG(INFO) << "    TupleType: " << (tuple_type ? "YES" : "NO");
    LOG(INFO) << "    ListType: " << (list_type ? "YES" : "NO");

    // Check if output is a Tuple
    if (tuple_type) {
      LOG(INFO) << "\tDetected Tuple output with "
                << tuple_type->elements().size() << " elements";

      // Process each element in the tuple - generic handling for all tuple
      // sizes
      auto elements = tuple_type->elements();

      for (size_t j = 0; j < elements.size(); ++j) {
        auto elem_type = elements[j]->cast<c10::TensorType>();
        if (!elem_type) {
          LOG(WARNING) << "\tTuple element " << j
                       << " is not a tensor. Skipping.";
          continue;
        }

        std::string elem_name = name + "_elem_" + std::to_string(j);
        auto shapes = elem_type->sizes().concrete_sizes();

        std::vector<int64_t> final_shape;
        if (!shapes || shapes->empty() ||
            std::any_of(shapes->begin(), shapes->end(),
                        [](int64_t s) { return s <= 0; })) {
          LOG(WARNING) << "\tTuple element " << j << " has dynamic shape";
          auto ndim = elem_type->dim();
          if (ndim.has_value()) {
            final_shape = std::vector<int64_t>(ndim.value(), -1);
            if (!final_shape.empty()) {
              final_shape[0] = batch_size;
            }
          } else {
            // Unknown dimensions - use minimal placeholder
            final_shape = {static_cast<int64_t>(batch_size), -1};
          }
        } else {
          final_shape = *shapes;
          if (!final_shape.empty()) {
            final_shape[0] = batch_size;
          }
        }

        LOG(INFO) << "\t" << elem_name << " : " << print_shape(final_shape);
        inference_metadata_.addOutput(elem_name, final_shape, batch_size);
      }
    }
    // Check if output is a List
    else if (auto list_type = output_type->cast<c10::ListType>()) {
      LOG(INFO) << "\tDetected List output";

      // Get the element type of the list
      auto elem_type = list_type->getElementType();
      LOG(INFO) << "\tList element type: " << elem_type->str();

      // Check if element is a tensor type
      if (auto tensor_elem = elem_type->cast<c10::TensorType>()) {
        LOG(INFO) << "\tList contains tensors";
        LOG(INFO) << "\tList outputs will be processed dynamically at runtime";
        LOG(INFO) << "\tNote: List size and shapes are determined during first "
                     "inference";

        // Mark that we have a list output - actual shapes determined at runtime
        // We'll add a placeholder to indicate list output exists
        std::string list_marker = name + "_list";
        inference_metadata_.addOutput(list_marker, {-1}, batch_size);
      } else {
        LOG(WARNING) << "\tList element type is not a tensor. Skipping.";
      }
    }
    // Check if output is a Tensor
    else if (auto tensor_type = output_type->cast<c10::TensorType>()) {
      auto shapes = tensor_type->sizes().concrete_sizes();
      if (!shapes || std::any_of(shapes->begin(), shapes->end(),
                                 [](int64_t s) { return s <= 0; })) {
        LOG(WARNING) << "Output " << name
                     << " has dynamic shape. Using (-1) as placeholder.";
        auto ndim = tensor_type->dim();
        shapes = std::vector<int64_t>(ndim.value_or(1), -1);
      }

      std::vector<int64_t> final_shape = *shapes;
      if (!final_shape.empty()) {
        final_shape[0] = batch_size; // Set batch size
      }

      LOG(INFO) << "\t" << name << " : " << print_shape(final_shape);
      inference_metadata_.addOutput(name, final_shape, batch_size);
    } else {
      LOG(WARNING) << "Output " << name
                   << " is neither a tensor, tuple, nor list. Skipping.";
      LOG(WARNING) << "  Type string: " << output_type->str();
      continue;
    }
  }
}

std::tuple<std::vector<std::vector<TensorElement>>,
           std::vector<std::vector<int64_t>>>
LibtorchInfer::get_infer_results(const std::vector<std::vector<uint8_t>> &input_tensors) {

  // Convert input images to torch tensors
  std::vector<torch::jit::IValue> torch_inputs;
  const auto &inputs_meta = inference_metadata_.getInputs();

  // Process user-provided input tensors
  if (input_tensors.size() != inputs_meta.size()) {
       throw std::runtime_error("Input tensor count mismatch. Expected " + std::to_string(inputs_meta.size()) + ", got " + std::to_string(input_tensors.size()));
  }
  
  for (size_t i = 0; i < inputs_meta.size(); ++i) {
    const auto &input_data = input_tensors[i];
    const auto &shape = inputs_meta[i].shape;
    
    // Determine type from stored metadata
    c10::ScalarType dtype = c10::ScalarType::Float;
    if (i < input_types_.size()) {
        dtype = input_types_[i];
    }
    
    // Validate size
    size_t element_size = c10::elementSize(dtype);
    if (input_data.size() % element_size != 0) {
        throw std::runtime_error("Input buffer size not multiple of element size for input " + std::to_string(i));
    }
    
    std::vector<int64_t> tensor_dims = shape;

    auto options = torch::TensorOptions().dtype(dtype);
    torch::Tensor input = torch::from_blob(
        const_cast<uint8_t*>(input_data.data()), 
        tensor_dims,
        options);
        
    input = input.to(device_);
    torch_inputs.push_back(input);
  }

  // Run inference
  auto output = module_.forward(torch_inputs);

  std::vector<std::vector<TensorElement>> output_vectors;
  std::vector<std::vector<int64_t>> shape_vectors;

  // Helper function to process a single tensor
  auto process_tensor = [](const torch::Tensor &tensor) {
    std::vector<TensorElement> tensor_data;
    tensor_data.reserve(tensor.numel());

    const auto data_type = tensor.scalar_type();
    switch (data_type) {
    case torch::kFloat32: {
      const float *output_data = tensor.data_ptr<float>();
      for (size_t i = 0; i < tensor.numel(); ++i) {
        tensor_data.emplace_back(output_data[i]);
      }
      break;
    }
    case torch::kInt64: {
      const int64_t *output_data = tensor.data_ptr<int64_t>();
      for (size_t i = 0; i < tensor.numel(); ++i) {
        tensor_data.emplace_back(output_data[i]);
      }
      break;
    }
    default:
      LOG(ERROR) << "Unsupported tensor type: " << data_type;
      std::exit(1);
    }
    return tensor_data;
  };

  if (output.isTuple()) {
    // Handle tuple output
    auto tuple_outputs = output.toTuple()->elements();
    for (const auto &output_tensor : tuple_outputs) {
      if (!output_tensor.isTensor()) {
        continue;
      }

      torch::Tensor tensor =
          output_tensor.toTensor().to(torch::kCPU).contiguous();

      output_vectors.push_back(process_tensor(tensor));
      shape_vectors.push_back(tensor.sizes().vec());
    }
  } else if (output.isList()) {
    // Handle list output (new!)
    auto list_outputs = output.toList();
    for (size_t i = 0; i < list_outputs.size(); ++i) {
      auto element = list_outputs.get(i);
      if (!element.isTensor()) {
        continue;
      }

      torch::Tensor tensor = element.toTensor().to(torch::kCPU).contiguous();

      output_vectors.push_back(process_tensor(tensor));
      shape_vectors.push_back(tensor.sizes().vec());
    }
  } else if (output.isTensor()) {
    // Handle single tensor output
    torch::Tensor tensor = output.toTensor().to(torch::kCPU).contiguous();

    output_vectors.push_back(process_tensor(tensor));
    shape_vectors.push_back(tensor.sizes().vec());
  } else {
    LOG(ERROR) << "Unsupported output type: neither tensor, tuple, nor list";
    std::exit(1);
  }

  return std::make_tuple(output_vectors, shape_vectors);
}