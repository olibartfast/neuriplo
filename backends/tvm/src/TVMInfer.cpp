#include "TVMInfer.hpp"
#include <sstream>
#include <fstream>

std::string TVMInfer::print_shape(const std::vector<int64_t>& shape)
{
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        ss << shape[i];
        if (i < shape.size() - 1)
        {
            ss << ", ";
        }
    }
    ss << ")";
    return ss.str();
}

TVMInfer::TVMInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                   const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}
{
    // Determine device
    if (use_gpu)
    {
        device_.device_type = kDLCUDA;
        device_.device_id = 0;
        LOG(INFO) << "Using CUDA GPU for TVM inference";
    }
    else
    {
        device_.device_type = kDLCPU;
        device_.device_id = 0;
        LOG(INFO) << "Using CPU for TVM inference";
    }

    try
    {
        // Load the compiled model
        // TVM models typically consist of:
        // 1. .so file (compiled library)
        // 2. .json file (graph definition)
        // 3. .params file (parameters)

        std::string lib_path = model_path + ".so";
        std::string graph_path = model_path + ".json";
        std::string params_path = model_path + ".params";

        // Load the compiled library
        module_ = tvm::runtime::Module::LoadFromFile(lib_path);

        // Load graph JSON
        std::ifstream json_file(graph_path);
        if (!json_file.is_open())
        {
            throw std::runtime_error("Failed to open graph JSON file: " + graph_path);
        }
        std::string json_data((std::istreambuf_iterator<char>(json_file)),
                             std::istreambuf_iterator<char>());
        json_file.close();

        // Load parameters
        std::ifstream params_file(params_path, std::ios::binary);
        if (!params_file.is_open())
        {
            throw std::runtime_error("Failed to open params file: " + params_path);
        }
        params_file.seekg(0, std::ios::end);
        size_t params_size = params_file.tellg();
        params_file.seekg(0, std::ios::beg);
        std::vector<char> params_data(params_size);
        params_file.read(params_data.data(), params_size);
        params_file.close();

        // Create graph runtime
        int device_type = static_cast<int>(device_.device_type);
        int device_id = device_.device_id;

        tvm::runtime::Module graph_rt = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))
            (json_data, module_, device_type, device_id);

        // Load parameters
        auto load_params = graph_rt.GetFunction("load_params");
        TVMByteArray params_arr;
        params_arr.data = params_data.data();
        params_arr.size = params_size;
        load_params(params_arr);

        // Get runtime functions
        set_input_ = graph_rt.GetFunction("set_input");
        get_output_ = graph_rt.GetFunction("get_output");
        run_ = graph_rt.GetFunction("run");

        // Get number of inputs and outputs
        auto get_num_inputs = graph_rt.GetFunction("get_num_inputs");
        auto get_num_outputs = graph_rt.GetFunction("get_num_outputs");
        num_inputs_ = get_num_inputs();
        num_outputs_ = get_num_outputs();

        LOG(INFO) << "TVM model loaded successfully";
        LOG(INFO) << "Number of inputs: " << num_inputs_;
        LOG(INFO) << "Number of outputs: " << num_outputs_;

        // Process input information
        LOG(INFO) << "Input Node Name/Shape:";
        for (int i = 0; i < num_inputs_; ++i)
        {
            std::vector<int64_t> input_shape;

            if (!input_sizes.empty() && i < static_cast<int>(input_sizes.size()))
            {
                input_shape = input_sizes[i];
            }
            else
            {
                // Default shape for image input (NCHW format)
                input_shape = {static_cast<int64_t>(batch_size), 3, 224, 224};
                LOG(WARNING) << "No input shape provided for input " << i
                            << ", using default: " << print_shape(input_shape);
            }

            input_shapes_.push_back(input_shape);
            std::string input_name = "input_" + std::to_string(i);
            LOG(INFO) << "\t" << input_name << " : " << print_shape(input_shape);
            model_info_.addInput(input_name, input_shape, batch_size);
        }

        // Process output information
        LOG(INFO) << "Output Node Name/Shape:";
        for (int i = 0; i < num_outputs_; ++i)
        {
            std::vector<int64_t> output_shape;

            // For classification models, default output shape
            if (output_shape.empty())
            {
                output_shape = {static_cast<int64_t>(batch_size), 1000};
                LOG(WARNING) << "No output shape metadata available for output " << i
                            << ", using default: " << print_shape(output_shape);
            }

            output_shapes_.push_back(output_shape);
            std::string output_name = "output_" + std::to_string(i);
            LOG(INFO) << "\t" << output_name << " : " << print_shape(output_shape);
            model_info_.addOutput(output_name, output_shape, batch_size);
        }
    }
    catch (const std::exception& e)
    {
        LOG(ERROR) << "Failed to load TVM model: " << e.what();
        throw ModelLoadException(e.what());
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
TVMInfer::get_infer_results(const cv::Mat& preprocessed_img)
{
    start_timer();

    try
    {
        // Convert OpenCV Mat to TVM tensor
        cv::Mat blob;
        cv::dnn::blobFromImage(preprocessed_img, blob, 1.0, cv::Size(), cv::Scalar(), false, false);

        // Create TVM NDArray for input
        std::vector<int64_t> shape = {1, blob.size[1], blob.size[2], blob.size[3]};

        DLTensor input_tensor;
        input_tensor.data = blob.data;
        input_tensor.device = DLDevice{kDLCPU, 0};
        input_tensor.ndim = 4;
        input_tensor.dtype = DLDataType{kDLFloat, 32, 1};
        input_tensor.shape = shape.data();
        input_tensor.strides = nullptr;
        input_tensor.byte_offset = 0;

        // Create TVM NDArray from DLTensor
        tvm::runtime::NDArray input_array = tvm::runtime::NDArray::FromDLPack(&input_tensor);

        // Copy to device if using GPU
        if (device_.device_type == kDLCUDA)
        {
            input_array = input_array.CopyTo(device_);
        }

        // Set input
        set_input_(0, input_array);

        // Run inference
        run_();

        // Get outputs
        std::vector<std::vector<TensorElement>> output_vectors;
        std::vector<std::vector<int64_t>> shape_vectors;

        for (int i = 0; i < num_outputs_; ++i)
        {
            tvm::runtime::NDArray output_array = get_output_(i);

            // Copy to CPU if on GPU
            if (output_array->device.device_type == kDLCUDA)
            {
                output_array = output_array.CopyTo(DLDevice{kDLCPU, 0});
            }

            // Get shape
            std::vector<int64_t> output_shape;
            for (int j = 0; j < output_array->ndim; ++j)
            {
                output_shape.push_back(output_array->shape[j]);
            }
            shape_vectors.push_back(output_shape);

            // Get data
            size_t num_elements = 1;
            for (auto dim : output_shape)
            {
                num_elements *= dim;
            }

            std::vector<TensorElement> output_data;
            output_data.reserve(num_elements);

            // Assuming float32 output (most common case)
            const float* data_ptr = static_cast<const float*>(output_array->data);
            for (size_t j = 0; j < num_elements; ++j)
            {
                output_data.emplace_back(data_ptr[j]);
            }

            output_vectors.push_back(output_data);
        }

        end_timer();
        return std::make_tuple(output_vectors, shape_vectors);
    }
    catch (const std::exception& e)
    {
        LOG(ERROR) << "TVM inference failed: " << e.what();
        throw InferenceExecutionException(e.what());
    }
}
