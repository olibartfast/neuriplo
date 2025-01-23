#include "TRTInfer.hpp"
#include <fstream>
#include <cuda_fp16.h> // For __half if using half-precision

// Added: CUDA error checking macro
#define CHECK_CUDA(status) \
    do { \
        auto ret = (status); \
        if (ret != cudaSuccess) { \
            LOG(ERROR) << "CUDA error: " << cudaGetErrorString(ret); \
            std::exit(1); \
        } \
    } while (0)

TRTInfer::TRTInfer(const std::string& model_path) : InferenceInterface{model_path, true}
{
    LOG(INFO) << "Initializing TensorRT for model " << model_path;
    initializeBuffers(model_path);
}
void TRTInfer::initializeBuffers(const std::string& engine_path)
{
    // Create TensorRT runtime
    Logger logger;
    runtime_ = nvinfer1::createInferRuntime(logger);

    // Load engine file
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file)
    {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }
    engine_file.seekg(0, std::ios::end);
    size_t file_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(file_size);
    engine_file.read(engine_data.data(), file_size);
    engine_file.close();

    // Deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), file_size)); 
    createContextAndAllocateBuffers();
}

// calculate size of tensor
size_t TRTInfer::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        if(dims.d[i] == -1 || dims.d[i] == 0)
        {
            continue;
        }
        size *= dims.d[i];
    }
    return size;
}

void TRTInfer::createContextAndAllocateBuffers()
{
    context_ = engine_->createExecutionContext();
    int num_tensors = engine_->getNbIOTensors();
    buffers_.resize(num_tensors);
    input_tensor_names_.resize(num_tensors);
    output_tensor_names_.resize(num_tensors);

    for (int i = 0; i < num_tensors; ++i)
    {
        std::string tensor_name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(tensor_name.c_str());
        auto size = getSizeByDim(dims);
        size_t binding_size = 0;
        switch (engine_->getTensorDataType(tensor_name.c_str()))
        {
            case nvinfer1::DataType::kFLOAT:
                binding_size = size * sizeof(float);
                break;
            case nvinfer1::DataType::kINT32:
                binding_size = size * sizeof(int);
                break;
            case nvinfer1::DataType::kHALF:
                binding_size = size * sizeof(__half);
                break;
            default:
                LOG(ERROR) << "Unsupported data type for tensor " << tensor_name;
                std::exit(1);
        }
        CHECK_CUDA(cudaMalloc(&buffers_[i], binding_size));

        if (engine_->getTensorIOMode(tensor_name.c_str()) == nvinfer1::TensorIOMode::kINPUT)
        {
            LOG(INFO) << "Input tensor " << num_inputs_ << ": " << tensor_name;
            input_tensor_names_[num_inputs_] = tensor_name;
            num_inputs_++;
        }
        else
        {
            LOG(INFO) << "Output tensor " << num_outputs_ << ": " << tensor_name;
            output_tensor_names_[num_outputs_] = tensor_name;
            num_outputs_++;
        }
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> TRTInfer::get_infer_results(const cv::Mat& preprocessed_img)
{
    // Convert the input image to a blob swapping channels order from hwc to chw    
    cv::Mat blob;
    cv::dnn::blobFromImage(preprocessed_img, blob, 1.0, cv::Size(), cv::Scalar(), false, false);

    for (size_t i = 0; i < num_inputs_; ++i)
    {
        std::string tensor_name = input_tensor_names_[i];
        nvinfer1::Dims dims = engine_->getTensorShape(tensor_name.c_str());
        size_t size = getSizeByDim(dims);
        size_t binding_size = 0;

        switch (engine_->getTensorDataType(tensor_name.c_str()))
        {
            case nvinfer1::DataType::kFLOAT:
                binding_size = size * sizeof(float);
                break;
            case nvinfer1::DataType::kINT32:
                binding_size = size * sizeof(int32_t);
                break;
            case nvinfer1::DataType::kHALF:
                binding_size = size * sizeof(__half);
                break;
            default:
                LOG(ERROR) << "Unsupported input data type for tensor " << tensor_name;
                std::exit(1);
        }

        // Copy data to the appropriate GPU buffer
        if (tensor_name == "input0") // Assuming the first input tensor name is "input0"
        {
            CHECK_CUDA(cudaMemcpy(buffers_[i], blob.data, binding_size, cudaMemcpyHostToDevice));
        }
        else if (tensor_name == "input1") // If there's a second input, e.g., for target sizes in RT-DETR model
        {
            std::vector<int32_t> orig_target_sizes = { static_cast<int32_t>(blob.size[2]), static_cast<int32_t>(blob.size[3]) };
            CHECK_CUDA(cudaMemcpy(buffers_[i], orig_target_sizes.data(), binding_size, cudaMemcpyHostToDevice));
        }
    }

    // Perform inference
    cudaStream_t stream = 0; // Assuming no stream, otherwise create one
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (size_t i = 0; i < num_inputs_; ++i) {
        if (!context_->setInputTensorAddress(input_tensor_names_[i].c_str(), buffers_[i])) {
            LOG(ERROR) << "Failed to set input tensor address for tensor: " << input_tensor_names_[i];
            std::exit(1);
        }
    }

    for (size_t i = 0; i < num_outputs_; ++i) {
        if (!context_->setOutputTensorAddress(output_tensor_names_[i].c_str(), buffers_[i + num_inputs_])) {
            LOG(ERROR) << "Failed to set output tensor address for tensor: " << output_tensor_names_[i];
            std::exit(1);
        }
    }

    if (!context_->enqueueV3(stream)) 
    {
        LOG(ERROR) << "Inference failed!";
        std::exit(1);
    }

    // Extract outputs and their shapes
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<std::vector<TensorElement>> outputs;
    
    for (size_t i = 0; i < num_outputs_; ++i)
    {
        std::string tensor_name = output_tensor_names_[i];
        nvinfer1::Dims dims = engine_->getTensorShape(tensor_name.c_str());
        auto num_elements = getSizeByDim(dims);
        
        std::vector<TensorElement> tensor_data;
        
        switch (engine_->getTensorDataType(tensor_name.c_str()))
        {
            case nvinfer1::DataType::kFLOAT:
            {
                std::vector<float> output_data_float(num_elements);
                CHECK_CUDA(cudaMemcpy(output_data_float.data(), buffers_[i + num_inputs_],  num_elements * sizeof(float), cudaMemcpyDeviceToHost));
                
                for (const auto& value : output_data_float) {
                    tensor_data.push_back(static_cast<float>(value));
                }
                break;
            }
            case nvinfer1::DataType::kINT32:
            {
                std::vector<int32_t> output_data_int(num_elements);
                CHECK_CUDA(cudaMemcpy(output_data_int.data(), buffers_[i + num_inputs_],  num_elements * sizeof(int32_t), cudaMemcpyDeviceToHost));
                
                for (const auto& value : output_data_int) {
                    tensor_data.push_back(static_cast<int32_t>(value));
                }
                break;
            }
            case nvinfer1::DataType::kHALF:
            {
                std::vector<__half> output_data_half(num_elements);
                CHECK_CUDA(cudaMemcpy(output_data_half.data(), buffers_[i + num_inputs_],  num_elements * sizeof(__half), cudaMemcpyDeviceToHost));
                
                for (const auto& value : output_data_half) {
                    tensor_data.push_back(static_cast<float>(__half2float(value)));
                }
                break;
            }
            default:
                LOG(ERROR) << "Unsupported output data type for tensor " << tensor_name;
                std::exit(1);
        }
        
        outputs.emplace_back(std::move(tensor_data));
        
        const int64_t curr_batch = dims.d[0] == -1 ? 1 : dims.d[0];
        std::vector<int64_t> out_shape;
        for (int j = 0; j < dims.nbDims; ++j)
        {
            out_shape.push_back(dims.d[j]);
        }
        output_shapes.emplace_back(out_shape);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));

    return std::make_tuple(std::move(outputs), std::move(output_shapes));
}