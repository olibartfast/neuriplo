#pragma once
#include "InferenceInterface.hpp"
#include <NvInfer.h>  // for TensorRT API
#include <cuda_runtime_api.h>  // for CUDA runtime API
#include <fstream>
#include <vector>

#include "Logger.hpp"

class TRTInfer : public InferenceInterface
{
    protected:
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        nvinfer1::IExecutionContext* context_;
        std::vector<void*> buffers_;
        nvinfer1::IRuntime* runtime_;
        size_t num_inputs_ = 0;
        size_t num_outputs_ = 0;
        std::vector<std::string> input_tensor_names_;
        std::vector<std::string> output_tensor_names_;
        int batch_size_ = 1;

    public:
        TRTInfer(const std::string& model_path, 
        bool use_gpu = true, 
        size_t batch_size = 1, 
        const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

        // Create execution context and allocate input/output buffers
        void createContextAndAllocateBuffers();

        void initializeBuffers(const std::string& engine_path);

        // calculate size of tensor
        size_t getSizeByDim(const nvinfer1::Dims& dims);    

        void infer(); // You might want to remove or update this depending on your needs

        std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> get_infer_results(const std::vector<cv::Mat>& input_tensors) override;

        void populateInferenceMetadata(const std::vector<std::vector<int64_t>>& input_sizes); 
        ~TRTInfer();
};