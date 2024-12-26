#pragma once
#include "InferenceInterface.hpp"
#include <torch/torch.h>
#include <torch/script.h>

class LibtorchInfer : public InferenceInterface
{
public:
    LibtorchInfer(const std::string& model_path, 
        bool use_gpu = false, 
        size_t batch_size = 1, 
        const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;

private:
    torch::DeviceType device_;
    torch::jit::script::Module module_;    
  
};