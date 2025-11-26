#pragma once
#include "InferenceInterface.hpp"
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <dlpack/dlpack.h>

class TVMInfer : public InferenceInterface
{
public:
    TVMInfer(const std::string& model_path,
        bool use_gpu = false,
        size_t batch_size = 1,
        const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    ~TVMInfer() override = default;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const cv::Mat& input_blob) override;

private:
    std::string print_shape(const std::vector<int64_t>& shape);
    tvm::runtime::Module module_;
    tvm::runtime::PackedFunc set_input_;
    tvm::runtime::PackedFunc get_output_;
    tvm::runtime::PackedFunc run_;
    DLDevice device_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    int num_inputs_;
    int num_outputs_;
};
