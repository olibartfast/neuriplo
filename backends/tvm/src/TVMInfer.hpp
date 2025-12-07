#pragma once

// Define missing macros for TVM 0.22.0 compatibility BEFORE including TVM headers
#ifndef TVM_ALWAYS_INLINE
#define TVM_ALWAYS_INLINE inline
#endif

#include "InferenceInterface.hpp"
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>

// Forward declarations for TVM 0.22.0
namespace tvm {
namespace runtime {
class NDArray;
}
}

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
    void* module_handle_;  // Use opaque pointer for TVM module
    DLDevice device_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    int num_inputs_;
    int num_outputs_;
    bool model_loaded_;
};
