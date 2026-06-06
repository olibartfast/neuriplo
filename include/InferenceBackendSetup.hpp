#pragma once
#include "InferenceInterface.hpp"
#include "common.hpp"
std::unique_ptr<InferenceInterface>
setup_inference_engine(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
                       const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());
