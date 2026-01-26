#include <cstdint>
#include "InferenceInterface.hpp"

InferenceInterface::InferenceInterface(const std::string& weights,
    bool use_gpu, 
    size_t batch_size,
    const std::vector<std::vector<int64_t>>& input_sizes)
    : model_path_(weights)
    , gpu_available_(use_gpu)
    , batch_size_(batch_size)
    , last_inference_time_ms_(0.0)
    , total_inferences_(0)
    , memory_usage_mb_(0)
{
}

InferenceMetadata InferenceInterface::get_inference_metadata() {
    // OpenCV DNN module does not have a method to get input layer shapes and names 
    if (inference_metadata_.getInputs().empty() && inference_metadata_.getOutputs().empty()) {
        throw ModelLoadException("Model information is not available - inputs and outputs are empty");
    }
    return inference_metadata_;
}

void InferenceInterface::clear_cache() noexcept {
    // Default implementation - do nothing
    // Derived classes can override this if they need cache management
}

size_t InferenceInterface::get_memory_usage_mb() const noexcept {
    // Default implementation - return 0
    // Derived classes can override this to provide actual memory usage
    return 0;
}



void InferenceInterface::start_timer() {
    inference_start_time_ = std::chrono::high_resolution_clock::now();
}

void InferenceInterface::end_timer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - inference_start_time_);
    last_inference_time_ms_ = duration.count() / 1000.0;
    total_inferences_++;
}



void InferenceInterface::validate_model_loaded() const {
    if (model_path_.empty()) {
        throw ModelLoadException("Model path is not specified");
    }
}


