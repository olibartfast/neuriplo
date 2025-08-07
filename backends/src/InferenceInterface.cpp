#include "InferenceInterface.hpp"

InferenceInterface::InferenceInterface(const std::string& weights,
    bool use_gpu, 
    size_t batch_size,
    const std::vector<std::vector<int64_t>>& input_sizes)
{

}

ModelInfo InferenceInterface::get_model_info() noexcept {
    // OpenCV DNN module does not have a method to get input layer shapes and names 
    if (model_info_.getInputs().empty() && model_info_.getOutputs().empty()) {
        // Add default input and output info if not set
        std::vector<int64_t> input_shape = {3, 224, 224}; // CHW format
        model_info_.addInput("input", input_shape, 1);
        
        std::vector<int64_t> output_shape = {1000}; // Classification output
        model_info_.addOutput("output", output_shape, 1);
    }
    return model_info_;
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

std::vector<float> InferenceInterface::blob2vec(const cv::Mat& input_blob)
{

    const auto channels = input_blob.size[1];
    const auto network_width = input_blob.size[2];
    const auto network_height = input_blob.size[3];
    size_t img_byte_size = network_width * network_height * channels * sizeof(float);  // Allocate a buffer to hold all image elements.
    std::vector<float> input_data = std::vector<float>(network_width * network_height * channels);
    std::memcpy(input_data.data(), input_blob.data, img_byte_size);

    std::vector<cv::Mat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::Mat(cv::Size(network_width, network_height), CV_32FC1, &(input_data[i * network_width * network_height])));
    }
    cv::split(input_blob, chw);

    return input_data;    
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

