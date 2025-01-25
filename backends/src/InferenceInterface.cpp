#include "InferenceInterface.hpp"

InferenceInterface::InferenceInterface(const std::string& weights,
    bool use_gpu, 
    size_t batch_size,
    const std::vector<std::vector<int64_t>>& input_sizes)
{

}

ModelInfo InferenceInterface::get_model_info() {
    // OpenCV DNN module does not have a method to get input layer shapes and names 
    if (model_info_.getInputs().empty() && model_info_.getOutputs().empty()) {
        throw std::runtime_error("Model parameters are not initialized, initialize the model info first inside the inference engine setup!");
    }
    return model_info_;
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

