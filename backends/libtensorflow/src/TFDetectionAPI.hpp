#pragma once
#include "InferenceInterface.hpp"
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include "opencv2/opencv.hpp"

class TFDetectionAPI : public InferenceInterface{

public:
    TFDetectionAPI(const std::string& model_path, 
        bool use_gpu = false, 
        size_t batch_size = 1, 
        const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    ~TFDetectionAPI() {
        // The session is owned by bundle_, so we don't need to close it manually
        // bundle_ will handle the session cleanup in its destructor
    }

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;

private:

    std::string model_path_;
    tensorflow::SavedModelBundle bundle_;   
    tensorflow::TensorInfo input_info_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    
};