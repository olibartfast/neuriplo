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
        if (session_) {
            tensorflow::Status status = session_->Close();
            if (!status.ok()) {
                std::cerr << "Warning: Error closing TensorFlow session: " << status.ToString() << std::endl;
            }
        }
    }

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;

private:

    std::string model_path_;
    tensorflow::SavedModelBundle bundle_;   
    std::unique_ptr<tensorflow::Session> session_; 
    tensorflow::TensorInfo input_info_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    
};