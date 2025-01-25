#pragma once
#include "InferenceInterface.hpp"

class OCVDNNInfer : public InferenceInterface
{
private:
	cv::dnn::Net net_;
    std::vector<int> outLayers_;
    std::string outLayerType_;
    std::vector<std::string> outNames_;
        
public:
    OCVDNNInfer(const std::string& model_path, 
        bool use_gpu = false, 
        size_t batch_size = 1, 
        const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;

    bool isCudaBuildEnabled() {
        std::string buildInfo = cv::getBuildInformation();
        size_t cudaPos = buildInfo.find("CUDA:");
        if (cudaPos != std::string::npos) {
            size_t statusPos = buildInfo.find("YES", cudaPos);
            if (statusPos != std::string::npos && statusPos > cudaPos){
            return true; // CUDA support enabled
            } else {
            return false; // CUDA support not enabled
            }
        } else {
            return false; // No CUDA info found
        }
    }

};
