#pragma once
#include "common.hpp"
#include <variant>
using TensorElement = std::variant<float, int32_t, int64_t>;

#include "ModelInfo.hpp"

class InferenceInterface{
    	
    public:
        InferenceInterface(const std::string& weights,
         bool use_gpu = false, 
         size_t batch_size = 1,
         const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>())
        {

        }

         ModelInfo get_model_info() {
            if (model_info_.getInputs().empty() || model_info_.getOutputs().empty()) {
                throw std::runtime_error("Model parameters are not initialized, dynamic shapes are not currently supported, stay tuned for future updates!");
            }
            return model_info_;
        }

        virtual std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) = 0;

    protected:
        ModelInfo model_info_;
        std::vector<float> blob2vec(const cv::Mat& input_blob);

};