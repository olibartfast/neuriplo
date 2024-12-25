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

        virtual ModelInfo get_model_info() = 0;
        virtual std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) = 0;

    protected:
        std::vector<float> blob2vec(const cv::Mat& input_blob);

};