#pragma once
#include "InferenceInterface.hpp"
#include <onnxruntime_cxx_api.h>  // for ONNX Runtime C++ API
#include <onnxruntime_c_api.h>    // for CUDA execution provider (if using CUDA)
#include <glog/logging.h>

class ORTInfer : public InferenceInterface
{
private:
    Ort::Env env_;
    Ort::Session session_{ nullptr };
    ModelInfo model_info_;
    static std::string getDataTypeString(ONNXTensorElementDataType type);

    template<typename T>
    void processTensorData(std::vector<TensorElement>& tensor_data, const T* data, size_t num_elements) {
        for (size_t i = 0; i < num_elements; ++i) {
            tensor_data.emplace_back(data[i]);
        }
    }
public:
    std::string print_shape(const std::vector<std::int64_t>& v);
    ORTInfer(const std::string& model_path, bool use_gpu = false);
    size_t getSizeByDim(const std::vector<int64_t>& dims);

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;
    ModelInfo get_model_info() override;
};