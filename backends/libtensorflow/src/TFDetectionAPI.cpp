#include "TFDetectionAPI.hpp"


std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> TFDetectionAPI::get_infer_results(const cv::Mat& input_blob) 
{
    // Prepare input tensor
    tensorflow::Tensor input_tensor(input_info_.dtype(), 
        tensorflow::TensorShape({1, input_blob.size[0], input_blob.size[1], input_blob.channels()}));
  
    std::memcpy(input_tensor.flat<float>().data(), input_blob.data, input_blob.total() * input_blob.elemSize());

    // Prepare inputs for running the session
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_for_session = {
        {input_name_, input_tensor}
    };

    // Run the inference
    std::vector<tensorflow::Tensor> outputs;
    auto status = session_->Run(inputs_for_session, output_names_, {}, &outputs);
    if (!status.ok()) {
        LOG(ERROR) << "Error running session: " << status.ToString();
        std::exit(1);
    }

    std::vector<std::vector<std::any>> convertedOutputs;
    std::vector<std::vector<int64_t>> shapes;

    for (const auto& tensor : outputs) {
        std::vector<std::any> outputData;
        if (tensor.dtype() == tensorflow::DataType::DT_FLOAT) {
            for (int i = 0; i < tensor.NumElements(); ++i) {
                outputData.push_back(tensor.flat<float>()(i));
            }
        } else if (tensor.dtype() == tensorflow::DataType::DT_INT32) {
            for (int i = 0; i < tensor.NumElements(); ++i) {
                outputData.push_back(tensor.flat<int32_t>()(i));
            }
        } else if (tensor.dtype() == tensorflow::DataType::DT_INT64) {
            for (int i = 0; i < tensor.NumElements(); ++i) {
                outputData.push_back(tensor.flat<int64_t>()(i));
            }
        } else {
            LOG(ERROR) << "Unsupported output data type";
        }
        convertedOutputs.push_back(outputData);

        std::vector<int64_t> outputShape;
        for (int i = 0; i < tensor.dims(); ++i) {
            outputShape.push_back(tensor.dim_size(i));
        }
        shapes.push_back(outputShape);
    }
    return std::make_tuple(convertedOutputs, shapes);
}