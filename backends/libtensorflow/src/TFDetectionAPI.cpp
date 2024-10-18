#include "TFDetectionAPI.hpp"


std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> TFDetectionAPI::get_infer_results(const cv::Mat& input_blob) 
{
    // Convert the frame to a TensorFlow tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({1, input_blob.size[1], input_blob.size[2], input_blob.size[3]})); // NHWC
    
    std::memcpy(input_tensor.flat<uint8_t>().data(), input_blob.data, input_blob.total() * input_blob.elemSize());

    // Run the inference
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {"serving_default_input_tensor:0", input_tensor}
    };
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status status = session_->Run(inputs, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:3", "StatefulPartitionedCall:4"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << "Error running session: " << status.ToString() << "\n";
        std::exit(1);
    }
        
    std::vector<std::vector<TensorElement>> convertedOutputs;
    std::vector<std::vector<int64_t>> shapes;
    for (const auto& tensor : outputs) {
        std::vector<TensorElement> outputData;
        auto numDims = tensor.dims(); 
        std::vector<int64_t> outputShape(numDims); 
        
        for (int i = 0; i < numDims; ++i) {
            outputShape[i] = tensor.dim_size(i);
        }
        
        shapes.push_back(outputShape);
        
        if (tensor.dtype() == tensorflow::DataType::DT_FLOAT) {
            for (int i = 0; i < tensor.NumElements(); ++i) {
                outputData.emplace_back(tensor.flat<float>()(i)); // Use emplace_back for variant
            }
        } else if (tensor.dtype() == tensorflow::DataType::DT_INT32) {
            for (int i = 0; i < tensor.NumElements(); ++i) {
                outputData.emplace_back(tensor.flat<int32_t>()(i));
            }
        } else if (tensor.dtype() == tensorflow::DataType::DT_INT64) {
            for (int i = 0; i < tensor.NumElements(); ++i) {
                outputData.emplace_back(tensor.flat<int64_t>()(i));
            }
        } else {
            throw std::runtime_error("Unsupported output data type encountered.");
        }
        
        convertedOutputs.push_back(outputData);
    }
    return std::make_tuple(convertedOutputs, shapes);
}   