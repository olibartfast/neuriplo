#include "OCVDNNInfer.hpp"

OCVDNNInfer::OCVDNNInfer(const std::string& weights, const std::string& modelConfiguration) : InferenceInterface{weights, modelConfiguration} 
{

        LOG(INFO) << "Running using OpenCV DNN runtime: " << weights;
        net_ = modelConfiguration.empty() ? cv::dnn::readNet(weights) : cv::dnn::readNetFromDarknet(modelConfiguration, weights);
        if (net_.empty())
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "weights-file: " << weights << std::endl;
            exit(-1);
        }
        outLayers_ = net_.getUnconnectedOutLayers();
        outLayerType_ = net_.getLayer(outLayers_[0])->type;
        outNames_ = net_.getUnconnectedOutLayersNames();


}
std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> OCVDNNInfer::get_infer_results(const cv::Mat& preprocessed_img)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(preprocessed_img, blob, 1.0, cv::Size(), cv::Scalar(), false, false);
    std::vector<std::vector<TensorElement>> outputs;
    std::vector<std::vector<int64_t>> shapes;

    std::vector<cv::Mat> outs;
    net_.setInput(blob);
    net_.forward(outs, outNames_);

    outputs.reserve(outs.size());
    shapes.reserve(outs.size());

    for (const auto& output : outs) {
        // Extracting dimensions of the output tensor
        std::vector<int64_t> shape;
        shape.reserve(output.dims);
        for (int j = 0; j < output.dims; ++j) {
            shape.push_back(output.size[j]);
        }
        shapes.push_back(std::move(shape));

        // Extracting data
        std::vector<TensorElement> tensor_data;
        tensor_data.reserve(output.total());

        if (output.type() == CV_32F) {
            const float* data = output.ptr<float>();
            for (int j = 0; j < output.total(); ++j) {
                tensor_data.push_back(data[j]);
            }
        } 
        else if (output.type() == CV_64F) {
            const double* data = output.ptr<double>();
            for (int j = 0; j < output.total(); ++j) {
                tensor_data.push_back(static_cast<float>(data[j]));
            }
        } 
        else {
            throw std::runtime_error("Unsupported data type in OCVDNNInfer::get_infer_results");
        }

        outputs.push_back(std::move(tensor_data));
    }

    return std::make_tuple(outputs, shapes);
}
