#pragma once
#include "InferenceInterface.hpp"
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include "opencv2/opencv.hpp"

class TFDetectionAPI : public InferenceInterface{

public:
    TFDetectionAPI(const std::string& model_path, bool use_gpu) : InferenceInterface{model_path, "", use_gpu}
    {
        tensorflow::SessionOptions session_options;
        tensorflow::RunOptions run_options;
        tensorflow::Status status = LoadSavedModel(session_options, run_options, 
            model_path, {tensorflow::kSavedModelTagServe}, &bundle_);

        if (!status.ok()) {
            LOG(ERROR) << "Error loading the model: " << status.ToString();
            std::exit(1);
        }

        session_.reset(bundle_.GetSession());

        // Get the SignatureDef
        const auto& signature_def = bundle_.GetSignatures().at("serving_default");

        // Get input tensor info
        const auto& inputs = signature_def.inputs();
        if (inputs.empty()) {
            LOG(ERROR) << "No inputs found in the model";
            std::exit(1);
        }
        input_info_ = inputs.begin()->second;
        input_name_ = input_info_.name();
        LOG(INFO) << "Tensor Input name: " << input_name_;

        // Get output tensor names
        LOG(INFO) << "Tensor output names:";
        for (const auto& output : signature_def.outputs()) {
            output_names_.push_back(output.second.name());
            LOG(INFO) << output.second.name();
        }

    }

    ~TFDetectionAPI() {
        tensorflow::Status status = session_->Close();
        if (!status.ok()) {
            std::cerr << "Error closing TensorFlow session: " << status.ToString() << std::endl;
        }
    }

private:

    std::string model_path_;
    tensorflow::SavedModelBundle bundle_;   
    std::unique_ptr<tensorflow::Session> session_; 
    tensorflow::TensorInfo input_info_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    
    std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;    
};