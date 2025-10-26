#pragma once
#include "common.hpp"
#include "InferenceInterface.hpp"
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <httplib.h>
#include <vector>
#include <string>
#include <variant>

namespace neuriplo {
namespace serialization {

using json = nlohmann::json;

// Serialize cv::Mat to JSON (raw tensor data, not image encoding)
// This is efficient for already-preprocessed model inputs (blobs)
inline json serialize_mat(const cv::Mat& mat) {
    json result;
    
    // Store shape information
    std::vector<int> shape;
    for (int i = 0; i < mat.dims; ++i) {
        shape.push_back(mat.size[i]);
    }
    result["shape"] = shape;
    result["type"] = mat.type();
    
    // Encode raw data as base64
    size_t data_size = mat.total() * mat.elemSize();
    result["data"] = httplib::detail::base64_encode(
        reinterpret_cast<const unsigned char*>(mat.data), 
        data_size
    );
    
    return result;
}

// Deserialize cv::Mat from JSON (raw tensor data)
inline cv::Mat deserialize_mat(const json& j) {
    // Extract shape and type
    std::vector<int> shape = j["shape"].get<std::vector<int>>();
    int type = j["type"];
    
    // Decode base64 data
    std::string data_b64 = j["data"];
    std::string decoded = httplib::detail::base64_decode(data_b64);
    
    // Create cv::Mat with the decoded data
    cv::Mat mat(shape.size(), shape.data(), type);
    std::memcpy(mat.data, decoded.data(), decoded.size());
    
    return mat;
}

// Convert TensorElement to JSON
inline json tensor_element_to_json(const TensorElement& element) {
    return std::visit([](auto&& arg) -> json {
        return json(arg);
    }, element);
}

// Convert JSON to TensorElement
inline TensorElement json_to_tensor_element(const json& j, const std::string& type) {
    if (type == "float") {
        return j.get<float>();
    } else if (type == "int32") {
        return j.get<int32_t>();
    } else if (type == "int64") {
        return j.get<int64_t>();
    }
    throw std::runtime_error("Unknown tensor element type: " + type);
}

// Serialize inference results to JSON
inline json serialize_inference_results(
    const std::vector<std::vector<TensorElement>>& outputs,
    const std::vector<std::vector<int64_t>>& shapes) {
    
    json result;
    result["outputs"] = json::array();
    
    for (size_t i = 0; i < outputs.size(); ++i) {
        json output_json;
        output_json["data"] = json::array();
        
        // Determine the type from the first element
        std::string type = "float";
        if (!outputs[i].empty()) {
            if (std::holds_alternative<int32_t>(outputs[i][0])) {
                type = "int32";
            } else if (std::holds_alternative<int64_t>(outputs[i][0])) {
                type = "int64";
            }
        }
        output_json["type"] = type;
        
        for (const auto& element : outputs[i]) {
            output_json["data"].push_back(tensor_element_to_json(element));
        }
        
        output_json["shape"] = shapes[i];
        result["outputs"].push_back(output_json);
    }
    
    return result;
}

// Deserialize inference results from JSON
inline std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
deserialize_inference_results(const json& j) {
    std::vector<std::vector<TensorElement>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    
    for (const auto& output_json : j["outputs"]) {
        std::vector<TensorElement> output;
        std::string type = output_json["type"];
        
        for (const auto& element : output_json["data"]) {
            output.push_back(json_to_tensor_element(element, type));
        }
        
        outputs.push_back(output);
        shapes.push_back(output_json["shape"].get<std::vector<int64_t>>());
    }
    
    return {outputs, shapes};
}

// Serialize ModelInfo to JSON
inline json serialize_model_info(const ModelInfo& model_info) {
    json result;
    result["inputs"] = json::array();
    result["outputs"] = json::array();
    
    for (const auto& input : model_info.getInputs()) {
        json input_json;
        input_json["name"] = input.name;
        input_json["shape"] = input.shape;
        input_json["batch_size"] = input.batch_size;
        result["inputs"].push_back(input_json);
    }
    
    for (const auto& output : model_info.getOutputs()) {
        json output_json;
        output_json["name"] = output.name;
        output_json["shape"] = output.shape;
        output_json["batch_size"] = output.batch_size;
        result["outputs"].push_back(output_json);
    }
    
    return result;
}

} // namespace serialization
} // namespace neuriplo
