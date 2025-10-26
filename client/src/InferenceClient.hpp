#pragma once
#include "common.hpp"
#include "InferenceInterface.hpp"
#include "Serialization.hpp"
#include <httplib.h>
#include <glog/logging.h>
#include <memory>
#include <string>
#include <chrono>

namespace neuriplo {
namespace client {

class InferenceClient : public InferenceInterface {
public:
    InferenceClient(const std::string& server_host = "localhost",
                   int server_port = 8080,
                   int timeout_seconds = 30)
        : InferenceInterface("remote_model", false, 1, {})
        , server_host_(server_host)
        , server_port_(server_port)
        , timeout_seconds_(timeout_seconds)
        , client_(std::make_unique<httplib::Client>(server_host, server_port)) {
        
        client_->set_read_timeout(timeout_seconds_, 0);
        client_->set_write_timeout(timeout_seconds_, 0);
        
        LOG(INFO) << "Initializing Inference Client for " << server_host_ << ":" << server_port_;
        
        // Check server health
        if (!check_health()) {
            throw std::runtime_error("Cannot connect to inference server at " + 
                                   server_host_ + ":" + std::to_string(server_port_));
        }
        
        // Fetch and cache model info
        fetch_model_info();
    }
    
    ~InferenceClient() override = default;
    
    // Implement the inference interface
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> 
    get_infer_results(const cv::Mat& input_blob) override {
        start_timer();
        
        try {
            // Serialize input blob (raw tensor data)
            nlohmann::json request;
            request["input_blob"] = serialization::serialize_mat(input_blob);
            
            LOG(INFO) << "Sending inference request to server";
            
            // Send POST request
            auto res = client_->Post("/infer", request.dump(), "application/json");
            
            if (!res) {
                throw InferenceExecutionException("Failed to connect to server: " + 
                                                 httplib::to_string(res.error()));
            }
            
            if (res->status != 200) {
                auto error_json = nlohmann::json::parse(res->body);
                throw InferenceExecutionException("Server error: " + 
                                                 error_json["error"].get<std::string>());
            }
            
            // Parse response
            auto response = nlohmann::json::parse(res->body);
            
            // Extract inference time from server
            if (response.contains("inference_time_ms")) {
                last_inference_time_ms_ = response["inference_time_ms"];
            }
            
            end_timer();
            
            LOG(INFO) << "Inference completed, total time: " << last_inference_time_ms_ << "ms";
            
            // Deserialize results
            return serialization::deserialize_inference_results(response);
            
        } catch (const std::exception& e) {
            end_timer();
            LOG(ERROR) << "Client inference error: " << e.what();
            throw InferenceExecutionException(std::string("Client error: ") + e.what());
        }
    }
    
    // Override get_model_info to return cached info from server
    ModelInfo get_model_info() noexcept override {
        return model_info_;
    }
    
    // Check if server is healthy
    bool check_health() {
        try {
            auto res = client_->Get("/health");
            if (res && res->status == 200) {
                auto health = nlohmann::json::parse(res->body);
                LOG(INFO) << "Server health check: " << health["status"];
                
                // Update GPU availability from server
                if (health.contains("gpu_available")) {
                    gpu_available_ = health["gpu_available"];
                }
                
                if (health.contains("model_path")) {
                    model_path_ = health["model_path"];
                }
                
                return true;
            }
            return false;
        } catch (const std::exception& e) {
            LOG(ERROR) << "Health check failed: " << e.what();
            return false;
        }
    }
    
    // Get server statistics
    nlohmann::json get_server_stats() {
        try {
            auto res = client_->Get("/stats");
            if (res && res->status == 200) {
                return nlohmann::json::parse(res->body);
            }
            throw std::runtime_error("Failed to get server stats");
        } catch (const std::exception& e) {
            LOG(ERROR) << "Failed to get server stats: " << e.what();
            throw;
        }
    }
    
    // Get server host
    std::string get_server_host() const { return server_host_; }
    
    // Get server port
    int get_server_port() const { return server_port_; }

private:
    void fetch_model_info() {
        try {
            auto res = client_->Get("/model_info");
            if (!res || res->status != 200) {
                LOG(WARNING) << "Failed to fetch model info from server";
                return;
            }
            
            auto info_json = nlohmann::json::parse(res->body);
            
            // Parse inputs
            for (const auto& input_json : info_json["inputs"]) {
                std::string name = input_json["name"];
                std::vector<int64_t> shape = input_json["shape"];
                int batch_size = input_json["batch_size"];
                model_info_.addInput(name, shape, batch_size);
            }
            
            // Parse outputs
            for (const auto& output_json : info_json["outputs"]) {
                std::string name = output_json["name"];
                std::vector<int64_t> shape = output_json["shape"];
                int batch_size = output_json["batch_size"];
                model_info_.addOutput(name, shape, batch_size);
            }
            
            LOG(INFO) << "Successfully fetched model info from server";
            
        } catch (const std::exception& e) {
            LOG(ERROR) << "Error fetching model info: " << e.what();
        }
    }
    
    std::string server_host_;
    int server_port_;
    int timeout_seconds_;
    std::unique_ptr<httplib::Client> client_;
};

} // namespace client
} // namespace neuriplo
