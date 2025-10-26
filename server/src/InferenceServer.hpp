#pragma once
#include "common.hpp"
#include "InferenceInterface.hpp"
#include "Serialization.hpp"
#include <httplib.h>
#include <memory>
#include <string>
#include <atomic>
#include <glog/logging.h>

namespace neuriplo {
namespace server {

class InferenceServer {
public:
    InferenceServer(std::unique_ptr<InferenceInterface> backend, 
                   const std::string& host = "0.0.0.0",
                   int port = 8080)
        : backend_(std::move(backend))
        , host_(host)
        , port_(port)
        , running_(false)
        , total_requests_(0)
        , failed_requests_(0) {
        
        if (!backend_) {
            throw std::runtime_error("Backend cannot be null");
        }
        
        LOG(INFO) << "Initializing Inference Server on " << host_ << ":" << port_;
        setup_routes();
    }
    
    ~InferenceServer() {
        stop();
    }
    
    // Start the server (blocking)
    void start() {
        if (running_) {
            LOG(WARNING) << "Server is already running";
            return;
        }
        
        running_ = true;
        LOG(INFO) << "Starting Inference Server on " << host_ << ":" << port_;
        
        if (!server_.listen(host_.c_str(), port_)) {
            running_ = false;
            throw std::runtime_error("Failed to start server on " + host_ + ":" + std::to_string(port_));
        }
    }
    
    // Stop the server
    void stop() {
        if (running_) {
            LOG(INFO) << "Stopping Inference Server";
            server_.stop();
            running_ = false;
        }
    }
    
    // Check if server is running
    bool is_running() const {
        return running_;
    }
    
    // Get server statistics
    size_t get_total_requests() const { return total_requests_; }
    size_t get_failed_requests() const { return failed_requests_; }

private:
    void setup_routes() {
        // Health check endpoint
        server_.Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
            handle_health(req, res);
        });
        
        // Model info endpoint
        server_.Get("/model_info", [this](const httplib::Request& req, httplib::Response& res) {
            handle_model_info(req, res);
        });
        
        // Inference endpoint
        server_.Post("/infer", [this](const httplib::Request& req, httplib::Response& res) {
            handle_inference(req, res);
        });
        
        // Statistics endpoint
        server_.Get("/stats", [this](const httplib::Request& req, httplib::Response& res) {
            handle_stats(req, res);
        });
        
        // Error handler
        server_.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json error_response;
            error_response["error"] = "Endpoint not found";
            error_response["status"] = res.status;
            res.set_content(error_response.dump(), "application/json");
        });
    }
    
    void handle_health(const httplib::Request& req, httplib::Response& res) {
        nlohmann::json response;
        response["status"] = "healthy";
        response["gpu_available"] = backend_->is_gpu_available();
        response["model_path"] = backend_->get_model_path();
        response["total_requests"] = total_requests_.load();
        
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    }
    
    void handle_model_info(const httplib::Request& req, httplib::Response& res) {
        try {
            auto model_info = backend_->get_model_info();
            auto response = serialization::serialize_model_info(model_info);
            
            res.set_content(response.dump(), "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            LOG(ERROR) << "Error getting model info: " << e.what();
            nlohmann::json error_response;
            error_response["error"] = e.what();
            res.set_content(error_response.dump(), "application/json");
            res.status = 500;
            failed_requests_++;
        }
    }
    
    void handle_inference(const httplib::Request& req, httplib::Response& res) {
        total_requests_++;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Parse JSON request
            auto json_req = nlohmann::json::parse(req.body);
            
            if (!json_req.contains("input_blob")) {
                throw std::runtime_error("Missing 'input_blob' field in request");
            }
            
            // Deserialize input blob (raw tensor data)
            cv::Mat input_blob = serialization::deserialize_mat(json_req["input_blob"]);
            
            if (input_blob.empty()) {
                throw std::runtime_error("Failed to deserialize input blob");
            }
            
            LOG(INFO) << "Processing inference request (blob shape: ";
            for (int i = 0; i < input_blob.dims; ++i) {
                LOG(INFO) << input_blob.size[i] << (i < input_blob.dims - 1 ? " x " : "");
            }
            LOG(INFO) << ")";
            
            // Run inference
            auto [outputs, shapes] = backend_->get_infer_results(input_blob);
            
            // Serialize results
            auto response = serialization::serialize_inference_results(outputs, shapes);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            response["inference_time_ms"] = backend_->get_last_inference_time_ms();
            response["total_time_ms"] = duration.count();
            
            res.set_content(response.dump(), "application/json");
            res.status = 200;
            
            LOG(INFO) << "Inference completed in " << duration.count() << "ms";
            
        } catch (const std::exception& e) {
            LOG(ERROR) << "Inference error: " << e.what();
            nlohmann::json error_response;
            error_response["error"] = e.what();
            res.set_content(error_response.dump(), "application/json");
            res.status = 500;
            failed_requests_++;
        }
    }
    
    void handle_stats(const httplib::Request& req, httplib::Response& res) {
        nlohmann::json response;
        response["total_requests"] = total_requests_.load();
        response["failed_requests"] = failed_requests_.load();
        response["success_rate"] = total_requests_ > 0 
            ? 100.0 * (total_requests_ - failed_requests_) / total_requests_ 
            : 100.0;
        response["total_inferences"] = backend_->get_total_inferences();
        response["avg_inference_time_ms"] = backend_->get_last_inference_time_ms();
        response["memory_usage_mb"] = backend_->get_memory_usage_mb();
        
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    }
    
    std::unique_ptr<InferenceInterface> backend_;
    httplib::Server server_;
    std::string host_;
    int port_;
    std::atomic<bool> running_;
    std::atomic<size_t> total_requests_;
    std::atomic<size_t> failed_requests_;
};

} // namespace server
} // namespace neuriplo
