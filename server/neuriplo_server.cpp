#include "InferenceBackendSetup.hpp"
#include "InferenceServer.hpp"
#include <glog/logging.h>
#include <signal.h>
#include <memory>
#include <iostream>
#include <cstdlib>

// Global server pointer for signal handling
std::unique_ptr<neuriplo::server::InferenceServer> g_server;

void signal_handler(int signal) {
    LOG(INFO) << "Received signal " << signal << ", shutting down server...";
    if (g_server) {
        g_server->stop();
    }
    exit(0);
}

void print_usage(const char* program_name) {
    std::cout << "Neuriplo Inference Server\n\n"
              << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required:\n"
              << "  --model PATH        Path to the model file\n\n"
              << "Optional:\n"
              << "  --host HOST         Server host address (default: 0.0.0.0)\n"
              << "  --port PORT         Server port (default: 8080)\n"
              << "  --gpu               Enable GPU acceleration\n"
              << "  --batch-size SIZE   Batch size for inference (default: 1)\n"
              << "  --help              Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --model model.onnx\n"
              << "  " << program_name << " --model model.onnx --gpu --port 9090\n"
              << "  " << program_name << " --model model.onnx --host 192.168.1.100 --batch-size 4\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Initialize Google's logging library
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    
    // Parse command line arguments
    std::string model_path;
    std::string host = "0.0.0.0";
    int port = 8080;
    bool use_gpu = false;
    size_t batch_size = 1;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (model_path.empty()) {
        std::cerr << "Error: --model is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        LOG(INFO) << "Starting Neuriplo Inference Server";
        LOG(INFO) << "Model: " << model_path;
        LOG(INFO) << "Host: " << host;
        LOG(INFO) << "Port: " << port;
        LOG(INFO) << "GPU: " << (use_gpu ? "enabled" : "disabled");
        LOG(INFO) << "Batch size: " << batch_size;
        
        // Setup inference backend
        auto backend = setup_inference_engine(model_path, use_gpu, batch_size);
        
        if (!backend) {
            LOG(ERROR) << "Failed to initialize inference backend";
            return 1;
        }
        
        LOG(INFO) << "Backend initialized successfully";
        
        // Create and start server
        g_server = std::make_unique<neuriplo::server::InferenceServer>(
            std::move(backend), host, port);
        
        // Setup signal handlers
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        LOG(INFO) << "Server ready to accept connections";
        LOG(INFO) << "Endpoints:";
        LOG(INFO) << "  - POST http://" << host << ":" << port << "/infer";
        LOG(INFO) << "  - GET  http://" << host << ":" << port << "/model_info";
        LOG(INFO) << "  - GET  http://" << host << ":" << port << "/health";
        LOG(INFO) << "  - GET  http://" << host << ":" << port << "/stats";
        LOG(INFO) << "Press Ctrl+C to stop the server";
        
        // Start server (blocking)
        g_server->start();
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Fatal error: " << e.what();
        return 1;
    }
    
    return 0;
}
