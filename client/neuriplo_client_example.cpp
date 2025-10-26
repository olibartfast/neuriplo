#include "InferenceBackendSetup.hpp"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <iostream>

void print_usage(const char* program_name) {
    std::cout << "Neuriplo Client Example\n\n"
              << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required:\n"
              << "  --image PATH        Path to input image\n"
              << "  --server HOST       Server hostname or IP (default: localhost)\n\n"
              << "Optional:\n"
              << "  --port PORT         Server port (default: 8080)\n"
              << "  --help              Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --image test.jpg\n"
              << "  " << program_name << " --image test.jpg --server 192.168.1.100 --port 9090\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Initialize Google's logging library
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    
    // Parse command line arguments
    std::string image_path;
    std::string server_host = "localhost";
    int port = 8080;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--server" && i + 1 < argc) {
            server_host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (image_path.empty()) {
        std::cerr << "Error: --image is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        LOG(INFO) << "Starting Neuriplo Client";
        LOG(INFO) << "Server: " << server_host << ":" << port;
        LOG(INFO) << "Image: " << image_path;
        
        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            LOG(ERROR) << "Failed to load image: " << image_path;
            return 1;
        }
        
        LOG(INFO) << "Image loaded: " << image.cols << "x" << image.rows;
        
        // Setup client (connects to server)
        auto client = setup_inference_engine(
            InferenceMode::CLIENT,
            server_host,
            false,  // use_gpu (ignored for client)
            1,      // batch_size (ignored for client)
            {},     // input_sizes (ignored for client)
            port
        );
        
        if (!client) {
            LOG(ERROR) << "Failed to initialize client";
            return 1;
        }
        
        LOG(INFO) << "Client connected to server";
        
        // Get model info from server
        auto model_info = client->get_model_info();
        LOG(INFO) << "Model info:";
        LOG(INFO) << "  Inputs: " << model_info.getInputs().size();
        for (const auto& input : model_info.getInputs()) {
            LOG(INFO) << "    - " << input.name << " shape: [" 
                     << input.shape[0] << ", " << input.shape[1] << ", " << input.shape[2] << "]";
        }
        LOG(INFO) << "  Outputs: " << model_info.getOutputs().size();
        for (const auto& output : model_info.getOutputs()) {
            LOG(INFO) << "    - " << output.name << " shape: [" << output.shape[0] << "]";
        }
        
        // Prepare input blob (you may need to preprocess according to your model)
        cv::Mat input_blob;
        cv::dnn::blobFromImage(image, input_blob, 1.0/255.0, cv::Size(224, 224), 
                              cv::Scalar(0, 0, 0), true, false);
        
        LOG(INFO) << "Running inference...";
        
        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        auto [outputs, shapes] = client->get_infer_results(input_blob);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        LOG(INFO) << "Inference completed in " << duration.count() << "ms";
        LOG(INFO) << "Backend inference time: " << client->get_last_inference_time_ms() << "ms";
        
        // Print results
        LOG(INFO) << "Results:";
        for (size_t i = 0; i < outputs.size(); ++i) {
            LOG(INFO) << "  Output " << i << " (shape: [";
            for (size_t j = 0; j < shapes[i].size(); ++j) {
                LOG(INFO) << shapes[i][j] << (j < shapes[i].size() - 1 ? ", " : "");
            }
            LOG(INFO) << "]):";
            
            // Print first few elements
            size_t max_print = std::min(outputs[i].size(), size_t(10));
            for (size_t j = 0; j < max_print; ++j) {
                std::visit([j](auto&& value) {
                    LOG(INFO) << "    [" << j << "] = " << value;
                }, outputs[i][j]);
            }
            if (outputs[i].size() > max_print) {
                LOG(INFO) << "    ... (" << outputs[i].size() - max_print << " more elements)";
            }
        }
        
        LOG(INFO) << "Client example completed successfully";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Fatal error: " << e.what();
        return 1;
    }
    
    return 0;
}
