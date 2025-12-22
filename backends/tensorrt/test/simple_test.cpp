#include <iostream>
#include <memory>
#include "TRTInfer.hpp"

int main() {
    try {
        std::cout << "Starting TensorRT test..." << std::endl;
        
        // Try to create TRTInfer object
        std::cout << "Creating TRTInfer object..." << std::endl;
        auto infer = std::make_unique<TRTInfer>("resnet18.engine", true);
        
        std::cout << "TRTInfer object created successfully!" << std::endl;
        
        // Try to get inference metadata
        std::cout << "Getting inference metadata..." << std::endl;
        auto inference_metadata = infer->get_inference_metadata();
        std::cout << "Inference metadata retrieved successfully!" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
} 