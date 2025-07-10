#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include "Logger.hpp"

int main() {
    try {
        std::cout << "Starting minimal TensorRT test..." << std::endl;
        
        // Create TensorRT runtime
        Logger logger;
        auto runtime = nvinfer1::createInferRuntime(logger);
        if (!runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return 1;
        }
        
        // Load engine file
        std::ifstream engine_file("resnet18.engine", std::ios::binary);
        if (!engine_file) {
            std::cerr << "Failed to open engine file" << std::endl;
            return 1;
        }
        
        engine_file.seekg(0, std::ios::end);
        size_t file_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(file_size);
        engine_file.read(engine_data.data(), file_size);
        engine_file.close();
        
        // Deserialize engine
        auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_data.data(), file_size));
        if (!engine) {
            std::cerr << "Failed to deserialize engine" << std::endl;
            return 1;
        }
        
        std::cout << "Engine loaded successfully!" << std::endl;
        
        // Create execution context
        auto context = engine->createExecutionContext();
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return 1;
        }
        
        std::cout << "Context created successfully!" << std::endl;
        
        // Get tensor information
        int num_tensors = engine->getNbIOTensors();
        std::cout << "Number of tensors: " << num_tensors << std::endl;
        
        for (int i = 0; i < num_tensors; ++i) {
            std::string tensor_name = engine->getIOTensorName(i);
            auto dims = engine->getTensorShape(tensor_name.c_str());
            auto data_type = engine->getTensorDataType(tensor_name.c_str());
            auto io_mode = engine->getTensorIOMode(tensor_name.c_str());
            
            std::cout << "Tensor " << i << ": " << tensor_name 
                      << " (IO: " << (io_mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << ")"
                      << " Shape: [";
            for (int j = 0; j < dims.nbDims; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << dims.d[j];
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "Test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
} 