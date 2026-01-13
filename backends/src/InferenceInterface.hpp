#pragma once
#include "common.hpp"
#include <variant>
#include <stdexcept>
#include <memory>
using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

#include "InferenceMetadata.hpp"

// Custom exceptions for better error handling
class InferenceException : public std::runtime_error {
public:
    explicit InferenceException(const std::string& message) : std::runtime_error(message) {}
};

class ModelLoadException : public InferenceException {
public:
    explicit ModelLoadException(const std::string& message) : InferenceException("Model loading failed: " + message) {}
};

class InferenceExecutionException : public InferenceException {
public:
    explicit InferenceExecutionException(const std::string& message) : InferenceException("Inference execution failed: " + message) {}
};

class InferenceInterface{
    	
    public:
        InferenceInterface(const std::string& weights,
         bool use_gpu = false, 
         size_t batch_size = 1,
         const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());
        
        virtual ~InferenceInterface() = default;
        
        // Core inference method - accepts vector of input tensors
        virtual std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> 
        get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) = 0;
        
        // Model information
        virtual InferenceMetadata get_inference_metadata();
        
        // Utility methods
        virtual bool is_gpu_available() const noexcept { return gpu_available_; }
        virtual size_t get_batch_size() const noexcept { return batch_size_; }
        virtual std::string get_model_path() const noexcept { return model_path_; }
        
        // Performance monitoring
        virtual double get_last_inference_time_ms() const noexcept { return last_inference_time_ms_; }
        virtual size_t get_total_inferences() const noexcept { return total_inferences_; }
        
        // Memory management
        virtual void clear_cache() noexcept;
        virtual size_t get_memory_usage_mb() const noexcept;

    protected:
        InferenceMetadata inference_metadata_;
        std::string model_path_;
        bool gpu_available_;
        size_t batch_size_;
        double last_inference_time_ms_;
        size_t total_inferences_;
        
        // Utility methods
        std::vector<float> blob2vec(const cv::Mat& input_blob);
        
        // Input validation
        void validate_model_loaded() const;
        
        // Performance tracking
        void start_timer();
        void end_timer();
        
        // Memory tracking
        mutable size_t memory_usage_mb_;

    private:
        std::chrono::high_resolution_clock::time_point inference_start_time_;
};