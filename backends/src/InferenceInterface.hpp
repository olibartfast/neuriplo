#pragma once
#include "common.hpp"

#include <memory>
#include <stdexcept>
#include <variant>
using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

#include "BackendState.hpp"
#include "InferenceMetadata.hpp"

// Element type tags for raw typed output buffers. Values mirror
// neuriplo_dtype_t in include/neuriplo/plugin_abi.h. Distinct from
// TensorDataType (ITensorConverter.hpp), which describes wire formats that
// widen on decode (Int8 -> Int32, Bool -> UInt8).
enum class TensorDtype : uint8_t { FP32 = 0, INT32 = 1, INT64 = 2, UINT8 = 3 };

constexpr size_t tensor_dtype_size(TensorDtype dtype) noexcept {
    switch (dtype) {
    case TensorDtype::FP32:
    case TensorDtype::INT32:
        return 4;
    case TensorDtype::INT64:
        return 8;
    case TensorDtype::UINT8:
        return 1;
    }
    return 0;
}

// One inference output as a typed contiguous native-endian byte buffer.
struct RawOutputTensor {
    TensorDtype dtype = TensorDtype::FP32;
    std::vector<uint8_t> bytes;
    std::vector<int64_t> shape;

    size_t element_count() const noexcept { return bytes.size() / tensor_dtype_size(dtype); }
};

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
    explicit InferenceExecutionException(const std::string& message)
        : InferenceException("Inference execution failed: " + message) {}
};

class InferenceInterface {

  public:
    InferenceInterface(const std::string& weights, bool use_gpu = false, size_t batch_size = 1,
                       const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    virtual ~InferenceInterface() = default;

    // Core inference method - accepts vector of input tensors
    virtual std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) = 0;

    // Raw typed-buffer inference: outputs as contiguous bytes instead of
    // per-element TensorElement variants (~16 bytes/scalar). The default
    // adapts get_infer_results(); backends override it to copy framework
    // output buffers directly. Calling through this->get_infer_results()
    // keeps decorator augmentations applied on the raw path.
    virtual std::vector<RawOutputTensor> get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors);

    // Model information
    virtual InferenceMetadata get_inference_metadata();

    // Lifecycle (State pattern). Default behavior preserves the current
    // "constructed == ready" semantics: backends that load in their constructor
    // can leave these defaults untouched; load() is a no-op that marks Ready.
    virtual BackendState state() const noexcept { return state_; }
    virtual void load() { state_ = BackendState::Ready; }

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
    BackendState state_{BackendState::Uninitialized};
    std::string model_path_;
    bool gpu_available_;
    size_t batch_size_;
    double last_inference_time_ms_;
    size_t total_inferences_;

    // Utility methods

    // Input validation
    void validate_input(const std::vector<std::vector<uint8_t>>& input_tensors) const;
    void validate_model_loaded() const;

    // Performance tracking
    void start_timer();
    void end_timer();

    // Memory tracking
    mutable size_t memory_usage_mb_;

  private:
    std::chrono::high_resolution_clock::time_point inference_start_time_;
};