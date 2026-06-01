#pragma once

#include "BackendState.hpp"
#include "InferenceInterface.hpp"

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

// Bridge pattern: ModelRunner is the abstraction, InferenceInterface is the
// implementor. ModelRunner owns a backend and delegates all work to it; it
// contains no backend-specific logic and only orchestrates lifecycle and
// forwards high-level calls.
class ModelRunner {
  public:
    explicit ModelRunner(std::unique_ptr<InferenceInterface> backend);

    // Lifecycle orchestration. Idempotent: returns early when already Ready.
    void load();
    BackendState state() const noexcept { return backend_->state(); }

    // High-level run API: ensure the backend is loaded, then run inference.
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    run(const std::vector<std::vector<uint8_t>>& input_tensors);

    // Convenience delegators.
    InferenceMetadata get_inference_metadata() { return backend_->get_inference_metadata(); }
    bool is_gpu_available() const noexcept { return backend_->is_gpu_available(); }
    size_t get_batch_size() const noexcept { return backend_->get_batch_size(); }
    std::string get_model_path() const noexcept { return backend_->get_model_path(); }
    double get_last_inference_time_ms() const noexcept { return backend_->get_last_inference_time_ms(); }
    size_t get_total_inferences() const noexcept { return backend_->get_total_inferences(); }
    void clear_cache() noexcept { backend_->clear_cache(); }
    size_t get_memory_usage_mb() const noexcept { return backend_->get_memory_usage_mb(); }

    InferenceInterface* backend() const noexcept { return backend_.get(); }

  private:
    std::unique_ptr<InferenceInterface> backend_;
};
