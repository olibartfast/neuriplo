#pragma once
#include "InferenceInterface.hpp"

#include <memory>
#include <utility>

// Decorator base class for InferenceInterface backends.
//
// Wraps an existing backend and forwards every virtual method to it. Subclasses
// (e.g. ProfilingBackend) override only the methods they augment, relying on the
// default pass-through behavior for everything else.
class BackendDecorator : public InferenceInterface {

  public:
    explicit BackendDecorator(std::unique_ptr<InferenceInterface> inner)
        : InferenceInterface(inner ? inner->get_model_path() : std::string(), inner ? inner->is_gpu_available() : false,
                             inner ? inner->get_batch_size() : 1, std::vector<std::vector<int64_t>>()) {
        if (!inner) {
            throw InferenceException("BackendDecorator requires a non-null inner backend");
        }
        inner_ = std::move(inner);
    }

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override {
        return inner_->get_infer_results(input_tensors);
    }

    InferenceMetadata get_inference_metadata() override { return inner_->get_inference_metadata(); }

    BackendState state() const noexcept override { return inner_->state(); }
    void load() override { inner_->load(); }

    bool is_gpu_available() const noexcept override { return inner_->is_gpu_available(); }
    size_t get_batch_size() const noexcept override { return inner_->get_batch_size(); }
    std::string get_model_path() const noexcept override { return inner_->get_model_path(); }

    double get_last_inference_time_ms() const noexcept override { return inner_->get_last_inference_time_ms(); }
    size_t get_total_inferences() const noexcept override { return inner_->get_total_inferences(); }

    void clear_cache() noexcept override { inner_->clear_cache(); }
    size_t get_memory_usage_mb() const noexcept override { return inner_->get_memory_usage_mb(); }

    InferenceInterface* inner() const noexcept { return inner_.get(); }

  protected:
    std::unique_ptr<InferenceInterface> inner_;
};
