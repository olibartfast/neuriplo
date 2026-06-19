#pragma once
#include "BackendDecorator.hpp"
#include "InferenceInterface.hpp"

#include <chrono>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

// Decorator that measures wall-clock inference time for the wrapped backend.
//
// Wraps any InferenceInterface and times each get_infer_results() call,
// exposing its own measured timings via get_last_inference_time_ms() and
// get_total_inferences(). The numeric inference output is forwarded unchanged;
// this decorator only augments timing/instrumentation.
class ProfilingBackend : public BackendDecorator {

  public:
    explicit ProfilingBackend(std::unique_ptr<InferenceInterface> inner) : BackendDecorator(std::move(inner)) {}

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override {
        const auto start = std::chrono::high_resolution_clock::now();
        auto result = BackendDecorator::get_infer_results(input_tensors);
        const auto end = std::chrono::high_resolution_clock::now();

        // Only record on success: an exception above propagates without
        // updating the counters, so we never double count or record partial work.
        last_inference_time_ms_local_ = std::chrono::duration<double, std::milli>(end - start).count();
        total_inference_time_ms_local_ += last_inference_time_ms_local_;
        total_inferences_local_ += 1;

        return result;
    }

    double get_last_inference_time_ms() const noexcept override { return last_inference_time_ms_local_; }

    size_t get_total_inferences() const noexcept override { return total_inferences_local_; }

    double average_inference_time_ms() const {
        if (total_inferences_local_ == 0) {
            return 0.0;
        }
        return total_inference_time_ms_local_ / static_cast<double>(total_inferences_local_);
    }

  private:
    double last_inference_time_ms_local_{0.0};
    double total_inference_time_ms_local_{0.0};
    size_t total_inferences_local_{0};
};
