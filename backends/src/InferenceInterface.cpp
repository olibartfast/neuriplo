#include "InferenceInterface.hpp"

#include <cstdint>

InferenceInterface::InferenceInterface(const std::string& weights, bool use_gpu, size_t batch_size,
                                       const std::vector<std::vector<int64_t>>& input_sizes)
    : model_path_(weights), gpu_available_(use_gpu), batch_size_(batch_size), last_inference_time_ms_(0.0),
      total_inferences_(0), memory_usage_mb_(0) {}

std::vector<RawOutputTensor>
InferenceInterface::get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) {
    auto [outputs, shapes] = get_infer_results(input_tensors);

    std::vector<RawOutputTensor> raw_outputs;
    raw_outputs.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        RawOutputTensor tensor;
        if (i < shapes.size()) {
            tensor.shape = std::move(shapes[i]);
        }

        const std::vector<TensorElement>& elements = outputs[i];
        if (elements.empty()) {
            raw_outputs.push_back(std::move(tensor));
            continue;
        }

        const size_t alternative = elements.front().index();
        auto flatten = [&](auto sample, TensorDtype tag) {
            using Element = decltype(sample);
            tensor.dtype = tag;
            tensor.bytes.resize(elements.size() * sizeof(Element));
            auto* typed = reinterpret_cast<Element*>(tensor.bytes.data());
            for (size_t j = 0; j < elements.size(); ++j) {
                if (elements[j].index() != alternative) {
                    throw InferenceExecutionException("output tensor mixes element types");
                }
                typed[j] = std::get<Element>(elements[j]);
            }
        };
        switch (alternative) {
        case 0:
            flatten(float{}, TensorDtype::FP32);
            break;
        case 1:
            flatten(int32_t{}, TensorDtype::INT32);
            break;
        case 2:
            flatten(int64_t{}, TensorDtype::INT64);
            break;
        case 3:
            flatten(uint8_t{}, TensorDtype::UINT8);
            break;
        default:
            throw InferenceExecutionException("unsupported output element type");
        }
        raw_outputs.push_back(std::move(tensor));
    }
    return raw_outputs;
}

InferenceMetadata InferenceInterface::get_inference_metadata() {
    // OpenCV DNN module does not have a method to get input layer shapes and names
    if (inference_metadata_.getInputs().empty() && inference_metadata_.getOutputs().empty()) {
        throw ModelLoadException("Model information is not available - inputs and outputs are empty");
    }
    return inference_metadata_;
}

void InferenceInterface::clear_cache() noexcept {
    // Default implementation - do nothing
    // Derived classes can override this if they need cache management
}

size_t InferenceInterface::get_memory_usage_mb() const noexcept {
    // Default implementation - return 0
    // Derived classes can override this to provide actual memory usage
    return 0;
}

void InferenceInterface::start_timer() { inference_start_time_ = std::chrono::high_resolution_clock::now(); }

void InferenceInterface::end_timer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - inference_start_time_);
    last_inference_time_ms_ = duration.count() / 1000.0;
    total_inferences_++;
}

void InferenceInterface::validate_model_loaded() const {
    if (model_path_.empty()) {
        throw ModelLoadException("Model path is not specified");
    }
}

void InferenceInterface::validate_input(const std::vector<std::vector<uint8_t>>& input_tensors) const {
    validate_model_loaded();

    if (input_tensors.empty()) {
        throw InferenceExecutionException("No input tensors provided");
    }

    if (input_tensors.size() != inference_metadata_.getInputs().size()) {
        throw InferenceExecutionException("Input tensor count mismatch: expected " +
                                          std::to_string(inference_metadata_.getInputs().size()) + ", got " +
                                          std::to_string(input_tensors.size()));
    }

    for (size_t i = 0; i < input_tensors.size(); ++i) {
        if (input_tensors[i].empty()) {
            throw InferenceExecutionException("Input tensor at index " + std::to_string(i) + " is empty");
        }
    }
}
