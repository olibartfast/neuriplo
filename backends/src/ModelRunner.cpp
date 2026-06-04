#include "ModelRunner.hpp"

ModelRunner::ModelRunner(std::unique_ptr<InferenceInterface> backend) : backend_(std::move(backend)) {
    if (!backend_) {
        throw InferenceException("ModelRunner requires a non-null backend");
    }
}

void ModelRunner::load() {
    const auto current_state = backend_->state();
    if (current_state == BackendState::Ready) {
        return;
    }
    if (current_state == BackendState::Failed) {
        throw ModelLoadException(
            "backend is in Failed state; reconstruct the backend or transition Failed -> Loading before reload");
    }

    backend_->load();
    const auto loaded_state = backend_->state();
    if (loaded_state != BackendState::Ready) {
        throw ModelLoadException("backend is not Ready after load() (state: " + std::string(to_string(loaded_state)) +
                                 ")");
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
ModelRunner::run(const std::vector<std::vector<uint8_t>>& input_tensors) {
    const auto current_state = backend_->state();
    if (current_state == BackendState::Failed) {
        throw InferenceExecutionException(
            "backend is in Failed state; reconstruct the backend or transition Failed -> Loading before retry");
    }
    if (current_state != BackendState::Ready) {
        load();
    }
    return backend_->get_infer_results(input_tensors);
}
