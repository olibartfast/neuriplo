#include "ModelRunner.hpp"

ModelRunner::ModelRunner(std::unique_ptr<InferenceInterface> backend) : backend_(std::move(backend)) {
    if (!backend_) {
        throw InferenceException("ModelRunner requires a non-null backend");
    }
}

void ModelRunner::load() {
    if (backend_->state() == BackendState::Ready) {
        return;
    }
    backend_->load();
    if (backend_->state() != BackendState::Ready) {
        throw ModelLoadException(
            "backend is not Ready after load() (state: " + std::string(to_string(backend_->state())) + ")");
    }
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
ModelRunner::run(const std::vector<std::vector<uint8_t>>& input_tensors) {
    if (backend_->state() != BackendState::Ready) {
        load();
    }
    return backend_->get_infer_results(input_tensors);
}
