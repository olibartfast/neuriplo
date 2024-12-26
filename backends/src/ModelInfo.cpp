#include  "ModelInfo.hpp"

void ModelInfo::addInput(const std::string& name, const std::vector<int64_t>& shape, size_t batch_size) {
    inputs.push_back({name, shape, batch_size});
}

void ModelInfo::addOutput(const std::string& name, const std::vector<int64_t>& shape, size_t batch_size) {
    outputs.push_back({name, shape, batch_size});
}   

const std::vector<LayerInfo>& ModelInfo::getInputs() const {
    return inputs;
}

const std::vector<LayerInfo>& ModelInfo::getOutputs() const {
    return outputs;
}
