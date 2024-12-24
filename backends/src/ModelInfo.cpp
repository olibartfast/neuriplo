#include  "ModelInfo.hpp"

void ModelInfo::addInput(const std::string& name, const std::vector<int64_t>& shape) {
    inputs.push_back({name, shape});
}

void ModelInfo::addOutput(const std::string& name, const std::vector<int64_t>& shape) {
    outputs.push_back({name, shape});
}   

const std::vector<LayerInfo>& ModelInfo::getInputs() const {
    return inputs;
}

const std::vector<LayerInfo>& ModelInfo::getOutputs() const {
    return outputs;
}
