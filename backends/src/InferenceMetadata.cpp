#include "InferenceMetadata.hpp"

void InferenceMetadata::addInput(const std::string& name, const std::vector<int64_t>& shape, size_t batch_size,
                                 TensorDataType datatype) {
    inputs.push_back({name, shape, batch_size, datatype});
}

void InferenceMetadata::addOutput(const std::string& name, const std::vector<int64_t>& shape, size_t batch_size,
                                  TensorDataType datatype) {
    outputs.push_back({name, shape, batch_size, datatype});
}

const std::vector<LayerInfo>& InferenceMetadata::getInputs() const { return inputs; }

const std::vector<LayerInfo>& InferenceMetadata::getOutputs() const { return outputs; }
