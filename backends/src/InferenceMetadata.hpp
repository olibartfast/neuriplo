#pragma once
#include "TensorDataType.hpp"

#include <string>
#include <vector>

struct LayerInfo {
    std::string name;
    std::vector<int64_t> shape;
    size_t batch_size;
    // Tensor element datatype. Defaults to Float32 so backends that do not report
    // a datatype keep their previous behaviour; backends that know the real type
    // (e.g. ONNX Runtime) populate it with any backend-supported element type so
    // non-FP32 tensors survive the serving/metadata boundary instead of being
    // silently treated as FP32.
    TensorDataType datatype{TensorDataType::Float32};
};

class InferenceMetadata {
  private:
    std::vector<LayerInfo> inputs;
    std::vector<LayerInfo> outputs;

  public:
    void addInput(const std::string& name, const std::vector<int64_t>& shape, size_t batch_size,
                  TensorDataType datatype = TensorDataType::Float32);
    void addOutput(const std::string& name, const std::vector<int64_t>& shape, size_t batch_size,
                   TensorDataType datatype = TensorDataType::Float32);
    const std::vector<LayerInfo>& getInputs() const;
    const std::vector<LayerInfo>& getOutputs() const;
};