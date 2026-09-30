#pragma once
// The engine's graph representation.
//
// A loaded model becomes a typed graph: nodes in execution order, a table of
// every tensor's type and shape, the embedded constants, and the declared
// inputs and outputs. It covers exactly the operators the reference model
// needs; compute is float32, with int64 constants allowed for shape inputs.
//
// Must not include or reference the backend abstraction layer.

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace engine {

// Raised for any malformed or unreadable model file, and for graphs the loader
// cannot accept. The message carries the offending node name and op type.
class ModelLoadException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Tensor element type. Compute is float32. Int64 appears only for embedded
// constant tensors that carry shape indices (Reshape's shape input); anything
// else decodes as Unknown and is rejected later with context.
enum class DataType {
    Float32,
    Int64,
    Unknown,
};

// The declared type and shape of a tensor, from the model's input, output, and
// value-info entries.
struct TensorInfo {
    DataType dtype = DataType::Unknown;
    std::vector<int64_t> dims;
};

// An operator attribute value. Covers the kinds the supported operators use:
// integer and float scalars, strings, and integer or float lists. Other kinds
// are rejected at load.
struct Attribute {
    std::string name;
    std::variant<
        int64_t,                   // int/int64/bool scalars
        float,                     // float scalar
        std::string,               // string scalar (auto_pad)
        std::vector<int64_t>,      // ints list (pads, strides, dilations, ...)
        std::vector<float>         // float list
        >
        value;

    Attribute() = default;
    explicit Attribute(std::string n, decltype(value) v)
        : name(std::move(n)), value(std::move(v)) {}
};

// A single graph node: its operator type, the tensors it consumes and
// produces, and its decoded attributes.
struct Node {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<Attribute> attributes;
};

// An embedded constant tensor. Weights are float32; integer constants carry
// shape indices. `dtype` selects which alternative of `values` is meaningful.
struct Initializer {
    DataType dtype = DataType::Unknown;
    std::vector<int64_t> dims;
    std::variant<std::vector<float>, std::vector<int64_t>> values;
};

// A whole model: nodes in execution order, the type-and-shape table for every
// named value, the embedded constants, and the declared inputs and outputs.
struct Graph {
    std::vector<Node> nodes;
    std::map<std::string, TensorInfo> tensors;            // value table
    std::map<std::string, Initializer> initializers;      // embedded constants
    std::vector<std::pair<std::string, TensorInfo>> inputs;
    std::vector<std::pair<std::string, TensorInfo>> outputs;
};

} // namespace engine
