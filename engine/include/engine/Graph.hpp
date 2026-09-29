#pragma once
// Graph IR for the first-party native engine (Phase N0, Group 2a).
//
// Minimal typed intermediate representation covering exactly what the
// [R-5] op set (opset 18) needs for the ResNet-18 fixture graph. Loaded by
// LoadGraphFromFile() (see ModelLoader.hpp); the protobuf wire decoding is
// isolated in the private WireReader under engine/src/.
//
// Boundary contract ([R-1]): this header must not include or reference the
// backend abstraction layer.

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace engine {

// Raised for any malformed/unreadable model file and for graphs the loader
// cannot accept ([V-3] asserts the message carries the node name and op type).
class ModelLoadException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Tensor element type. Only what [R-3] (static, float32-only) needs today;
// everything else decodes as Unknown and is rejected later with context.
enum class DataType {
    Float32,
    Unknown,
};

// Declared tensor metadata from the model's graph I/O / value-info table.
struct TensorInfo {
    DataType dtype = DataType::Unknown;
    std::vector<int64_t> dims;
};

// Attribute values, covering EXACTLY the [A-1] attribute sets:
//   Conv       : pads, strides, dilations, group, auto_pad
//   MaxPool    : kernel_shape, pads, strides, dilations, ceil_mode,
//                storage_order, auto_pad
//   ReduceMean : keepdims, noop_with_empty_axes
//   Reshape    : allowzero
//   Gemm       : alpha, beta, transA, transB
//
// Int lists model both int lists (ints) and int64 lists (longs); float
// scalars/sub-graphs are not named in the loader interface and use their
// natural protobuf coding (double for singles, strings for text).
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

// A single graph node: op name, op type, edges, and its decoded attributes.
struct Node {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<Attribute> attributes;
};

// Whole-model graph: execution order, value-info table, embedded float
// initializers, and declared I/O shapes ([V-2] asserts node count, op types,
// initializer count, and declared input/output shapes from these).
struct Graph {
    std::vector<Node> nodes;
    std::map<std::string, TensorInfo> tensors;              // value table
    std::map<std::string, std::vector<float>> initializers; // embedded weights
    std::vector<std::pair<std::string, TensorInfo>> inputs;
    std::vector<std::pair<std::string, TensorInfo>> outputs;
};

} // namespace engine
