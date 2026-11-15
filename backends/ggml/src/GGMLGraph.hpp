#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Small programmatic multi-input graph executor on top of ggml.
//
// This is the execution substrate for a future ONNX-subset importer, not an
// ONNX parser itself. Shapes are ggml dimensions (ne[0] first); callers that
// import ONNX must reverse the ONNX axes.
namespace ggml_graph {

enum class DType { F32, I32, I64 };

enum class Op {
    Add,
    Sub,
    Mul,
    Div,
    IntegerBinary,
    Sqrt,
    Clip,
    MatMul,
    BatchedMatMul,
    BatchedMatMulTransposed,
    Linear,
    ScaledDotProductAttention,
    Softmax,
    LayerNorm,
    Gelu,
    Gather,
    CastI64ToI32,
    CastI32ToF32,
    Reshape,
    Transpose,
    Concat,
    Slice,
    Relu,
    Sigmoid,
    Abs,
    Log,
    Erf,
    Square,
    ReduceMean,
    ReduceSum,
    ReduceMax,
    TopKIndices,
    TopKValues,
    GatherElements,
    GridSample,
    Pad,
    Interpolate,
    Repeat,
    Conv2d,
    Conv2dDepthwise,
    MaxPool2d,
};

struct Output {
    DType dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

class Graph {
  public:
    Graph();
    ~Graph();
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    int add_input(const std::string& name, DType dtype, std::vector<int64_t> shape);
    int add_const_f32(const std::string& name, std::vector<int64_t> shape, const std::vector<float>& data);
    int add_const_i32(const std::string& name, std::vector<int64_t> shape, const std::vector<int32_t>& data);

    // `attrs` is op-specific: LayerNorm uses attrs[0] as eps (default 1e-5).
    int add_op(Op op, const std::vector<int>& inputs, const std::vector<float>& attrs = {});

    void mark_output(int tensor_id);

    // `input_blobs` must match the inputs added, in order, with exact byte
    // sizes for their dtype. Returns every marked output in mark order.
    std::vector<Output> compute(const std::vector<std::vector<uint8_t>>& input_blobs);

    // Actual dtype of a tensor id. Useful for importers that preserve dtype
    // through reshape/transpose/concat/slice.
    DType dtype(int tensor_id) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ggml_graph
