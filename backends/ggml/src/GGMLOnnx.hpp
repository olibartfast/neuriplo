#pragma once

#include "GGMLGraph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Minimal ONNX (protobuf) importer that emits a ggml_graph::Graph.
//
// Supported scope is deliberately small and fails loudly outside it:
//   * static shapes, rank 1..4;
//   * FLOAT/DOUBLE/FLOAT16, INT32 and INT64 tensors (F16/DOUBLE are widened);
//   * graph inputs with FLOAT/INT32/INT64;
//   * ops: Add, Sub, Mul, Div, Sqrt, Clip, Conv, MaxPool, MatMul, Gemm, Relu,
//     Sigmoid, Softmax, LayerNormalization, Gather, Reshape, Transpose,
//     Concat, Squeeze, Unsqueeze, Flatten, Cast, Identity, Constant, Shape.
//
// ONNX shapes are returned in ONNX order. The graph itself stores reversed
// axes, matching GGMLInfer's NCHW -> [W,H,C,N] convention.
namespace ggml_onnx {

struct Input {
    std::string name;
    ggml_graph::DType dtype;
    std::vector<int64_t> shape; // ONNX order
};

struct Model {
    ggml_graph::Graph graph;
    std::vector<Input> inputs;
    std::vector<std::string> output_names;
};

std::unique_ptr<Model> load(const std::string& path);
std::unique_ptr<Model> parse(const std::vector<uint8_t>& bytes, const std::string& label);
std::unique_ptr<Model> parse(const uint8_t* data, size_t size, const std::string& label);

} // namespace ggml_onnx
