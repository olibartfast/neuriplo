#pragma once
// Private CPU kernel declarations.
//
// Every kernel matches the KernelFn calling convention fixed in Device.hpp: it
// reads the node's attributes and the input views and writes the output views,
// owning no storage. FindCpuKernel maps an op type to its implementation. This
// header is private to engine/src/kernels; nothing outside the engine includes
// it, and it must not reference the backend abstraction layer.

#include <string>
#include <vector>

#include "engine/Device.hpp"

namespace engine {
namespace kernels {

// Elementwise max(0, x) over the output element count.
void Relu(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// NumPy multidirectional broadcast of the two inputs into the output shape.
void Add(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// NumPy multidirectional broadcast of the two inputs into the output shape.
void Mul(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// NumPy multidirectional broadcast of the two inputs into the output shape;
// division by zero follows IEEE float semantics and never throws.
void Div(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// NumPy multidirectional broadcast of the two inputs into the output shape.
void Sub(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Elementwise 1 / (1 + exp(-x)) over the output element count.
void Sigmoid(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Element conversion between float32, int64, and bool per the `to` attribute.
void Cast(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Numerically stable softmax over one `axis` (defaults to -1).
void Softmax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Maximum over the requested axes (or all axes when none are supplied).
void ReduceMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Truncating integer remainder with NumPy multidirectional broadcasting.
void Mod(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Elementwise comparison with NumPy multidirectional broadcasting; bool output.
void Equal(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Ternary selection with NumPy multidirectional broadcasting across all three.
void Where(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Copy of the input flat buffer in order; verifies an optional int64 shape.
void Reshape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Mean over the given axes (or all axes when none are supplied).
void ReduceMean(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// General matrix multiplication with optional transposes, alpha/beta scaling,
// and an optional rank-1 or rank-2 bias.
void Gemm(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// ONNX matrix product with 1-D operand promotion and batch broadcasting.
void MatMul(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// NCHW convolution with groups, strides, dilations, and explicit or automatic
// padding, plus an optional rank-1 bias.
void Conv(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// NCHW max pooling with a required kernel_shape, strides, dilations, explicit
// or automatic padding, and ceil_mode. Only the primary output is written.
void MaxPool(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Returns the kernel for `op_type`, or null when the CPU device has no
// implementation for it.
KernelFn FindCpuKernel(const std::string& op_type);

// Join the inputs along `axis` (negative values count from the back);
// float32 or int64 data.
void Concat(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Divide the input along `axis` into one output per part, sized by the
// optional int64 sizes input or evenly; float32 or int64 data.
void Split(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Insert size-1 axes at the int64 axes-input positions; float32 or int64 data.
void Unsqueeze(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Broadcast the input to the int64 shape input; float32 or int64 data.
void Expand(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Permute the input axes per `perm` (absent means reverse); float32 or int64.
void Transpose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Strided ranges from the int64 starts/ends (and optional axes/steps) inputs;
// float32 or int64 data.
void Slice(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Whole-slice selection along `axis` with int64 indices; float32 or int64 data.
void Gather(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Per-element selection along `axis` with int64 indices; float32 or int64 data.
void GatherElements(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Nearest-neighbor upsampling from the float32 scales input; float32 or int64.
void Resize(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Collapse to [prod(dims[:axis]), prod(dims[axis:])]; float32 or int64 data.
void Flatten(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Emit the input dims as a rank-1 int64 tensor.
void Shape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Fill an int64 tensor of the shape-input dims with the `value` attribute.
void ConstantOfShape(const Node& node, const std::vector<TensorView>& inputs,
    const std::vector<TensorView>& outputs);

// K largest elements along `axis`, descending; float32 values, int64 indices.
void TopK(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

} // namespace kernels
} // namespace engine
