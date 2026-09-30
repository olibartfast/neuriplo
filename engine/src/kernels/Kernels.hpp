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

// Copy of the input flat buffer in order; verifies an optional int64 shape.
void Reshape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Mean over the given axes (or all axes when none are supplied).
void ReduceMean(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

// Returns the kernel for `op_type`, or null when the CPU device has no
// implementation for it.
KernelFn FindCpuKernel(const std::string& op_type);

} // namespace kernels
} // namespace engine
