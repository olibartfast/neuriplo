#pragma once
// The first-party inference engine.
//
// A self-contained runtime: it reads an ONNX model, builds a typed graph,
// infers shapes, plans memory, and executes the graph with CPU reference
// kernels. It stands on its own — nothing here includes or links the backend
// abstraction layer, and the build enforces that separation.

namespace engine {

// Placeholder so the library has a symbol while the runtime is built out.
// Replaced by the executor once it exists.
class Engine {
public:
    Engine() = default;
};

} // namespace engine
