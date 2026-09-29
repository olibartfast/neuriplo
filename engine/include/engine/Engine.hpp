#pragma once
// First-party inference engine (CPU reference spine).
//
// Phase N0 Group 1: skeleton only. The engine intentionally has no public API
// surface beyond this stub yet; Groups 2-5 fill in ONNX loading, graph IR,
// shape inference, kernels, and the executor per plan.md.
//
// Boundary contract ([R-1]): this directory must not include or link the
// backend abstraction layer. The configure-time guard in the CMakeLists.txt
// enforces this.

namespace engine {

// Placeholder translation-unit anchor so the static library is non-empty and
// the standalone build is linkable. Replaced by the executor in Group 5.
class Engine {
public:
    Engine() = default;
};

} // namespace engine
