// First-party inference engine (CPU reference spine) — skeleton source.
//
// Phase N0 Group 1: intentionally minimal so the standalone configure/build
// and the abstract integrated build both stay green. Groups 2-5 add the model
// reader, graph IR, shape inference, planner, kernels, and executor.

#include "engine/Engine.hpp"

namespace engine {

// Anchor definition so the static library is non-empty. Removed once the
// executor lands in Group 5.
void engine_spine_anchor()
{
}

} // namespace engine
