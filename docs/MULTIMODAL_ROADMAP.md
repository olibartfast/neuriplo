# Multimodal Convergence Roadmap

## Goal

Converge the existing `origin/copilot/llamacpp-add-backend-support` and `origin/feature/cactus-compute` work into one coherent multimodal backend plan for Neuriplo without breaking existing backend selection, fallback behavior, or device placement assumptions.

This branch is intentionally roadmap-only. It documents the integration path before any inference-path merge work happens.

## Current State

Both feature branches follow the same broad backend integration pattern:

- add a backend implementation under `backends/<backend>/src/`
- register the backend in `CMakeLists.txt`
- wire backend selection and linking through `cmake/`
- expose the backend through `include/InferenceBackendSetup.hpp` and `src/InferenceBackendSetup.cpp`
- extend validation, test setup, Docker, CI, and backend test scripts

The main structural difference is runtime dependency and execution model:

- `llamacpp` uses the `llama.cpp` C API and targets LLM and multimodal inference support
- `cactus` uses the `cactus` API and currently follows a similar prompt-in / generated-text-out contract

## Convergence Principles

- Keep Neuriplo's public backend-selection behavior unchanged.
- Do not merge backend-specific inference logic until the shared backend contract is clear.
- Preserve CPU/GPU placement semantics per backend instead of forcing one cross-backend assumption.
- Preserve fallback behavior when optional runtimes are absent.
- Reuse one test and packaging pattern wherever backend behavior is materially identical.

## Target End State

Neuriplo should expose `LLAMACPP` and `CACTUS` as multimodal-capable backends that share one integration pattern and one validation strategy, while still allowing backend-specific runtime options behind the scenes.

The converged shape should include:

- one shared definition of "multimodal backend" expectations inside Neuriplo
- parallel CMake, Docker, CI, and script wiring for both backends
- aligned smoke tests for prompt decoding, batch handling, and output tensor formatting
- backend-specific runtime adapters for library initialization, model loading, token generation, and optional accelerator use

## Proposed Workstream

### Phase 1: Normalize the Branch Surfaces

Before merging either backend branch, align the non-runtime file layout and naming so the two branches touch equivalent integration points:

- keep backend directories parallel: `backends/llamacpp/` and `backends/cactus/`
- keep CMake modules parallel: `cmake/LlamaCpp.cmake` and `cmake/Cactus.cmake`
- keep Docker runners parallel: `docker/Dockerfile.llamacpp`, `docker/Dockerfile.cactus`, matching `run_*_tests.sh`
- keep test model generation and test binary naming parallel

Deliverable:

- a no-behavior-change cleanup branch that makes both backends structurally symmetric where possible

### Phase 2: Extract a Shared Multimodal Backend Contract

The two branches already mirror each other at the class boundary. Capture that similarity explicitly instead of duplicating assumptions in two implementations.

Suggested contract areas:

- prompt bytes to backend input conversion
- generated response to `TensorElement` conversion
- batch-size handling rules
- model-loaded state and initialization failure behavior
- backend capability flags: text-only, vision-text, accelerator support

Candidate implementation shape:

- add a lightweight shared helper or abstract adapter under `backends/src/` for prompt/result conversion rules
- keep backend library calls in backend-local source files

Deliverable:

- one preparatory refactor branch that introduces shared helpers without changing runtime results

### Phase 3: Align Dependency Validation and Configuration

Both branches add similar configuration surfaces. Unify the policy even if the actual dependency checks remain backend-specific.

Converge these areas:

- `versions.env`
- `cmake/versions.cmake`
- `cmake/DependencyValidation.cmake`
- `scripts/setup_dependencies.sh`
- `scripts/test_backends.sh`
- CI matrix entries

Required behavior:

- optional backends stay optional
- missing runtime dependencies fail clearly at configure or test time
- backend enablement does not change existing defaults

Deliverable:

- one infra branch that aligns validation and test hooks for both backends

### Phase 4: Unify Test Semantics

Do not assume the same runtime internals, but do require the same observable backend contract from tests.

Shared test semantics should cover:

- model load success and failure paths
- deterministic handling of empty prompt input
- batch input acceptance or explicit rejection
- stable output tensor shape conventions
- safe cleanup on destructor and failed initialization

Backend-specific tests can then cover:

- `llamacpp` sampler and context behavior
- `cactus` runtime-specific generation and compute-path behavior

Deliverable:

- aligned test cases with common assertions and backend-specific extensions

### Phase 5: Merge in Order

Recommended merge order:

1. structural normalization
2. shared contract extraction
3. dependency and CI alignment
4. backend-specific runtime rebases
5. final multimodal documentation pass
6. close superseded source feature branches

This reduces merge conflict pressure in `cmake/`, `scripts/`, and `InferenceBackendSetup` files.

### Phase 6: Close Source Feature Branches

After the converged multimodal work is merged and validated on the integration branch, close the source feature branches so they stop attracting new changes.

Branches to close after convergence:

- `origin/copilot/llamacpp-add-backend-support`
- `origin/feature/cactus-compute`

Closure criteria:

- the converged branch has landed on `develop`
- multimodal tests and dependency validation pass in the merged state
- any backend-specific follow-up work has been re-opened as new branches against the converged contract instead of continuing on the old feature branches

Recommended follow-up:

- lock or delete the old feature branches in the remote once their content is superseded
- direct all new work to the converged multimodal branch line

## Risks

- Both branches modify the same backend registration surfaces, so direct merging will create avoidable conflicts.
- The two runtimes may not agree on GPU offload semantics; forcing uniform behavior too early would violate repo constraints.
- "Multimodal" may mean different capability sets in each runtime. Capability discovery should be explicit rather than inferred from backend name.
- Test assets may drift independently unless one shared naming and generation convention is enforced.

## Recommended Next Branches

- `feature/multimodal-contract`
- `feature/multimodal-dependency-alignment`
- `feature/llamacpp-rebase-on-contract`
- `feature/cactus-rebase-on-contract`
- `feature/multimodal-merge`

## Source Branches Reviewed

- `origin/copilot/llamacpp-add-backend-support`
- `origin/feature/cactus-compute`
