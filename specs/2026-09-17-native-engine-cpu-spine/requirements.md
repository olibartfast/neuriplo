# Requirements — Native Engine: CPU Reference Spine

Spec: `specs/2026-09-17-native-engine-cpu-spine/requirements.md` ·
Branch: `feature/native-engine-cpu-spine` · Roadmap: Native Engine Track, Phase N0

## Goal

Stand up a self-contained inference runtime inside this repository — an ONNX
parser, graph IR, shape inference, memory planner, and sequential executor with
unoptimized CPU reference kernels — and prove it by running one classification
model end to end through a new `NATIVE` backend whose outputs match
`ONNX_RUNTIME` on the same file within tolerance.

No CUDA in this phase. The GPU path is what the engine exists for, but the
correctness spine and the reference kernels it will be validated against must
exist first.

## In Scope

- [R-1] A new top-level `engine/` static library target (`neuriplo_engine`)
  that compiles standalone and includes nothing from `include/neuriplo/`,
  `backends/src/`, or `src/`. The dependency arrow points one way only.
- [R-2] ONNX model loading: graph topology, initializers, declared inputs and
  outputs, and attributes for the op set in [R-5], from a `.onnx` file on disk.
- [R-3] Static shape inference over the loaded graph. An unsupported op, an
  unsupported attribute combination, or a shape that cannot be resolved
  statically fails at load with the offending node name and op type — never at
  first inference, and never by guessing.
- [R-4] A device-agnostic core (parser, IR, shape inference, memory planner,
  executor) separated from a per-device layer (allocator, transfer, kernel
  table keyed by op). This phase ships only the CPU device layer, but the seam
  is a requirement of this phase, not a later refactor.
- [R-5] CPU reference kernels sufficient for ResNet-18: `Conv`, `Gemm`,
  `MatMul`, `Add`, `Relu`, `MaxPool`, `GlobalAveragePool`, `Flatten`,
  `Reshape`. FP32 only.
- [R-6] A sequential executor over the planned graph with an arena allocator:
  one allocation per inference at most, buffers reused by liveness.
- [R-7] Device is selected once at load time for the whole graph. There is no
  per-op device fallback, now or later. A graph that cannot run entirely on the
  requested device is rejected at load.
- [R-8] A `NATIVE` backend: registry ID, CMake module, and a thin adapter under
  `backends/native/` implementing `InferenceInterface`, including the
  `RawOutputTensor` path via `get_infer_results_raw`, and reporting
  `InferenceMetadata` derived from the engine's own graph inspection.
- [R-9] `use_gpu = true` (legacy overload) or an equivalent `EngineOptions`
  device request fails clearly on the `NATIVE` backend in this phase. It does
  not silently run on CPU.
- [R-10] A parity test: the same ONNX file through `NATIVE` and through
  `ONNX_RUNTIME`, outputs compared elementwise within a stated tolerance,
  runnable in CI with no GPU.
- [R-11] Backend inventory updated coherently in the same change:
  `cmake/BackendRegistry.cmake`, `docs/backends.yaml`, regenerated `GEN:`
  sections, Docker metadata, and CI backend identifiers.

## Out of Scope

- CUDA kernels and the GPU device layer — Phase N1.
- Fusion passes, layout transformation, tactic selection, plan serialization.
  The engine is an interpreter in this design, not an AOT compiler (see [D-2]).
- Quantization, FP16, INT8, dynamic shapes, batching beyond the fixture's
  declared batch dimension.
- Attention / transformer ops — needed for the ViT-based models eventually
  targeted, deferred to Phase N2.
- Object detection fixtures and postprocessing.
- Making `NATIVE` the `DEFAULT_BACKEND`, and any deprecation of `OPENCV_DNN` —
  Phase N3, and gated on [Q-3].
- Plugin packaging of the native backend.
- Any optimization of the CPU kernels (see [D-4]).

## Decisions

- [D-1] The engine lives at top-level `engine/`, not `src/engine/` and not
  `backends/native/src/`. Root `src/` holds the abstraction's own glue
  (`InferenceBackendSetup.cpp`); nesting a dependency-free library inside it
  inverts the arrow in [R-1]. `backends/native/` holds only the adapter.
  A later extraction is then `git subtree split engine/` with history intact.
- [D-2] The engine is an interpreter (`load` → `forward`, shapes resolved at
  load), not a plan-building compiler with autotuning and a serialized engine
  file. Rationale: the target is a replacement for OpenCV's `dnn` module, whose
  API and usage model this matches; the compiler model is TensorRT's ground and
  `TENSORRT` already occupies that slot in this repository.
- [D-3] Device chosen once per loaded graph ([R-7]). Per-op host round-trips are
  the specific failure mode that makes `OPENCV_DNN`'s CUDA target hard to reason
  about, and the constitution already forbids silent device movement.
- [D-4] CPU reference kernels are deliberately naive and stay that way. They are
  the executable specification and the correctness oracle for the CUDA kernels
  in Phase N1. Vectorizing them is a separate, separately approved effort, and
  documentation must say so, so nobody benchmarks this path and concludes the
  engine is slow.
- [D-5] `NATIVE` is registered but not default in this phase. The default
  backend change is a public-behavior change and gets its own packet.
- [D-6] No new runtime dependency is introduced without resolving [Q-1] first;
  `specs/tech-stack.md` forbids it, and this phase does not treat ONNX parsing
  as an exception to that rule.

## Constraints

- Constitution amendment required and included in this branch: `mission.md`
  currently states that Neuriplo *orchestrates* inference runtimes and *does not
  replace vendor SDKs*. A first-party runtime contradicts that boundary as
  written. The amendment narrows the non-goal to vendor SDK *reimplementation
  for its own sake* and admits a first-party runtime as a supported backend.
  Without this, the phase is specified against a constitution it violates.
- `InferenceInterface` lifecycle, metadata, input validation, and output
  semantics are preserved. The adapter adapts; the interface does not change.
- The legacy `setup_inference_engine(model_path, use_gpu, batch_size,
  input_sizes)` overload keeps working.
- C++17. CMake ≥ 3.10. glog is the only required library; OpenCV must remain
  required by `OPENCV_DNN` alone — the native engine must not pull it in.
- Enabling `NATIVE` must not make any toolchain a hard requirement for someone
  building a different backend.

## Dependencies

- From the codebase (verified on `origin/develop` @ `62825dd`):
  `backends/src/InferenceInterface.hpp` (contract, `RawOutputTensor`,
  exception hierarchy), `backends/src/IBackendRuntimeFactory.hpp` and
  `BackendRuntimeRegistry.cpp` (registration), `cmake/BackendRegistry.cmake`
  (14 IDs today), `cmake/LinkBackend.cmake`, `scripts/gen_backend_docs.py`,
  `docs/backends.yaml`.
- Fixture: `backends/onnx-runtime/test/export_torchvision_classifier.py`
  already produces a torchvision classifier ONNX export. Reuse it rather than
  adding a second fixture path.
- Parity oracle: the `ONNX_RUNTIME` backend must be buildable in the same
  configuration as `NATIVE` for [R-10].

## Assumptions & Open Questions

- [A-1] ResNet-18 as exported by the existing script needs only the [R-5] op
  set. Basis: standard torchvision opset-17 export. Confirm by dumping the
  actual node list before committing to the kernel list — if the export emits
  `BatchNormalization` unfused, the set grows by one.
- [A-2] A tolerance of 1e-4 max absolute difference against ORT FP32 is
  achievable for this graph without matching accumulation order. Basis:
  inference; confirm empirically in [V-6] and record the real figure.
- [Q-1] **ONNX parsing dependency — needs a decision before Group 2.**
  Option A: depend on protobuf and the vendored `onnx.proto`. Standard,
  robust, but a new required dependency for this backend and a compatibility
  review under `specs/tech-stack.md`. Option B: a minimal hand-written
  protobuf wire-format reader for the subset of `ModelProto` actually used.
  No dependency, full control, and consistent with a from-scratch engine; costs
  a few hundred lines and owns its own bug surface. Recommendation: Option B,
  scoped to the fields [R-2] needs, with the reader isolated behind one
  interface so Option A stays reachable.
- [Q-2] Sequencing against roadmap Phase 1 (Backend Metadata Consistency,
  currently `Next`). Phase 1 makes inventory drift fail early; `NATIVE` is a
  new inventory entry with no `versions.env` version and no setup script, which
  is a case Phase 1's check must handle. Doing Phase 1 first makes this phase's
  [R-11] cheaper and better tested. Owner: maintainer.
- [Q-3] Does `NATIVE` eventually replace `OPENCV_DNN` in this repository, or
  coexist? If it replaces it, the default-backend change and OpenCV's removal
  from the default build are a public-behavior migration needing a deprecation
  window — which `specs/mission.md` lists as an unresolved policy question.
- [Q-4] Whether cuBLAS/cuDNN are admissible in Phase N1, or the CUDA kernels
  are first-party all the way down. It does not block this phase, but it is the
  credibility line for the project and belongs in `tech-stack.md` before any
  kernel is written. Recommendation: first-party by default, vendor libraries
  behind an opt-in flag used as a performance baseline.

## Definition of Done (requirements level)

- [ ] Every [R-n] implemented or explicitly deferred with a tracked location
- [ ] [Q-1] resolved and recorded as a decision before Group 2 starts
- [ ] Constitution amendment reviewed and merged in this branch
- [ ] No In Scope behavior silently dropped; no Out of Scope work smuggled in
