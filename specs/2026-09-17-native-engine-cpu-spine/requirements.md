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
- [R-9] A GPU request on `NATIVE` fails clearly in this phase and never
  silently runs on CPU. The failure has two boundaries, because the existing
  contract differs between them and the requirement must match it:
  backend construction (the factory, called directly) throws an
  `InferenceException` whose message names the phase limitation; the public
  `setup_inference_engine` overloads return `nullptr` and log, since both catch
  `InferenceException` and `std::exception` and translate them
  (`src/InferenceBackendSetup.cpp:147-160`, with the legacy
  `use_gpu`/`batch_size`/`input_sizes` overload delegating to the
  `EngineOptions` one). Either way no inference runs. This requirement adapts
  to that contract rather than changing it.
- [R-10] A parity test: the same ONNX file through `NATIVE` and through
  `ONNX_RUNTIME`, outputs compared elementwise within a stated tolerance,
  runnable in CI with no GPU.
- [R-11] Backend inventory updated coherently in the same change:
  `cmake/BackendRegistry.cmake`, `docs/backends.yaml`, regenerated `GEN:`
  sections, Docker metadata, and CI backend identifiers.
- [R-12] Shape inference and memory planning take concrete input shapes as an
  argument and return a plan. This phase calls them once at load for one fixed
  input shape, but neither may be written as a one-time side effect of loading:
  re-running them for a different input shape must be possible without touching
  the parser, the IR, or any kernel. Dynamic shapes stay out of scope here
  ([A-3]), and this is the seam that keeps them additive in Phase N2 instead of
  a rewrite.

## Out of Scope

- CUDA kernels and the GPU device layer — Phase N1.
- Fusion passes, layout transformation, tactic selection, plan serialization.
  The engine is an interpreter in this design, not an AOT compiler (see [D-2]).
- Quantization, FP16, INT8, and batching beyond the fixture's declared batch
  dimension.
- Dynamic input and output shapes at runtime — deferred to Phase N2, and
  deliberately deferred rather than designed out: [R-12] requires the planning
  seam that makes them additive.
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
  A later extraction is then `git subtree split --prefix=engine/` with history
  intact — the prefix is an option, not a positional argument, and the bare form
  exits `fatal: you must provide the --prefix option` (verified on git 2.43.0).
- [D-2] The engine is an interpreter (`load` → `forward`, shapes resolved at
  load), not a plan-building compiler with autotuning and a serialized engine
  file. Rationale: the target is a replacement for OpenCV's `dnn` module, whose
  API and usage model this matches; the compiler model is TensorRT's ground and
  `TENSORRT` already occupies that slot in this repository. The design reference
  for this shape of engine is OpenCV 5's rewritten `dnn` engine rather than the
  4.x classic one — a typed operation graph with real shape inference, constant
  folding, and fusion, executed by an interpreter. It is the closest modern
  statement of the design being built here, which is why it is worth reading
  before writing the IR. Read as a reference only: see Dependencies.
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
  already produces a torchvision ResNet-18 ONNX export. Reuse it rather than
  adding a second fixture path — but reuse it with eyes open. As it stands the
  ORT test only `configure_file`s the script into the build tree and never runs
  it; the test reads `model_path.txt`, silently falls back to a mock when it is
  missing, and `GTEST_SKIP`s its integration test. A parity test modelled on
  that would pass while comparing nothing. So [R-10] additionally requires:
  deterministic fixture provisioning as an explicit step before parity runs,
  both backends given the identical path and identical input, and a **hard
  failure rather than a skip or a mock** when the fixture is absent. The script
  also hard-codes its output to `/workspace/`, which the provisioning step must
  not silently inherit.
- Parity oracle: the `ONNX_RUNTIME` backend must be buildable in the same
  configuration as `NATIVE` for [R-10].
- Design reference, read and not linked: OpenCV 5's new `dnn` engine. It is a
  rewritten typed-graph interpreter with shape inference, constant folding,
  fusion, subgraph and dynamic-shape support, covering over 64% of the ONNX
  specification, selected at load through `readNet`'s `engine` argument
  (`ENGINE_AUTO`) or `OPENCV_FORCE_DNN_ENGINE`, and CPU-only with GPU support
  deferred to later OpenCV releases. Useful as a reference for graph
  representation, shape handling, and where fusion belongs.
  This is explicitly **not** a dependency, a component, or a migration
  proposal. Neuriplo does not adopt OpenCV 5's engine, does not link it from
  `engine/`, and does not change its pinned `OPENCV_VERSION` on account of this
  phase — `OPENCV_DNN` stays exactly as it is. The native engine is
  first-party code informed by that design, which is the whole point of
  building it.

## Assumptions & Open Questions

- [A-1] ResNet-18 as exported by the existing script needs only the [R-5] op
  set. Corrected basis: the script exports with **`opset_version=12`**, not 17
  (`backends/onnx-runtime/test/export_torchvision_classifier.py:23`). Group 2
  must implement attribute and shape semantics against opset 12's schemas, or
  the exporter must be changed and pinned first — targeting the wrong opset is
  possible even when [T-3] confirms the node names match. Confirm by dumping the
  actual node list before committing to the kernel list; if the export emits
  `BatchNormalization` unfused, the set grows by one.
- [A-2] A tolerance of 1e-4 max absolute difference against ORT FP32 is
  achievable for this graph without matching accumulation order. Basis:
  inference; confirm empirically in [V-6] and record the real figure.
- [A-3] The static-shape restriction in [R-3] is a phase boundary, not a
  property of the design. Basis: shapes are resolved from concrete input shapes
  either way; dynamic support differs in when and how often planning runs, plus
  a plan cache, not in what the parser, IR, or kernels look like. The reference
  engine in [D-2] supports dynamic shapes on the same kind of IR, which is
  evidence the boundary is a schedule choice rather than a ceiling. Confirm by
  [V-13]; if it turns out false, [R-12] is the requirement that failed and
  Phase N2's cost estimate is wrong.
- [Q-1] **ONNX parsing dependency — needs a decision before Group 2.**
  Option A: depend on protobuf and the vendored `onnx.proto`. Standard,
  robust, but a new required dependency for this backend and a compatibility
  review under `specs/tech-stack.md`. Option B: a minimal hand-written
  protobuf wire-format reader for the subset of `ModelProto` actually used.
  No dependency, full control, and consistent with a from-scratch engine; costs
  a few hundred lines and owns its own bug surface. Recommendation: Option B,
  scoped to the fields [R-2] needs, with the reader isolated behind one
  interface so Option A stays reachable.
  Dissent on record: automated review of this packet argued for Option A
  first — a mature wire-format implementation lowers parser-correctness and
  malformed-input risk and tracks schema evolution — with Option B as the
  fallback only if the dependency cannot be kept exclusive to `NATIVE`.
  Both readings agree on the isolating interface, so Group 1 is unaffected
  either way. Still the maintainer's decision; still blocking Group 2.
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
