# Plan — Native Engine: CPU Reference Spine

Each group ends in something runnable or checkable. Groups 1–6 keep `NATIVE`
unreachable from a normal build (option off by default) so `develop` stays green
throughout.

## Group 0 — Decisions and constitution

- [T-1] Resolve [Q-1] (ONNX parsing dependency) and record it as [D-7] in
  `requirements.md`. Blocks Group 2.
- [T-2] Amend `specs/mission.md` (Boundaries and Non-Goals) and
  `specs/tech-stack.md` (native engine section, non-choices, open decisions) so
  a first-party runtime is admissible. Add the Native Engine Track to
  `specs/roadmap.md`.
  - Checks: `./scripts/quality/format.sh --check`; both files still readable in
    under five minutes.
- [T-3] Confirm [A-1] by dumping the node list and op types of the exported
  ResNet-18, and record the opset the fixture is actually pinned to (it is 12,
  not 17). Deliverable: the actual op set, pasted into `requirements.md`
  under Context, [R-5] adjusted if it differs, and an explicit decision to
  either implement opset 12 semantics or re-pin the exporter — before Group 2
  writes any attribute handling.

## Group 1 — Skeleton and build seam

- [T-4] Create `engine/` with `CMakeLists.txt` producing the `neuriplo_engine`
  static library, its own `engine/include/` public headers, and no link or
  include path into the abstraction.
- [T-5] Teach the registry and `validate_backend_versions()` an explicit
  no-external-SDK case **before** `NATIVE` is registered, then add
  `cmake/Native.cmake` and the `NATIVE` entry in `cmake/BackendRegistry.cmake`
  (module `Native`, test dir `backends/native/test`).
  The order is load-bearing, not cosmetic: `neuriplo_get_backend_property`
  raises `FATAL_ERROR` on an undefined property
  (`cmake/BackendRegistry.cmake:177-182`), and `validate_backend_versions()`
  asks every registered ID for `VERSION_VAR` unconditionally on a top-level
  configure (`CMakeLists.txt:38`, `cmake/versions.cmake:155-167`). All 14
  current backends define one. `NATIVE` is the first that cannot, so
  registering it while the property is merely absent fails *every* top-level
  configure — including the `OPENCV_DNN` build Groups 1-6 promise to keep
  green. Give the validator a declared-null version it understands rather than
  leaving the property undefined, and do not defer this to Group 7. This is
  [Q-2]'s cheaper-order argument made concrete.
- [T-6] Add a build-time guard that the arrow in [R-1] holds: fail configure or
  a test if any translation unit under `engine/` includes abstraction headers.
  - Checks: `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN` still
    configures, builds, and tests green *with `NATIVE` registered* — the
    regression [T-5] exists to prevent; `-DDEFAULT_BACKEND=NATIVE` configures
    and builds an empty engine.
- [T-7] Add `docs/backends.yaml` entry with `setup_script: null`,
  `version_var: null`, and regenerate `GEN:` sections.

## Group 2 — ONNX loading and graph IR

- [T-8] Implement the model reader per [D-7]: `ModelProto` subset → nodes,
  attributes, initializers, graph inputs/outputs, tensor dtypes and shapes.
- [T-9] Define the graph IR: node list in topological order, tensor value table,
  initializer storage, and an explicit distinction between graph inputs,
  initializers, and intermediates.
- [T-10] Reject at load, with node name and op type: unknown ops, unsupported
  attributes, non-FP32 tensors, unresolvable dynamic dimensions.
  - Checks: unit tests load the fixture and assert node/initializer counts;
    malformed and unsupported-op files produce `ModelLoadException`.

## Group 3 — Shape inference and memory planning

- [T-11] Static shape inference for the [R-5] op set, producing a concrete
  shape and dtype for every tensor in the graph.
- [T-12] Liveness analysis over the topological order, then an arena plan:
  offset and size per intermediate, arena size computed once at load.
- [T-13] Device-layer seam: an allocator/transfer/kernel-table abstraction with
  exactly one implementation (CPU) in this phase, per [R-4].
- [T-27] Shape seam per [R-12]: shape inference and the planner take concrete
  input shapes and return a plan, rather than mutating the graph at load. This
  phase calls them once, for one shape. Reference read before writing this:
  how the [D-2] reference engine separates graph from per-shape plan.
  - Checks: unit test asserts inferred shapes for the fixture against ORT's own
    inferred shapes; a plan test asserts arena reuse actually overlaps
    non-overlapping lifetimes rather than summing all buffers; a second plan for
    a different input shape is produced without touching parser, IR, or kernels
    ([V-13]).

## Group 4 — CPU reference kernels

- [T-14] Implement the [R-5] kernels, naive and readable, one file per op, no
  vectorization or threading ([D-4]).
- [T-15] Per-kernel unit tests with hand-computed small cases, including
  non-default `Conv` stride/pad/dilation and `Gemm` transpose flags.
  - Checks: kernel tests pass; each kernel's test includes at least one case
    computed by hand rather than captured from another runtime.

## Group 5 — Executor

- [T-16] Sequential executor: bind graph inputs, walk nodes, dispatch through
  the kernel table, expose outputs. One arena allocation per inference at most.
- [T-17] End-to-end engine test: load the ResNet-18 fixture, run a fixed input,
  assert output shape and finiteness (parity comes in Group 7).
  - Checks: `ctest -R engine` green; run under ASan and UBSan.

## Group 6 — Backend adapter

- [T-18] `backends/native/` adapter: `NativeInfer` implementing
  `InferenceInterface`, plus `NativeRuntimeFactory` registered with
  `BackendRuntimeRegistry`, following the shape of
  `backends/onnx-runtime/src/ORTRuntimeFactory.hpp`.
- [T-19] Implement `get_infer_results_raw` to copy engine output buffers
  straight into `RawOutputTensor` without materializing `TensorElement`
  variants; keep `get_infer_results` as the adapted path.
- [T-20] Metadata from engine graph inspection; lifecycle states mapped to the
  engine's load/ready states.
- [T-21] Device request handling per [R-9]: the factory throws with a clear
  message naming the phase limitation, which the public setup overloads then
  translate into their existing `nullptr`-and-log contract. Do not change
  `src/InferenceBackendSetup.cpp` to make the exception escape.
  - Checks: the shared backend contract tests pass with
    `-DDEFAULT_BACKEND=NATIVE`.

## Group 7 — Parity, inventory, docs

- [T-27a] Fixture provisioning for [R-10], before any parity task: a single
  deterministic step that actually generates the ResNet-18 export, writes it to
  a path under the build tree rather than the script's hard-coded `/workspace/`,
  pins the exporter's opset and torchvision version, and reports the resolved
  path. Copying the script into the build directory is not provisioning.
- [T-22] Parity harness for [R-10]: build with `NATIVE` and `ONNX_RUNTIME`
  both enabled, feed both the identical file from [T-27a] and the identical
  input, compare elementwise, and record the observed max absolute difference.
  The test fails — never skips, never substitutes a mock — if the fixture is
  missing, unlike the ORT test it is modelled on.
- [T-23] Complete [R-11]: registry, `docs/backends.yaml`, regenerated docs,
  Docker metadata, CI identifiers. Handle the no-external-SDK case surfaced by
  [Q-2].
- [T-24] `engine/README.md` stating the interpreter design [D-2], the one-device
  rule [D-3], and the explicit statement that CPU kernels are a reference
  implementation and not a performance target [D-4].
- [T-25] Execute everything in `validation.md` and record evidence.
- [T-26] `CHANGELOG.md` entry; update roadmap status for Phase N0.

## Notes

- Conventions reused: per-backend `src/` + `test/` layout, factory-per-backend
  registration, `versions.env` + `docs/backends.yaml` + generated docs as the
  single inventory path, `feature/<slug>` branches off `develop` merged by PR.
- `NATIVE` stays behind its CMake option and off the default path until Phase N3
  proposes the default-backend change.
- Record in this section anything that deviated from the plan and why.
