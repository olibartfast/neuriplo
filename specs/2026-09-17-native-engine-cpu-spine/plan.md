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
- [x] [T-3] Confirmed [A-1] by dumping the node list and op types of the
  exported ResNet-18 (2026-09-30, torch 2.12: opset 18, 49 nodes, no
  BatchNormalization; evidence in [A-1]). Decision taken: re-pin the exporter
  to opset 18 explicitly ([D-8]); [R-5] adjusted to the opset-18 set. Group 2
  implements attribute and shape semantics against opset 18's schemas.

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
  attributes, unsupported tensor dtypes, unresolvable dynamic dimensions.
  Compute tensors (graph inputs and outputs) are float32-only; embedded
  initializers may additionally be int64 shape constants ([D-9]).
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
- Group 1 (2026-09-30, orchestrator amendment): Group 1 may create
  `backends/native/test/CMakeLists.txt` as an **empty test-dir stub** (no
  tests, no sources). Reason: `neuriplo_add_backend_tests`
  (`cmake/BackendRegistry.cmake`) runs an unguarded `add_subdirectory` over
  every enabled backend's `TEST_DIR`, so the required `-DDEFAULT_BACKEND=NATIVE`
  configure with tests on fails while the directory is absent — and the packet
  otherwise forbids creating `backends/native/` until Group 6. The stub is
  build glue, not adapter work; Group 6 fills the directory with the real
  adapter tests. The alternative (an existence guard in the shared function)
  was rejected: silently skipping missing test dirs could mask real errors for
  every backend. Also accepted in Group 1: the `version_var: null` NATIVE entry
  renders as a cosmetic `` `None` `` row in the GEN `cmake-version-variables`
  table; the generator is out of scope for this group.
- Group 2b (2026-09-30, maintainer decision): once the loader tests actually
  ran, the real fixture exposed two int64 initializers (`val_226`, `val_230`)
  that `Reshape` consumes. Recorded as [D-9] and folded into Group 2b rather
  than deferred: the IR now tags initializers with a dtype, `value_info` accepts
  `Int64`, and graph inputs/outputs stay float32-only. Also fixed in Group 2b:
  the engine tests were registering no cases (the engine dir is added before the
  top-level `enable_testing()`), and a hermetic encoder wrote the opset entry
  unwrapped — both were silent until the tests were made to run. This is the
  T-10/T-8 completion; [V-2]/[V-3] evidence recorded in `validation.md`.
- Group 3a (2026-09-30, attempt 2): T-11 static shape inference landed as
  `engine/include/engine/Shapes.hpp` (`InferShapes(const Graph&, const ShapeMap&)`
  → `InferredShapes`) and `engine/src/ShapeInference.cpp`, covering the full
  [R-5] op set against opset-18 semantics and consuming `Int64` constants for
  `Reshape` shape and `ReduceMean` axes. Inference is per-shape and reads no
  `value_info` for intermediates, which is the [R-12]/[V-13] shape seam:
  `engine_shapes_test` proves the same loaded graph shapes two input dims with
  no reload. Attempt 1 was rejected for inventing a Reshape "batch carry" to
  force the batch-1-pinned fixture to infer at batch 2; the rework enforces
  strict ONNX Reshape and asserts the fixture's batch-2 inference fails at that
  node. 3b (arena plan) and 3c (device seam) remain.
- Group 3b (2026-09-30): T-12 landed as `engine/include/engine/Plan.hpp`
  (`PlanMemory(const Graph&, const InferredShapes&)` → `MemoryPlan`) and
  `engine/src/MemoryPlanner.cpp`. Node outputs that are not inputs or
  initializers get arena buffers; lifetimes are half-open `[definition,
  last_use)` with graph outputs live to the end; each buffer takes the lowest
  64-byte-aligned offset that avoids every overlapping lifetime, in a
  deterministic order, and `arena_size` is the aligned high-water mark. This is
  the plan half of [R-12]/[V-13]: a hermetic two-`Conv` graph is planned at two
  spatial sizes from one load, and `arena_size < sum(intermediate sizes)` proves
  reuse ([V-5]b). 3c (device seam) remains.
- Group 3c (2026-09-30): T-13 landed as `engine/include/engine/Device.hpp` and
  `engine/src/CpuDevice.cpp` — `TensorView`, the `KernelFn` calling convention,
  and `Allocator`/`Transfer`/`KernelTable`/`Device` behind interfaces, with the
  single CPU implementation `CpuDevice()` (64-byte-aligned allocation, memcpy
  transfers, device-owned empty kernel table). This is the [R-4] seam the CUDA
  device layer slots into in Phase N1, and it fixes the kernel signature Group 4
  implements. Group 3 is complete; Group 4 (CPU reference kernels, T-14/T-15) is
  next.
- Group 4a (2026-09-30): `engine::InferenceException` added beside
  `ModelLoadException`; the private kernel layer (`engine/src/kernels/`) with
  the `FindCpuKernel` table wired into `CpuDevice`; and `Relu`, `Add`
  (multidirectional broadcast), `Reshape`, `ReduceMean` implemented naively with
  hand-computed tests. 4b (`Gemm`/`MatMul`) and 4c (`Conv`/`MaxPool`) remain.
- Group 4b (2026-09-30): `Gemm` (transA/transB, alpha/beta, optional rank-1
  `[N]` or rank-2 `[M,N]` bias) and `MatMul` (batch broadcasting, 1-D promotion
  on either side) landed with hand-computed tests. 4c (`Conv`/`MaxPool`) remains.
- Group 4c (2026-09-30): `Conv` (NCHW, groups, strides, dilations, explicit and
  automatic padding, optional bias) and `MaxPool` (kernel shape, ceil mode,
  padding) complete the [R-5] reference kernel set; the eight-op table resolves
  through `CpuDevice`. Group 4 is complete; Group 5 (sequential executor,
  T-16/T-17) is next.
- Group 5 (2026-09-30, attempt 2): T-16/T-17 landed as
  `engine/include/engine/Executor.hpp` (`Model(graph, input_dims, device)`,
  `Run(inputs) -> InferenceResult`) and `engine/src/Executor.cpp`. Construction
  infers shapes, plans memory, and allocates the single arena through the device
  once; `Run` binds inputs and initializers, walks nodes through the kernel
  table, and copies graph outputs out, allocating no arena memory. The fixture
  runs end to end to a finite `[1,1000]` output. Attempt 1 was rejected: it hid
  a planner defect behind a per-call heap copy. The rework changed
  `PlanMemory` lifetimes to the inclusive `[definition, last_use]` so a value
  stays live through its consumer and **no node output shares bytes with an
  input of its defining node** — the invariant read-then-write kernels need and
  the reason the executor needs no copy. `EnginePlan.OutputNeverAliasesItsInput`
  guards it. Group 6 (backend adapter, T-18..T-21) is next.
- Group 6a (2026-09-30): T-18..T-21 landed as `backends/native/src/NativeInfer.{hpp,cpp}`
  (implements `InferenceInterface`: metadata from the graph, typed and raw output
  paths, lifecycle Ready, engine exceptions translated to the backend's global
  types) and `NativeRuntimeFactory.hpp` (GPU requests throw the backend
  `InferenceException`, naming the CPU-only phase limitation). Build wiring:
  `cmake/Native.cmake` compiles the adapter into `neuriplo` with `USE_NATIVE`
  (and marks `neuriplo_engine` PIC), `cmake/LinkBackend.cmake` links the engine,
  and `BackendRuntimeRegistry.cpp` registers `"NATIVE"`. The adapter tests run
  the fixture through both output paths and assert the GPU-request throw.
- Group 6b (2026-09-30): the public boundary is asserted too. With
  `backend_id="NATIVE"`, `setup_inference_engine(EngineOptions{use_gpu=true})`
  and the legacy `setup_inference_engine(path, true, 1, {})` both return
  `nullptr` (the factory's throw translated by the existing catch-and-log), and
  a `use_gpu=false` control returns a Ready backend. Group 6 is complete; Group
  7 (parity, inventory, docs) is next. Open point carried into Group 7: [V-7]
  asks for the *shared* `BackendHybridTestBase` contract to run against NATIVE,
  but that template is currently instantiated only for `MockInferenceInterface`
  and its `TestEdgeCases` expects `std::invalid_argument` while the real
  interface throws `InferenceExecutionException`; the NATIVE adapter contract
  is covered by `NativeInferTest` instead. Resolve or document in Group 7.
- Group 7a (2026-09-30): T-27a/T-22 landed. `backends/native/test/provision_parity_fixture.py`
  deterministically exports the ResNet-18 fixture at opset 18 (torchvision pinned
  to `0.27.0`) into the build tree, rejecting `/workspace` and the legacy
  `Identity`/`Flatten`/`GlobalAveragePool` ops; the CMake target
  `native_parity_fixture` provisions it and `native_parity_test` (`ParityTest.cpp`)
  runs the same file and input through `NATIVE` and `ONNX_RUNTIME`, comparing
  elementwise. Observed maximum absolute difference **3.14713e-05**, inside the
  [A-2] 1e-4 budget; a missing fixture fails, never skips. Remaining: Group 7b
  (inventory/docs/README/changelog/roadmap and the [V-7] gap) and T-25 (full
  validation run).
- Group 7b (2026-10-01): T-23/T-24/T-26 docs half landed. `docs/backends.yaml`
  NATIVE entry now declares `dockerfile: null` (intentional: no external SDK,
  absent from the CI vendor-image matrix, exercised via
  `-DDEFAULT_BACKEND=NATIVE`) and `test_exe: NativeInferTest`; `gen_backend_docs.py
  --check` clean with no GEN diff. New `engine/README.md` (interpreter design,
  one-device rule, CPU kernels as non-target oracle, boundary, layout,
  build/test incl. parity). CHANGELOG `[Unreleased]/Added` NATIVE entry.
  Enumeration for [V-9a]: 15 registry IDs agree with 15 yaml IDs; 13 yaml
  `dockerfile` paths exist; NATIVE null is intentional; two pre-existing gaps
  observed and left untouched (DALI entry references missing
  `docker/Dockerfile.dali`; MIGRAPHX entry spells `Dockerfile.migrachx` while
  the file is `Dockerfile.migraphx`) — tracked as follow-ups outside this
  packet. Remaining: T-25 full validation run (Group 7c).
