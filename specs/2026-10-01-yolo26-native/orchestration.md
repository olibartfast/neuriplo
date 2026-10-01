# Orchestration — YOLO26 Detection on the Native Engine

Branch discipline from the spine phase holds: a packet is committed here
before every worker dispatch; one ledger row per attempt; workers never
commit; the orchestrator re-scores and commits. Spec files stay
orchestrator-owned; workers get bounded writable paths per packet.

Packet index:

- Y1a (below) — IR dtypes + Constant folding + Flatten admission
- Y1b (below) — allowlist + attribute policies + shape rules for the 23 ops, [Q1]–[Q3]
- Y2 (below) — elementwise / unary / reduction kernels
- Y2-repair (below) — Mod shape dtype follows input (int64)

### Packet Y2-repair — Mod shape dtype follows input

Orchestrator re-score of Y2 found a cross-packet contract bug: `compute_outputs`
in `engine/src/ShapeInference.cpp` types `Mod` output as float32, but the
fixture's `Mod` is int64 (Y1b [Q2]: fmod=0, int64 inputs) and the Y2 `Mod`
kernel is int64-only. End-to-end inference would infer f32 for a tensor the
kernel reads as int64. One-rule fix, nothing else.

- **Writable:** `engine/src/ShapeInference.cpp` (the `Mod` arm only),
  `engine/test/ShapeInferenceTest.cpp` (extend in place: int64 propagation
  case + keep the existing float32-behavior coverage untouched).
  Never: `specs/**`, kernels, loader, `Graph.hpp`, anything else.
- **Required final state:** `Mod` output dtype = first input dtype (both
  inputs must already agree per the loader; reject a mismatch with node
  context if not already enforced — verify, do not assume). Committed tests
  assert int64 in → int64 out.
- **Budget:** 5 turns. **Handback:** `GROUP Y2-REPAIR HANDBACK pass|fail`,
  evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_shapes|engine_kernels" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y2 — elementwise / unary / reduction kernels

Y1b landed (allowlist, policies, shape rules). This packet implements the 10
kernels with no data-movement semantics: `Mul`, `Div`, `Sub`, `Sigmoid`,
`Cast`, `Softmax`, `ReduceMax`, `Mod`, `Equal`, `Where`. No loader/shape
changes (policies pinned in Y1b); no executor changes.

Conventions (read the cited files before writing): one file per op in
`engine/src/kernels/` following `Add.cpp` (broadcast helpers) / `Relu.cpp`
(unary loop); declare in `engine/src/kernels/Kernels.hpp`; register in the
`FindCpuKernel` table in `CpuKernels.cpp`; list each new `.cpp` in
`engine/CMakeLists.txt:48-56` (explicit source list, no glob); tests in
`engine/test/KernelsTest.cpp` with hand-computed expected values.

- **Writable:** new `engine/src/kernels/{Mul,Div,Sub,Sigmoid,Cast,Softmax,ReduceMax,Mod,Equal,Where}.cpp`,
  `engine/src/kernels/Kernels.hpp`, `engine/src/kernels/CpuKernels.cpp`,
  `engine/CMakeLists.txt` (source list only), `engine/test/KernelsTest.cpp`.
  Never: `specs/**`, `backends/**`, `ModelLoader.cpp`, `ShapeInference.cpp`,
  `Graph.hpp`, Device/Executor/Planner sources, `scripts/**`, `cmake/**`
  (except the `engine/CMakeLists.txt` source list).
- **Required final state:**
  1. `Mul`/`Div`/`Sub`: NumPy multidirectional broadcast like `Add` (float32).
     `Div` by zero follows IEEE float (`inf`/`nan`), never throws.
  2. `Sigmoid`: `1/(1+exp(-x))` float32, computed against hand values
     (0→0.5, large→1, large-negative→0), not captured from another runtime.
  3. `Cast`: float32↔int64↔bool conversions per the `to` attr (integral
     truncation toward zero; nonzero→true); reject unsupported `to` with node
     context.
  4. `Softmax`: stable (subtract max) over `axis` (negative axis normalized),
     float32.
  5. `ReduceMax`: `axes`/`keepdims` per the Y1b policy; empty axes = all axes.
  6. `Mod`: fmod=0 trunc-remainder on int64 (dividend's sign; Y1b [Q2]).
  7. `Equal`: elementwise comparison → bool output (broadcast like `Add`).
  8. `Where`: (bool cond, X, Y) with NumPy broadcast across all three.
  9. Every kernel validates dtype/shape preconditions and throws
     `InferenceException` with node context on violation; every kernel has at
     least one hand-computed test plus one off-default test where a config
     exists (axis ≠ default, broadcast case, Cast pair).
- **Budget:** 15 turns. **Handback:** `GROUP Y2 HANDBACK pass|fail`, one line
  per obligation with evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R engine_kernels --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y1b — allowlist, attribute policies, shape rules, Q1–Q3

Y1a landed (Bool IR, Constant folding, Flatten). This packet teaches the loader
and shape inference the 23 detection ops. No kernels (table stays empty for
them; Y2/Y3), no executor changes.

Surveyed attribute surface (yolo26n NMS-embedded, opset 18 — oracle, not a
test dependency):

- `Cast{to}`, `Concat{axis}`, `ConstantOfShape{value}`, `Conv` (unchanged),
  `Flatten{axis}`, `Gather{axis}`, `GatherElements{axis}`, `MaxPool` (unchanged),
  `Mod{fmod}`, `ReduceMax{keepdims}`, `Reshape{allowzero}` (unchanged),
  `Resize{coordinate_transformation_mode, cubic_coeff_a, mode, nearest_mode}`,
  `Softmax{axis}`, `Split{axis}`, `TopK{axis, largest, sorted}`, `Transpose{perm}`.
- Input-form (not attrs) at opset 18: `Slice` (starts/ends/axes/steps),
  `Split` (split sizes), `Unsqueeze`/`Expand`/`ConstantOfShape`/`Reshape` data
  inputs, `TopK` K input, `Cast`/`Gather`/`GatherElements` indices.
- `Mul/Div/Sub/Add/Sigmoid/Shape/Equal/Where` carry no attrs in the fixture.

- **Writable:** `engine/src/ModelLoader.cpp` (allowlist + per-op attr tables),
  `engine/src/ShapeInference.cpp` (shape rules), `engine/test/LoaderTest.cpp`,
  `engine/test/ShapeInferenceTest.cpp` (extend in place). Never: `specs/**`,
  `backends/**`, `cmake/**`, `scripts/**`, kernels, Device/Executor/Planner
  sources, `Graph.hpp`.
- **Read-only:** `/tmp/yolo26survey/yolo26n.onnx` (local oracle for discovery
  and for a /tmp-only scratch load check — no committed test may open it).
- **Required final state:**
  1. All 23 ops in `allowed_attributes()` with opset-18 policies pinned to the
     surveyed surface; anything outside the policy rejects with node + op
     context. Resolve and report [Q1] (Resize mode/attrs), [Q2] (Mod fmod
     semantics), [Q3] (TopK largest/sorted, Slice input form) — report the
     answers in the handback; the orchestrator records them in requirements.
  2. Static shape rules for all 23 (dtypes: `Shape`/`ConstantOfShape`→int64,
     `Equal`→bool, `Cast`→target type, rest float32). TopK output shapes assume
     constant K; a non-constant K is a load rejection, never a dynamic shape.
  3. Full-file load verified with a /tmp-only scratch driver (never committed):
     report node/initializer/tensor counts for the oracle file.
  4. Committed tests stay hermetic (hand-built fixtures covering each new rule
     at least once, plus one rejection case per policy family).
- **Budget:** 15 turns. **Handback:** `GROUP Y1b HANDBACK pass|fail`, one line
  per obligation with evidence, the three Q answers, deviations, NO-GO,
  `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_loader|engine_shapes|engine_plan" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y1a — IR dtypes, Constant folding, Flatten admission

Survey correction (orchestrator): the fixture needs **23** new ops, not 22
(`Mul, Sigmoid, Concat, Split, Unsqueeze, Expand, Transpose, GatherElements,
Gather, Cast, Softmax, Resize, Div, Slice, TopK, ConstantOfShape, Equal,
Where, Shape, Sub, ReduceMax, Flatten, Mod`); spec files corrected in the same
commit as this packet. This packet covers only the dtype/folding foundation —
no allowlist or shape-rule changes for the 23 ops (those are Y1b), no kernels.

- **Writable:** `engine/include/engine/Graph.hpp`, `engine/src/ModelLoader.cpp`,
  `engine/src/ShapeInference.cpp` and `engine/src/MemoryPlanner.cpp` only
  insofar as dtype handling requires, `engine/test/LoaderTest.cpp`,
  `engine/test/MemoryPlannerTest.cpp` (extend in place; no new test files
  unless the registration pattern in `engine/test/CMakeLists.txt` demands it —
  read it first). Never: `specs/**`, `backends/**`, `cmake/**`,
  `scripts/**`, kernel files, `Device`/`Executor` sources.
- **Read-only:** `engine/include/engine/Shapes.hpp`, `Plan.hpp`, `Device.hpp`;
  `scripts/2026-09-17` spine specs [D-9]/[R-12] notes for the dtype precedent.
- **Required final state:**
  1. `DataType::Bool` added (`Graph.hpp:37-41`); intermediate tensors may be
     float32/int64/bool; graph inputs/outputs stay float32-only (reject
     otherwise with node context — verify the current enforcement, do not
     assume it).
  2. `Constant` nodes fold at load into `initializers` (float32 + int64;
     survey oracle: 64 Constants = 53 int64 + 11 float). Folded nodes vanish
     from `graph.nodes` and consumers rewire to the initializer name. A
     `Constant` of any other dtype is a load rejection with node context.
  3. `Flatten` loads (attribute `axis`; survey uses axis=1).
  4. Planner sizes buffers by dtype (bool 1, int64 8, float32 4); mixed-dtype
     graphs plan without overlap errors.
  5. Tests are hermetic (hand-built fixtures; do NOT depend on
     `/tmp/yolo26survey/yolo26n.onnx`). That survey file may be used as a
     local read-only oracle to confirm counts (485 nodes → 421 after folding;
     204 → 268 initializers) but no test may open it.
- **Budget:** 12 turns. **Handback:** `GROUP Y1a HANDBACK pass|fail`, one line
  per obligation with evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_loader|engine_plan" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

## Run ledger

| Attempt | Group | Role | Model | Turns | Wall clock | First-pass acceptance | Interventions | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | survey | Orchestrator | muse-spark | — | — | n/a | 0 | Fixture surveyed: yolo26n NMS-embedded, opset 18, 485 nodes, 23 new ops, [1,300,6] static; packet written |
| 1 | Y1a (implement) | Implementer | opencode `general` subagent | — | — | Pass (orchestrator re-scored: 5 files in scope, `engine_loader|engine_plan` 4/4 green) | 0 | `DataType::Bool`, f32-only I/O enforced, Constant folding f32+i64 with rewiring, Flatten admission, dtype-aware planner; hermetic tests only |
| 2 | Y1b (implement) | Implementer | opencode `general` subagent | — | — | Pass (orchestrator re-scored: 4 files in scope, `engine_*` 11/11 green, no ResNet regression) | 0 | 23-op allowlist + policies, static shape rules, Q1–Q3 answered; deviation accepted: `kernel_shape` admitted+ignored on Conv (spatial dims come from weights); full-file scratch load 421 nodes / 268 initializers |
| 3 | Y2 (implement) | Implementer | opencode `general` subagent | — | — | Pass with 1 finding (orchestrator re-score: scope exact, `engine_kernels` 2/2 green; Mod shape/kernel dtype mismatch) | 0 | 10 kernels (Mul/Div/Sub/Sigmoid/Cast/Softmax/ReduceMax/Mod/Equal/Where) with hand-computed + off-default tests; table + CMake wired; finding repaired in Y2-repair |
