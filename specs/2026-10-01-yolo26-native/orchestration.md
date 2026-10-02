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
- Y3 (below) — data-movement / shape / selection kernels
- Y3-repair (below) — rule/kernel dtype alignment (Resize sizes, TopK indices)
- Y4 (below) — YOLO parity harness (provisioning + native_yolo_parity_test)
- Y4-repair (below) — admit load-folded Constant in fixture gate + full chain
- Y5 (below) — static shape-parameter evaluation + int64 arithmetic
- Y5-repair (below) — Expand max-broadcast (rule + kernel)
- Y4-retry (below) — build + run yolo parity (harness defects allowed)
- Y6 (below) — Reshape -1/0 semantics in kernel
- Y4-retry-2 (below) — re-run parity after Y6 (no code changes)

### Packet Y4-retry-2 — re-run parity after Y6

Y6 fixed the Reshape `-1` runtime rejection that stopped the last parity run
at `/model.23/Reshape`. The harness is unchanged since. This packet only
re-runs the chain — no code changes unless the run exposes a harness defect
(same latitude + reporting duty as Y4-retry).

- **Writable:** the three Y4 harness files, and only to fix a harness defect
  (report first use). Never: `specs/**`, `engine/**`, anything else.
- **Required final state:** green chain with `YOLO_PARITY_MAX_ABS_DIFF`. Over
  1e-4 → stop, report `fail` with the figure + per-region stats, retune
  nothing.
- **Budget:** 20 turns. **Handback:** `GROUP Y4-RETRY-2 HANDBACK pass|fail`,
  observed figure, evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-parity --target native_yolo_parity_fixture --parallel 6 && cmake --build build-parity --parallel 6 && ctest --test-dir build-parity -R yolo_parity --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y4-retry — build + run yolo parity

Y5 unblocked the load path (full-file InferShapes succeeds; fixture provisioned
at `build-parity/native_yolo_parity_fixture/yolo26n.onnx`). The Y4 harness
(`backends/native/test/CMakeLists.txt` modification + new
`YoloParityTest.cpp` + `provision_yolo_fixture.py`, all uncommitted in-tree)
has never compiled or run. This packet finishes it.

- **Writable:** `backends/native/test/YoloParityTest.cpp`,
  `backends/native/test/provision_yolo_fixture.py`,
  `backends/native/test/CMakeLists.txt` (harness defects may be fixed here;
  each fix reported in the handback). Never: `specs/**`, `engine/**`,
  anything else.
- **Required final state:** the chain below runs green end to end with
  `YOLO_PARITY_MAX_ABS_DIFF` observed. If parity exceeds 1e-4, stop and report
  `fail` with the observed figure and per-region diff stats — do not retune
  kernels, tolerances, inputs, or engine code (orchestrator work).
- **Budget:** 20 turns. **Handback:** `GROUP Y4-RETRY HANDBACK pass|fail`,
  observed `YOLO_PARITY_MAX_ABS_DIFF`, evidence, deviations, NO-GO,
  `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-parity --target native_yolo_parity_fixture --parallel 6 && cmake --build build-parity --parallel 6 && ctest --test-dir build-parity -R yolo_parity --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y6 — Reshape -1/0 semantics in kernel

Y4-retry failed at runtime: `/model.23/Reshape` (data `[1,4,80,80]`, shape
`[1,4,-1]`) throws "shape operand product does not match" because
`engine/src/kernels/Reshape.cpp` multiplies the raw shape values including
`-1`. The shape rule (`reshape_shape`) already implements ONNX `-1`/`0`
semantics; the kernel never needed them until this fixture (ResNet shapes are
all explicit). Fixture audit 2026-10-01: all 15 Reshape sites carry float32
data — no dtype work in this packet.

- **Writable:** `engine/src/kernels/Reshape.cpp`,
  `engine/test/KernelsTest.cpp` (extend in place). Never: `specs/**`,
  shape rule, loader, `Graph.hpp`, other kernels, Device/Executor/Planner,
  `backends/**`, `cmake/**`, `scripts/**`.
- **Required final state:** kernel resolves the effective shape exactly as
  `reshape_shape` does (0 → copy of the input dim unless `allowzero`; at most
  one `-1` → `inCount / known product`, divisible; final product must equal
  the element count) and rejects the same cases with node context. Tests:
  `-1` inference, `0`-copy, `allowzero`, two-`-1` rejection, indivisible
  rejection — all hand-computed.
- **Budget:** 8 turns. **Handback:** `GROUP Y6 HANDBACK pass|fail`, evidence,
  deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y5-repair — Expand max-broadcast

Y5 failed only at obligation 5: `/model.23/Expand_6` rejects data `[1,300,1]`
against evaluated target `[1,1,80]` because the Y1b/Y3 Expand contract demands
output == target exactly. ONNX/ORT `Expand` is symmetric max-broadcast
(worker-probed: `[1,300,1]+[1,1,80]→[1,300,80]`, `[2,3]+[2,1]→[2,3]`). The
evaluated values are correct (cross-checked vs ORT); only the contract is wrong.

- **Writable:** `engine/src/ShapeInference.cpp` (the `expand_shape` arm only:
  output = max-broadcast(data dims, target); keep the existing rejections for
  genuinely incompatible pairs), `engine/src/kernels/Expand.cpp` (same
  semantics), `engine/test/ShapeInferenceTest.cpp`,
  `engine/test/KernelsTest.cpp` (extend in place: `[1,300,1]+[1,1,80]→[1,300,80]`
  cases for rule and kernel, plus an incompatible-pair negative each).
  Never: `specs/**`, loader, `Graph.hpp`, other kernels/rules, Device/Executor/
  Planner, `backends/**`, `cmake/**`, `scripts/**`.
- **Required final state:** rule and kernel agree on max-broadcast; the five
  fixture sites infer (verified with a /tmp-only scratch driver, never
  committed).
- Out of scope (recorded observation, not this packet): `GatherElements`
  shape rule hardcodes float32 while its kernel is dual-dtype — same latent
  class as the Gather deviation Y5 accepted; no fixture site triggers it.
- **Budget:** 8 turns. **Handback:** `GROUP Y5-REPAIR HANDBACK pass|fail`,
  evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y5 — static shape-parameter evaluation + int64 arithmetic

Y4-repair failed at NATIVE load: `/model.23/Slice[_1]` ends/starts and
`Expand_[67]` shapes come from `Shape→Gather→Add/Div/Mul` and
`Where`-of-constants chains — all int64, all fixed at load for the fixed
`[1,3,640,640]` input, but not initializers, so the constant-only shape rules
reject them. Two coupled gaps (both in scope, nothing else):

(a) No load-time evaluation of shape-parameter expressions.
(b) `Add/Sub/Mul/Div` shape rules type int64 arithmetic as float32 and the Y2
    kernels are float32-only; `Where` shape rule types X/Y as float32 and the
    kernel takes float32 X/Y — but the fixture runs all of these on int64
    (surveyed 2026-10-01: every constant in the five chains is dtype 7).

- **Writable:** `engine/src/ShapeInference.cpp`, `engine/src/kernels/{Add,Sub,Mul,Div,Where}.cpp`,
  `engine/test/ShapeInferenceTest.cpp`, `engine/test/KernelsTest.cpp`
  (extend in place). Never: `specs/**`, loader, `Graph.hpp`, other kernels,
  Device/Executor/Planner, `backends/**`, `cmake/**`, `scripts/**`.
- **Read-only:** `/tmp` fixture at `build-parity/native_yolo_parity_fixture/yolo26n.onnx`
  (oracle for the five chains; no committed test opens it).
- **Required final state:**
  1. Static evaluator: when a shape-input position requires a constant absent
     from initializers, recursively evaluate over exactly
     {Shape, Gather, Add, Sub, Mul, Div, Equal, Where, ConstantOfShape}.
     `Shape`-of-X uses X's already-inferred dims (topological order holds).
     Evaluator arithmetic must match kernel arithmetic exactly (int64
     trunc-division, Mul/Add/Sub int64, Equal→bool, Where select,
     ConstantOfShape fill, Gather with negative wrap + range check). Any op
     outside the whitelist, any data-dependent root, any failure → the
     existing load rejection, unchanged.
  2. `Add/Sub/Mul/Div` shape rules propagate input dtype (both inputs must
     agree; mismatch rejects with node context — the Mod precedent); kernels
     gain int64 paths (templated like the Y3 data-movement kernels).
  3. `Where` shape rule outputs X's dtype (X/Y must agree; mismatch rejects);
     kernel gains int64 X/Y (cond stays bool).
  4. Hermetic tests: evaluator chains mirroring the fixture's five sites
     (Shape→Gather→Add→Div→Mul ends; Where-of-constants Expand shape;
     int64 Add/Mul/Div/Where dtype propagation + agreement rejections;
     whitelist-violation rejection e.g. Sigmoid-rooted expression).
  5. Full-file load verified with a /tmp-only scratch driver (never committed):
     report node/initializer counts and that `/model.23/Slice[_1]` +
     `Expand_[67]` infer.
- **Budget:** 20 turns. **Handback:** `GROUP Y5 HANDBACK pass|fail`, one line
  per obligation with evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

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

### Packet Y3-repair — rule/kernel dtype alignment

Orchestrator re-score of Y3 found two rule/kernel mismatches in
`engine/src/ShapeInference.cpp` (kernels follow the Y3 packet; the rules must
be aligned to them):

1. `resize_shape` accepts a sizes-only input, but the Y3 `Resize` kernel (per
   its packet order) rejects a sizes input. Align the rule to the kernel:
   a present sizes input is a load rejection with node context (scales-only
   stays the single admitted form).
2. `topk_shapes` types the indices output as float32, but the Y3 `TopK`
   kernel writes int64 indices (and the fixture feeds them to `Gather`,
   which requires int64). Align the rule: indices output is int64.

- **Writable:** `engine/src/ShapeInference.cpp` (the two arms only),
  `engine/test/ShapeInferenceTest.cpp` (extend in place: sizes-rejection
  case, TopK int64-indices case). Never: `specs/**`, kernels, loader,
  `Graph.hpp`, anything else.
- **Budget:** 5 turns. **Handback:** `GROUP Y3-REPAIR HANDBACK pass|fail`,
  evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_shapes|engine_kernels" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y3 — data-movement / shape / selection kernels

Y2 landed (10 elementwise kernels). This packet implements the 13 kernels that
move, reshape, or select data: `Concat`, `Split`, `Unsqueeze`, `Expand`,
`Transpose`, `Slice`, `Gather`, `GatherElements`, `Resize`, `Flatten`,
`Shape`, `ConstantOfShape`, `TopK`. No loader/shape changes (Y1b rules are the
contract — match them exactly, including every rejection); no executor changes.

Read-before-write (mandatory): the Y1b shape-rule functions in
`engine/src/ShapeInference.cpp` (`concat_shape`, `split_shapes`,
`unsqueeze_shape`, `expand_shape`, `transpose_shape`, `gather_shape`,
`gatherelements_shape`, `resize_shape`, `slice_shape`, `topk_shapes`,
`constantofshape_shape`, `shape_shape`, `flatten_shape`, `cast_shape`) plus
`engine/src/kernels/Reshape.cpp` (shape-operand handling precedent).
Kernel behavior must agree with its shape rule on dims, dtype, and every
rejection case — a kernel that accepts what its rule rejects (or vice versa)
fails this packet.

Conventions (as Y2): one file per op, declare in `Kernels.hpp`, register in
`CpuKernels.cpp`, list in `engine/CMakeLists.txt`, hand-computed tests in
`engine/test/KernelsTest.cpp`.

- **Writable:** new `engine/src/kernels/{Concat,Split,Unsqueeze,Expand,Transpose,Slice,Gather,GatherElements,Resize,Flatten,Shape,ConstantOfShape,TopK}.cpp`,
  `engine/src/kernels/Kernels.hpp`, `engine/src/kernels/CpuKernels.cpp`,
  `engine/CMakeLists.txt` (source list only), `engine/test/KernelsTest.cpp`.
  Never: `specs/**`, `backends/**`, `ModelLoader.cpp`, `ShapeInference.cpp`,
  `Graph.hpp`, Device/Executor/Planner sources, `scripts/**`, `cmake/**`
  (except the `engine/CMakeLists.txt` source list).
- **Required final state:**
  1. `Concat` (any axis incl. negative), `Split` (axis + sizes input, unequal
     parts), `Unsqueeze`/`Expand` (axes/shape inputs), `Transpose` (perm),
     `Flatten` (axis) — exact copies/reordering, float32 and int64 data.
  2. `Slice`: opset-18 input form (data/starts/ends/axes, steps omitted);
     negative-index normalization + clamping exactly as the shape rule does.
  3. `Gather`/`GatherElements` (axis incl. negative, int64 indices, negative
     index wrap); out-of-range index throws with node context.
  4. `Resize` nearest per Y1b [Q1] (asymmetric, floor, scales input; sizes
     absent in fixture — reject sizes input if present, matching the rule).
  5. `Shape` (→int64 shape tensor), `ConstantOfShape` (value attr + int64
     input shape → filled output).
  6. `TopK` (axis=-1, largest=1, sorted=1, scalar int64 K input): values +
     int64 indices outputs, descending; K > dim clamps or rejects exactly as
     the shape rule does — verify, do not assume.
  7. Dynamic parameters (Slice starts/ends, Split sizes, Expand/Unsqueeze
     axes-shape, TopK K, ConstantOfShape input, Resize scales) are read from
     the input TensorViews as int64/float data and validated; mismatch with
     the inferred contract throws `InferenceException` with node context.
  8. Every kernel has a hand-computed test plus an off-default case (negative
     axis, clamping Slice, unequal Split, permuted Transpose, K < dim TopK);
     every documented rejection has a negative test.
- **Budget:** 15 turns. **Handback:** `GROUP Y3 HANDBACK pass|fail`, one line
  per obligation with evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R engine_kernels --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y4-repair — admit load-folded Constant in fixture gate

Y4 failed at fixture provisioning for a script-level reason the worker diagnosed:
`provision_yolo_fixture.py` rejects the raw export because it contains 64
`Constant` nodes outside the engine's 31-op allowlist — but `Constant` nodes
never reach the allowlist; the loader folds them into initializers first
(`ModelLoader.cpp` fold + Y1a survey: 53 int64 + 11 float, 485 → 421 nodes).
The gate must admit what the loader folds.

- **Writable:** `backends/native/test/provision_yolo_fixture.py` (one change:
  admit `"Constant"` in the op-set check with a comment citing the loader
  fold; nothing else), `backends/native/test/YoloParityTest.cpp` and
  `backends/native/test/CMakeLists.txt` only if the full chain exposes a
  defect in them (report first use of this latitude in the handback).
  Never: `specs/**`, `engine/**`, anything else.
- **Required final state:** the Y4 acceptance chain runs green end to end and
  reports `YOLO_PARITY_MAX_ABS_DIFF`. If parity exceeds 1e-4, stop and report
  `fail` with the observed figure and per-region diff stats — do not retune
  kernels, tolerances, or inputs (that is orchestrator work).
- **Budget:** 20 turns. **Handback:** `GROUP Y4-REPAIR HANDBACK pass|fail`,
  observed `YOLO_PARITY_MAX_ABS_DIFF`, evidence, deviations, NO-GO,
  `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-parity --target native_yolo_parity_fixture --parallel 6 && cmake --build build-parity --parallel 6 && ctest --test-dir build-parity -R yolo_parity --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y4 — YOLO parity harness

Y3 landed (all 23 ops have kernels). This packet proves end-to-end detection:
the same NMS-embedded yolo26n file through NATIVE and ONNX_RUNTIME, compared
elementwise. Mirror the 7a precedent exactly
(`backends/native/test/provision_parity_fixture.py`, `ParityTest.cpp`, and the
`ONNX_RUNTIME`-gated block in `backends/native/test/CMakeLists.txt`) — read
all three before writing.

- **Writable:** new `backends/native/test/provision_yolo_fixture.py`, new
  `backends/native/test/YoloParityTest.cpp`,
  `backends/native/test/CMakeLists.txt` (extend the `ONNX_RUNTIME`-gated
  block in place). Never: `specs/**`, `engine/**`, other `backends/**`,
  `cmake/**`, `scripts/**`, docs.
- **Required final state:**
  1. `provision_yolo_fixture.py <output_dir>`: exports Ultralytics `yolo26n`
     with `nms=True`, `imgsz=640`, opset 18 into `<output_dir>/yolo26n.onnx`;
     pins ultralytics/torch/onnx versions (fail loudly on mismatch — survey
     oracle: ultralytics 8.4.113, torch 2.12, onnx 1.22); refuses `/workspace`;
     verifies the emitted graph (opset 18, single `[1,3,640,640]` input,
     single `[1,300,6]` output, op set ⊆ the 31 supported ops) and prints the
     resolved path. Downloading `yolo26n.pt` weights (~5 MB) on first run is
     expected and must be stated in the script docstring.
  2. CMake target `native_yolo_parity_fixture` provisions into the build tree
     (always re-runs, like the ResNet target); test target
     `native_yolo_parity_test` with ctest name `native_yolo_parity`, fixture
     via env `NEURIPLO_YOLO_PARITY_FIXTURE`, TIMEOUT 1800.
  3. `YoloParityTest.cpp`: deterministic `1×3×640×640` float input (same
     `0.01*((i%7)+1)` pattern as `ParityTest`), both backends, elementwise max
     abs diff ≤ 1e-4, prints `YOLO_PARITY_MAX_ABS_DIFF=<v>`; missing fixture
     fails, never skips. Output is `[1,300,6]` = 1800 floats.
  4. Only meaningful (and only wired) when `ONNX_RUNTIME` is enabled — same
     `IN_LIST` gate as the ResNet block. Default `OPENCV_DNN` builds are
     untouched.
- CHANGELOG, validation evidence, and the merge-hold decision stay
  orchestrator-owned — do not touch them.
- **Budget:** 20 turns (parity build + run is slow). **Handback:**
  `GROUP Y4 HANDBACK pass|fail`, observed `YOLO_PARITY_MAX_ABS_DIFF`, evidence,
  deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-parity --target native_yolo_parity_fixture --parallel 6 && cmake --build build-parity --parallel 6 && ctest --test-dir build-parity -R yolo_parity --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
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
| 4 | Y2-repair (implement) | Implementer | opencode `general` subagent | — | — | Pass (orchestrator re-scored: 2 files in scope, shapes+kernels 4/4 green) | 0 | Mod output dtype follows input with mismatch rejection; int64 propagation + negative tests |
| 5 | Y3 (implement) | Implementer | opencode `general` subagent | — | — | Pass with 2 findings (orchestrator re-score: scope exact, 119 kernel tests green) | 0 | 13 kernels (Concat/Split/Unsqueeze/Expand/Transpose/Slice/Gather/GatherElements/Resize/Flatten/Shape/ConstantOfShape/TopK), 26+23 tests; findings (Resize sizes + TopK indices dtype) repaired in Y3-repair |
| 6 | Y3-repair (implement) | Implementer | opencode `general` subagent | — | — | Pass (orchestrator re-scored: 2 files in scope, shapes+kernels 4/4 green) | 0 | Resize sizes-input rejected, TopK indices int64; rules now match kernels |
| 7 | Y4 (implement) | Implementer | opencode `general` subagent | — | — | Fail (gate bug, diagnosed precisely) | 0 | Harness written; provision gate rejected load-folded `Constant`; repaired in Y4-repair |
| 8 | Y4-repair (implement) | Implementer | opencode `general` subagent | — | — | Fail (gate fixed, provision green; load blocked at `/model.23/Slice`) | 0 | Gate admits `Constant`; computed Slice/Expand params need static evaluation → Y5 |
| 9 | Y5 (implement) | Implementer | opencode `general` subagent | — | — | Pass with 1 finding (evaluator + int64 work verified in the green tree; Expand finding → Y5-repair) | 1 (Gather rule propagates data dtype — required, accepted) | Static evaluator (9-op whitelist), int64 Add/Sub/Mul/Div + Where, hermetic chain tests |
| 10 | Y5-repair (implement) | Implementer | opencode `general` subagent | — | — | Pass (orchestrator re-scored: `engine_` 11/11 green; full-file InferShapes succeeds per scratch) | 1 (acceptance ran twice: typo fix then green — recorded) | Expand max-broadcast in rule + kernel; all 8 fixture Expand sites infer |
| 11 | Y4-retry (validate) | Validator | opencode `general` subagent | — | — | Fail (harness sound; NATIVE runtime rejects Reshape `-1` at `/model.23/Reshape`) | 0 | No harness fixes needed; engine gap → Y6 |
| 12 | Y6 (implement) | Implementer | opencode `general` subagent | — | — | Pass (orchestrator re-scored: 2 files in scope, `engine_` 11/11 green) | 0 | Reshape kernel mirrors rule `-1`/`0` semantics; all-15 fixture sites float32, no dtype work |
