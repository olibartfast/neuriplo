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
- Y7 (below) — plan buffers for unconsumed node outputs
- Y8 (below) — data-movement dtype propagation audit
- Y4-retry-2 (below) — re-run parity after Y6 (no code changes)
- Y4-retry-3 (below) — re-run parity after Y7 (no code changes)
- Y4-retry-4 (below) — parity run on the green tree (no code changes)
- Y9 (below) — order-invariant parity + non-degenerate input (maintainer: both fixes, 2026-10-09)
- Y10 (below) — GatherElements axis-stride fix (dormant-but-real, 2026-10-09)
- Y10-repair (below) — correct the unequal-stride expectation
- Y11 (below) — real-image threshold-gated parity (maintainer: real image + threshold, 2026-10-10)
- Y12 (below) — NATIVE CI job (maintainer go-ahead, 2026-10-10)

### Packet Y10 — GatherElements axis-stride fix

Audit finding (2026-10-09, fixture-dormant, real): `engine/src/kernels/GatherElements.cpp:62`
advances the gather-axis offset by `k * strides[axis]` (indices-grid stride)
where `k` indexes the data grid — it must be `in_strides[axis]` (the non-axis
arms already use `in_strides`). Dormant in the fixture only because all three
sites have equal trailing dims. Fix the latent bug now with a hermetic
unequal-stride proof.

- **Writable:** `engine/src/kernels/GatherElements.cpp` (the stride line only),
  `engine/test/KernelsTest.cpp` (extend in place: unequal-stride GatherElements
  case where old/new disagree + keep existing coverage green). Never:
  `specs/**`, other kernels, shape rule, loader, `Graph.hpp`,
  Device/Executor/Planner, `backends/**`, `cmake/**`, `scripts/**`.
- **Required final state:** stride uses the data-grid stride; new test fails
  against the old line and passes with the fix (hand-computed expectation);
  full `engine_` suite green; no behavior change on equal-stride sites.
- **Budget:** 8 turns. **Handback:** `GROUP Y10 HANDBACK pass|fail`, evidence,
  deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y12 — NATIVE CI job

No CI job builds the native backend (verified 2026-10-10: zero `NATIVE`
occurrences in `.github/`; vendor matrices cover 10 SDK backends; vendor
builds never compile `engine/`). Green CI therefore proves nothing about
NATIVE, and post-merge breakage would be invisible. This packet closes that
gap before the N0 branch merges.

- **Writable:** `.github/workflows/ci.yml` (one new job only),
  `docs/LOCAL_CI.md` (only if a new act pattern is needed; report first
  use). Never: `specs/**` (this packet text excepted), `engine/**`,
  `backends/**`, `cmake/**`, `scripts/**`, Dockerfiles.
- **Required final state:** a `native` job that reuses the `onnx-runtime`
  builder image (ORT + gtest + glog already present), installs the pinned
  torch export stack (`torch==2.12.0+cpu` + `torchvision==0.27.0+cpu` from
  the CPU index, then `onnx==1.22.0` + `ultralytics==8.4.113` + pillow from
  PyPI — versions verified resolvable 2026-10-10; scripts fail loudly on
  mismatch), configures `-DDEFAULT_BACKEND=NATIVE
  -DNEURIPLO_BACKENDS=ONNX_RUNTIME`, provisions both fixtures, and runs
  `ctest -R "engine_|NativeInferTest|native_parity|native_yolo_parity"`.
  Timeout 45 min. No new Docker image (NATIVE has no SDK; `dockerfile: null`
  stands).
- **Validation (repo rule for workflow changes):** `act push --job native
  --dryrun` then a full local `act push --job native` run, both green,
  before pushing.
- **Budget:** n/a (orchestrator-implemented). **Handback:** job URL + act
  evidence + first CI run link on the PR.
- **Acceptance:** the `native` job is green on the PR (GitHub), plus the
  local act run recorded below.
- **Evidence 2026-10-10:** `act --dryrun` resolves clean. Full `act` run
  BLOCKED by a local environment fault (buildkit containers cannot resolve
  registry-1.docker.io; host DNS + `docker pull` fine; evidence
  `/tmp/act-native*.log`) — validated instead via the documented
  direct-Docker equivalent (same image, verbatim job script): green,
  14/14 (`engine_` 11 + `NativeInferTest` + both parities). The equivalent
  run caught and fixed THREE real defects before push: missing `onnxscript`
  pin, missing system `libgl1` + `libglib2.0-0` for opencv-python, and the
  `NEURIPLO_NATIVE_FIXTURE` pre-provisioning the container needs (tests
  fail-by-design without it). No repo-code change was needed for any of
  them — workflow-only fixes, which is exactly what this packet owns.

### Packet Y11 — real-image threshold-gated parity

Y9 + /tmp column dump (2026-10-10) proved the noise-input elementwise bar
meaningless: scores match to 5.2e-07 while background-cell selection (~1e-5
scores, 1e-9 deltas) picks different cells per backend. Maintainer decision
2026-10-10: parity on a REAL image, gated to confident detections, matched by
IoU/class. The background tail is explicitly out of scope
(backend-noise-dependent); [R5] is restated, not lowered — confident
detections must agree.

- **Writable:** `backends/native/test/YoloParityTest.cpp`,
  `backends/native/test/provision_yolo_fixture.py` (extend: image + tensor
  provisioning), `backends/native/test/CMakeLists.txt` (only if wiring
  changes; report first use). Never: `specs/**`, `engine/**`, anything else.
  No new C++ dependency: image decode/resize/normalize happens in the
  provision script (PIL via the already-pinned torch/ultralytics env), the
  test reads a float bin.
- **Read-before-write (mandatory):** current `YoloParityTest.cpp` (Y9 gates —
  they are REPLACED, not extended), `provision_yolo_fixture.py` (pin/image
  conventions, `/workspace` refusal, allowlist gate).
- **Required final state:**
  1. Provisioning exports, alongside `yolo26n.onnx`, a real photo
     (Ultralytics sample, e.g. `bus.jpg`, pinned URL + sha256, fail loudly
     on mismatch — same discipline as the `.pt` download) AND a
     deterministic `1×3×640×640` float input tensor (resize + `/255`,
     documented) saved as a raw bin; CMake target provisions both into the
     build tree; test reads the bin via a new env var (fixture missing →
     fail, never skip).
  2. Test runs the identical tensor through NATIVE and ONNX_RUNTIME, takes
     rows with score > 0.25 on EACH side, and requires: same confident-row
     count on both sides; each ORT row matched to a NATIVE row with same
     class and IoU ≥ 0.999; matched score diff ≤ 1e-4. Prints
     `YOLO_PARITY_CONFIDENT_ROWS=<n>`, `YOLO_PARITY_MIN_IOU=<v>`,
     `YOLO_PARITY_MAX_SCORE_DIFF=<v>`.
  3. If confident-row counts differ or any match fails → report `fail` with
     the three figures + per-row dump (retune nothing; that is orchestrator
     work). If the image yields ZERO confident rows → `fail` (wrong image,
     not a pass).
  4. The Y9 noise-input gates are removed (record the removal in the
     handback). ResNet `native_parity` untouched.
- **Budget:** 20 turns. **Handback:** `GROUP Y11 HANDBACK pass|fail`, the
  three figures, evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-parity --target native_yolo_parity_fixture --parallel 6 && cmake --build build-parity --parallel 6 && ctest --test-dir build-parity -R yolo_parity --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y10-repair — correct the unequal-stride expectation

Y10 attempt applied the right one-line fix but wrote a wrong hand-computation
(orchestrator re-trace 2026-10-09): for output `[2,1,3]` on data `[2,3,4]`,
ONNX defines `y[pos] = data[pos with axis-coord ← k]`, i.e. offset =
`r*12 + k*4 + c`. The correct vector is `{0,5,10,20,17,14}`, not the
committed `{0,4,8,20,17,12}` (which drops `+c`) and not the attempt's
`{0,4,8,20,16,12}` guess. Pre-fix line yields `{0,4,8,18,16,14}` — the test
still discriminates on 4 of 6 elements.

- **Writable:** `engine/test/KernelsTest.cpp` (expectation + comment only),
  `engine/src/kernels/GatherElements.cpp` (only if the failure output proves
  the fix line itself wrong — report first use; default: do NOT touch, the
  line is verified by re-trace). Never: `specs/**`, anything else.
- **Required final state:** expectation `{0,5,10,20,17,14}` with the `r*12 +
  k*4 + c` comment; if the actual test output disagrees with this vector,
  STOP and report `fail` with actual-vs-expected (do not force-fit — that
  means the re-trace is wrong and it is orchestrator work). Full `engine_`
  suite green.
- **Budget:** 5 turns. **Handback:** `GROUP Y10-REPAIR HANDBACK pass|fail`,
  actual test output lines, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y9 — order-invariant parity + non-degenerate input

Y4-retry-4 scratch: 633.003 elementwise on the degenerate `0.01*((i%7)+1)`
input (7 distinct values over 1.2M pixels → massive TopK ties over 8400
cells; tie order spec-silent, native lower-index-wins). Maintainer decision
2026-10-09: both fixes — realistic input AND order-invariant comparison. The
1e-4 bar is restated, not lowered.

- **Writable:** `backends/native/test/YoloParityTest.cpp`,
  `backends/native/test/provision_yolo_fixture.py` (only if the input pattern
  lives there — verify first), `backends/native/test/CMakeLists.txt` (only if
  wiring changes are needed; report first use). Never: `specs/**`,
  `engine/**`, anything else.
- **Read-before-write (mandatory):** current `YoloParityTest.cpp` (input
  pattern, compare block, `kMaxAllowedAbsDiff`), the ResNet `ParityTest.cpp`
  precedent, TopK tie semantics (`engine/src/kernels/TopK.cpp:130-134`,
  read-only — do NOT modify the kernel in this packet).
- **Required final state:**
  1. Non-degenerate deterministic input: distinct values across the
     `1x3x640x640` tensor (seeded PRNG or irrational-slope pattern —
     document the choice in a comment), same buffer fed to both backends.
     The old 7-value tiled pattern is gone.
  2. Elementwise max-abs-diff ≤ 1e-4 stays the primary gate, printed as
     `YOLO_PARITY_MAX_ABS_DIFF=<v>`.
  3. Order-invariant secondary: sort both `[300,6]` outputs by score column
     (or greedy nearest-row match — document the choice) and print
     `YOLO_PARITY_MATCHED_MAX_ABS_DIFF=<v>`; it must also be ≤ 1e-4.
     If primary fails but secondary passes → stop, report `fail` with both
     figures (tie-break pin becomes orchestrator work — retune nothing).
  4. Missing fixture fails, never skips (unchanged).
- **Budget:** 20 turns. **Handback:** `GROUP Y9 HANDBACK pass|fail`, both
  figures, evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-parity --target native_yolo_parity_fixture --parallel 6 && cmake --build build-parity --parallel 6 && ctest --test-dir build-parity -R yolo_parity --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y4-retry-4 — parity run on the green tree

Y8 fixed the last known load/run blocker (scratch full NATIVE inference
reaches `[1,300,6]` with 0 non-finite). This packet runs the committed Y4
harness to a figure — no code changes unless the run exposes a harness
defect (same latitude + reporting duty as previous retries).

- **Writable:** the three Y4 harness files, harness-defect fixes only.
  Never: `specs/**`, `engine/**`, anything else.
- **Required final state:** green chain with `YOLO_PARITY_MAX_ABS_DIFF`. Over
  1e-4 → stop, report `fail` with the figure + per-region stats, retune
  nothing.
- **Budget:** 20 turns. **Handback:** `GROUP Y4-RETRY-4 HANDBACK pass|fail`,
  observed figure, evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-parity --target native_yolo_parity_fixture --parallel 6 && cmake --build build-parity --parallel 6 && ctest --test-dir build-parity -R yolo_parity --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y8 — data-movement dtype propagation audit

Y4-retry-3 failed at runtime: `Unsqueeze (node '/model.23/Unsqueeze_7'):
output dtype must match the input dtype`. Root cause (orchestrator,
read-only): `/model.23/Unsqueeze_7` moves TopK int64 output, but the
`unsqueeze_shape` rule hardcodes float32 while the kernel (correctly) demands
output dtype == data dtype. Same latent class as the Gather deviation Y5
accepted and the GatherElements observation Y5-repair recorded — now audited
everywhere instead of one by one.

- **Writable:** `engine/src/ShapeInference.cpp` (data-movement arms),
  `engine/src/kernels/*.cpp` ONLY where the fixture exercises a dtype the
  kernel rejects (each widening reported in the handback; prefer rule-side
  fixes), `engine/test/ShapeInferenceTest.cpp`, `engine/test/KernelsTest.cpp`
  (extend in place). Never: `specs/**`, loader, `Graph.hpp`, Device/Executor/
  Planner, `backends/**`, `cmake/**`, `scripts/**`.
- **Required final state:**
  1. Audit table (handback evidence): for every supported op, the shape-rule
     output dtype(s) vs the kernel's accepted input dtype(s). Mark each
     agree/disagree.
  2. Every data-movement rule (`flatten, concat, split, unsqueeze, expand,
     transpose, slice, gather, gatherelements, resize, reshape`) propagates
     the data dtype instead of hardcoding float32, unless its kernel is
     genuinely float32-only AND the fixture never feeds it another dtype —
     those cases stay as-is and are listed explicitly with the justification.
  3. Kernel widenings only where the fixture demands (int64 data paths must
     already exist or be added; bool data stays rejected everywhere — no bool
     data movement in the fixture; verify this claim against the export,
     do not assume it).
  4. Hermetic tests: int64 data through each retyped rule + kernel
     (concat/split/unsqueeze/transpose/slice/gather/expand/flatten/resize),
     plus agreement-rejection negatives where rules enforce them.
  5. Full-file verification with a /tmp-only scratch driver (never committed):
     InferShapes AND full inference to the `[1,300,6]` output on the oracle
     fixture — report output shape, finiteness, and first/last values.
     (Deeper numeric parity stays Y4's job; this only proves the run completes.)
- **Budget:** 20 turns. **Handback:** `GROUP Y8 HANDBACK pass|fail`, audit
  table, evidence per obligation, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

### Packet Y4-retry-3 — re-run parity after Y7

Y7 fixed the missing-buffer throw that stopped the last parity run at
`/model.23/TopK`. Same terms as Y4-retry-2 (harness-defect latitude +
reporting duty, stop-and-report over 1e-4, retune nothing), same verbatim
acceptance chain.

- **Writable:** the three Y4 harness files, harness-defect fixes only.
  Never: `specs/**`, `engine/**`, anything else.
- **Budget:** 20 turns. **Handback:** `GROUP Y4-RETRY-3 HANDBACK pass|fail`,
  observed `YOLO_PARITY_MAX_ABS_DIFF`, evidence, deviations, NO-GO,
  `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-parity --target native_yolo_parity_fixture --parallel 6 && cmake --build build-parity --parallel 6 && ctest --test-dir build-parity -R yolo_parity --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

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

### Packet Y7 — plan buffers for unconsumed node outputs

Y4-retry-2 failed at runtime: `executor: no arena buffer for output
'/model.23/TopK_output_0' of node '/model.23/TopK'`. Root cause (orchestrator,
read-only): `MemoryPlanner.cpp:158-161` gives no buffer to a defined tensor
that is never consumed and is not a graph output. TopK's values output is
exactly that (only the indices flow onward), but the executor unconditionally
resolves a buffer for every node output before dispatching the kernel — the
kernel writes both outputs, so "no storage" is not an option. The skip
optimization is invalid under the executor contract; ResNet never tripped it
because every output there is consumed.

- **Writable:** `engine/src/MemoryPlanner.cpp`,
  `engine/test/MemoryPlannerTest.cpp` (extend in place). Never: `specs/**`,
  executor, kernels, shape rule, loader, `Graph.hpp`, `backends/**`,
  `cmake/**`, `scripts/**`.
- **Required final state:** every node output defined in the graph gets a
  planned buffer (remove the dead-tensor skip; an unconsumed non-output lives
  `[definition, definition]`). Reuse is preserved (existing
  `SequentialArenaSmallerThanSum`-class tests must stay green). New tests: a
  dead-output tensor gets a buffer; a two-output node with one dead output
  (TopK-shaped) plans both.
- **Budget:** 8 turns. **Handback:** `GROUP Y7 HANDBACK pass|fail`, evidence,
  deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
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
| 13 | Y4-retry-2 (validate) | Validator | opencode `general` subagent | — | — | Fail (harness sound; executor has no buffer for unconsumed TopK values output) | 0 | Planner dead-tensor skip invalid under executor contract → Y7 |
| 14 | Y7 (implement) | Implementer | opencode `general` subagent | — | — | Pass (orchestrator re-scored: 2 files in scope, plan+executor 4/4 green) | 0 | Dead tensors live `[definition, definition]`; TopK-shaped both-outputs test; reuse intact |
| 15 | Y4-retry-3 (validate) | Validator | opencode `general` subagent | — | — | Fail (harness sound; Unsqueeze rule hardcodes f32, kernel demands dtype match on int64 data) | 0 | Rule/kernel dtype class bug → Y8 audit |
| 16 | Y8 (implement) | Implementer | opencode `general` subagent | — | — | Pass (orchestrator re-scored: 2 files in scope, `engine_` 11/11 green) | 1 (Concat agreement rejection added — required, accepted) | 9 rules propagate data dtype; reshape/TopK-values stay f32 (justified); zero kernel widenings; scratch full NATIVE run → [1,300,6], 0 non-finite |
| 17 | Y4-retry-4 (validate) | Validator | implementer subagent | — | — | Incomplete (step ceiling before the single acceptance run; TEMP-DIAG edit reverted by orchestrator, tree clean) | 0 | Scratch pre-flight: max diff 633.003 / mean 69.7 / 318-1800 exact / 0 non-finite. Two read-only explores converge: ordering-fragile harness (TopK tie-order spec-silent + 7-distinct-value degenerate input → permuted box rows, 633≈640) + latent real `GatherElements.cpp:62` stride bug (dormant here). Replan: Y9 parity-method fix, Y10 stride fix |
| 18 | Y10 (implement) | Implementer | implementer subagent | — | — | Incomplete (step ceiling; fix + test applied, acceptance never ran; worker's 17-vs-16 self-diagnosis wrong per orchestrator re-trace) | 0 | One-line stride fix correct; test expectation wrong (dropped `+c`). Repaired in Y10-repair |
| 19 | Y10-repair (implement) | Implementer | implementer subagent | — | — | Pass (orchestrator re-trace `{0,5,10,20,17,14}` confirmed by actual, no force-fit; `engine_` 11/11 green, format clean) | 0 | GatherElements axis-stride fixed with unequal-stride proof; pre-fix yields `{0,4,8,18,16,14}` on the same case |
| 20 | Y9 (implement) | Implementer | implementer subagent | — | — | Incomplete (step ceiling; harness written — splitmix64 distinct input + score-sorted secondary — acceptance never ran) | 0 | Scratch: elementwise AND matched diffs BOTH 635.463. Orchestrator /tmp column dump refutes tie hypothesis: score col ≤5.2e-07, diverging rows all near-zero scores (~2–4e-05, 1e-9 deltas) with different classes ⇒ TopK near-tie background-cell selection, not box decoding. Open: Y11 threshold-gated/real-image parity (maintainer decision) |
| 21 | Y11 (implement+score) | Implementer + Orchestrator | implementer subagent + muse-spark | — | — | Pass (worker wrote harness; orchestrator ran verbatim acceptance once: configure + fixture + build + `yolo_parity` 96s + format all green) | 1 (orchestrator clang-format pass on the new test before scoring — mechanical, no logic change) | bus.jpg real-image parity: 6 confident rows, min IoU 0.999999, max score diff 2.7e-06. Y9 noise gates removed. Hold-lift proposed (maintainer's call post-review) |
